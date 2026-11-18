"""Train the image-free KV updater against a frozen original Pi0.5.

Run on ROSE, never on the local Mac.  The action loss deliberately keeps the
student expert forward differentiable even though every expert weight is frozen.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext, suppress
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import socket
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from openpi.dual.storage import wait_for_space, metrics as storage_metrics
from openpi.dual.config import UpdaterConfig
from openpi.dual.data import DEFAULT_DATA, MixedEpisodeDataset, RawEpisodeDataset, RecordedEpisodeDataset, merge_split_lineage
from openpi.dual.history import condition_features, timed_condition_features
from openpi.dual.model import SelfUpdater, UpdaterNorm, masked_kv_loss


def _sha256(path: Path) -> str:
    if os.environ.get("PI_FS_SKIP_HASH_VERIFICATION") == "1": return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_save(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def emit_record(output: Path, record: dict, wandb_run=None) -> None:
    """Persist each event, using optimizer step as a custom W&B x-axis.

    W&B's internal row step advances on every log() call. Training and
    validation may therefore share train_step without dropping the second row.
    Never pass the optimizer step as wandb.log(step=...) here.
    """
    record = dict(record, train_step=record["step"])
    with (output / "log.jsonl").open("a") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(record, allow_nan=False), flush=True)
    if wandb_run is not None:
        wandb_run.log(record)


def _wandb_resume_kwargs(args, output: Path, prior) -> dict:
    if not args.resume:
        return {"id": args.wandb_id, "resume": "never"} if args.wandb_id else {}
    receipt = output / "wandb.json"
    saved = json.loads(receipt.read_text()) if receipt.is_file() else (prior or {}).get("provenance", {}).get("wandb")
    if not saved:
        if args.wandb_id:
            return {"id": args.wandb_id, "resume": "must"}
        raise ValueError("online W&B resume requires the previous run identity; use --wandb-id or --init-ckpt")
    if args.wandb_id and args.wandb_id != saved["id"]:
        raise ValueError("--wandb-id differs from the previous run identity")
    for field in ("entity", "project"):
        requested = getattr(args, "wandb_" + field)
        if saved.get(field) and requested and saved[field] != requested:
            raise ValueError(f"W&B {field} changed during exact resume")
    return {"id": saved["id"], "resume": "must"}


def _checkpoint_identity(path: str) -> dict:
    root = Path(path).resolve()
    files = [root / "model.safetensors", *sorted((root / "assets").rglob("*.json"))]
    if len(files) < 2 or not all(item.is_file() for item in files):
        raise ValueError("base checkpoint must contain original weights and normalization assets")
    return {"path": str(root), "files": {str(item.relative_to(root)): _sha256(item) for item in files}}


def estimate_norm(dataset: RawEpisodeDataset, bridge, config: UpdaterConfig, pairs: int, seed: int,
                  age_lo: int = 20, age_hi: int = 39) -> UpdaterNorm:
    if pairs < 1:
        raise ValueError("normalization needs at least one training pair")
    mean, sd = dataset.state_statistics()
    rng = np.random.default_rng(seed)
    kv_sum = torch.zeros((2, config.layers), device=bridge.device, dtype=torch.float64)
    delta_sum = torch.zeros_like(kv_sum)
    count = 0
    with torch.no_grad():
        for _ in range(pairs):
            for attempt in range(10_000):
                batch = dataset.sample_pairs(rng, 1, 1)
                if age_lo <= int(batch.ages[0]) <= age_hi:
                    break
            else:
                raise ValueError("dataset cannot supply normalization pairs in the requested age range")
            old, now = bridge.prefix(batch.snapshots), bridge.prefix(batch.current)
            if not bridge.dynamic_prefix_mask and not torch.equal(old.mask, now.mask):
                raise ValueError("teacher mask changes inside an episode")
            if bridge.dynamic_prefix_mask:
                old.k = torch.where(old.mask[:, None, :, None], old.k, 0.)
                old.v = torch.where(old.mask[:, None, :, None], old.v, 0.)
            mask = now.mask[:, None, :, None]
            count += int(now.mask.sum()) * config.head_dim
            for index, (a, b) in enumerate(((old.k, now.k), (old.v, now.v))):
                a, b = a.double(), b.double()
                kv_sum[index] += (((a.square() + b.square()) * .5) * mask).sum((0, 2, 3))
                delta_sum[index] += ((b - a).square() * mask).sum((0, 2, 3))
    return UpdaterNorm(
        kv_sd=(kv_sum / count).sqrt().float().clamp_min(1e-4).cpu(),
        dkv_sd=(delta_sum / count).sqrt().float().clamp_min(1e-4).cpu(),
        state_mu=torch.from_numpy(mean), state_sd=torch.from_numpy(sd),
    )


def _batch_tensors(batch, device) -> dict:
    return {name: torch.as_tensor(getattr(batch, name), device=device) for name in (
        "snapshot_for_pair", "snapshot_state", "current_state", "states_after", "executed_actions", "ages",
    )}


def compute_losses(updater, bridge, batch, *, noise_generator, action_weight: float = 2., grip_weight: float = .1,
                   kv_weight: float = 1., with_stale: bool = False, pair_metrics: dict | None = None):
    """Teacher labels use no_grad; the student hand MUST remain in this graph."""
    module = updater.module if isinstance(updater, DistributedDataParallel) else updater
    tensors = _batch_tensors(batch, bridge.device)
    with torch.no_grad():
        source, target = bridge.prefix(batch.snapshots), bridge.prefix(batch.current)
        index = tensors["snapshot_for_pair"]
        k0, v0, old_mask = source.k[index], source.v[index], source.mask[index]
        if not bridge.dynamic_prefix_mask and not torch.equal(old_mask, target.mask):
            raise ValueError("pair has different image/text masks; prompt must remain constant")
        output_mask = target.mask if bridge.dynamic_prefix_mask else old_mask
        features = condition_features(tensors["snapshot_state"], tensors["current_state"], tensors["states_after"],
                                      tensors["executed_actions"], tensors["ages"], module.norm, module.config.history)
        if module.config.time_conditioned:
            features=timed_condition_features(tensors["snapshot_state"],tensors["current_state"],tensors["states_after"],tensors["executed_actions"],tensors["ages"],module.norm,module.config.history,torch.as_tensor(batch.interval_s,device=bridge.device))
        noise = torch.randn((len(batch.ages), 10, 32), device=bridge.device, generator=noise_generator)
        start_time = 1.
        use_cache = (getattr(bridge,"cache_probability",0.) > 0 and
                     float(torch.rand((),device=bridge.device,generator=noise_generator)) < bridge.cache_probability)
        use_cache = use_cache and bool(np.all(batch.t1 >= getattr(bridge,"cache_replan_steps",5)))
        use_cache=use_cache and bool(np.all(getattr(batch,'cache_valid',True)))
        if use_cache:
            if batch.previous is None: raise ValueError("cache boundary requires previous observations")
            previous=bridge.prefix(batch.previous)
            prior_noise=torch.randn(noise.shape,device=bridge.device,generator=noise_generator)
            endpoint=bridge.hand(previous.k,previous.v,previous.mask,previous.state,prior_noise,num_steps=1)
            endpoint=bridge.rebase_endpoint(endpoint, previous.state, target.state)
            shift=bridge.cache_replan_steps
            shifted=torch.cat((endpoint[:,shift:],endpoint[:,-1:].expand(-1,shift,-1)),dim=1)
            start_time=bridge.cache_restart_time
            noise=(1-start_time)*shifted + start_time*noise
        target_actions = bridge.hand(target.k, target.v, target.mask, target.state, noise, num_steps=1,start_time=start_time)
    # Mixed precision is confined to the lightweight updater. The frozen
    # teacher/expert keep exactly the same numerical path as deployment.
    with torch.autocast(device_type=bridge.device.type, dtype=torch.bfloat16, enabled=bridge.device.type == "cuda"):
        if bridge.dynamic_prefix_mask:
            pred_k, pred_v = updater(k0, v0, old_mask, features, target.mask)
        else:
            pred_k, pred_v = updater(k0, v0, old_mask, features)
    # This call is outside no_grad: expert parameters stay frozen, KV input does not.
    predicted_actions = bridge.hand(pred_k, pred_v, output_mask, target.state, noise, num_steps=1,start_time=start_time)
    error = (predicted_actions[..., :7].float() - target_actions[..., :7].float()).square()
    weights = torch.tensor([1.] * 6 + [grip_weight], device=error.device)
    act_loss = (error * weights).mean()
    kv_loss = masked_kv_loss(pred_k, pred_v, target.k, target.v, target.mask, module.norm)
    loss = kv_weight * kv_loss + action_weight * act_loss
    metrics = {"loss": loss.detach(), "kv_pred": kv_loss.detach(), "action_weighted": act_loss.detach(),
               "cache_boundary": torch.tensor(float(use_cache),device=error.device),
               "pred": error.mean().detach(), "pred_arm": error[..., :6].mean().detach(),
               "pred_grip": error[..., 6].mean().detach(), "age_mean": tensors["ages"].float().mean()}
    if pair_metrics is not None:
        with torch.no_grad():
            if hasattr(batch,'interval_s'):
                pair_metrics.update(elapsed_ms=torch.as_tensor(batch.interval_s.sum(1)*1000,device=error.device),target_observed=torch.as_tensor(batch.target_observed,device=error.device))
            pair_metrics.update(pred=error.mean((1, 2)).detach(), pred_arm=error[..., :6].mean((1, 2)).detach(),
                                pred_grip=error[..., 6].mean(1).detach(),
                                action_weighted=(error * weights).mean((1, 2)).detach(),
                                kv_pred=_pair_kv_loss(pred_k, pred_v, target.k, target.v, target.mask, module.norm))
    if with_stale:
        with torch.no_grad():
            stale = bridge.hand(k0, v0, old_mask, target.state, noise, num_steps=1,start_time=start_time)
            stale_error = (stale[..., :7].float() - target_actions[..., :7].float()).square()
            metrics.update(stale=stale_error.mean(), stale_arm=stale_error[..., :6].mean(),
                           stale_grip=stale_error[..., 6].mean(),
                           kv_stale=masked_kv_loss(k0, v0, target.k, target.v, target.mask, module.norm))
            if pair_metrics is not None:
                pair_metrics.update(stale=stale_error.mean((1, 2)), stale_arm=stale_error[..., :6].mean((1, 2)),
                                    stale_grip=stale_error[..., 6].mean(1),
                                    kv_stale=_pair_kv_loss(k0, v0, target.k, target.v, target.mask, module.norm))
    return loss, metrics


@torch.no_grad()
def _pair_kv_loss(k, v, target_k, target_v, mask, norm):
    """Per-pair normalized KV error for age strata; never part of the objective."""
    total = torch.zeros(k.shape[0], dtype=torch.float32, device=k.device)
    for index, (predicted, target) in enumerate(((k, target_k), (v, target_v))):
        residual = (predicted.float() - target.float()) / norm.dkv_sd[index][None, :, None, None]
        residual = torch.where(mask[:, None, :, None], residual, 0.)
        total += residual.square().sum((1, 2, 3))
    denominator = 2 * mask.sum(1) * k.shape[1] * k.shape[3]
    return total / denominator


def evaluate(updater, bridge, dataset, args) -> dict:
    """Fixed held-out pair/noise seeds every time, independent of training RNG."""
    updater.eval()
    rng = np.random.default_rng(args.seed + 100_000)
    generator = torch.Generator(device=bridge.device).manual_seed(args.seed + 200_000)
    totals = {}
    histogram = np.zeros(args.history + 1, dtype=np.int64)
    task_totals = {}
    observed_totals = {True: {"count":0,"sums":{}}, False:{"count":0,"sums":{}}}
    buckets = [(lo, min(hi, args.history)) for lo, hi in ((1, 4), (5, 9), (10, 19), (20, 29), (30, 39), (40, args.history))
               if lo <= args.history]
    bucket_totals = {(lo, hi): {"count": 0, "sums": {}} for lo, hi in buckets}
    count = 0
    with torch.no_grad():
        while count < args.eval_pairs:
            n = min(args.targets_per_snapshot, args.eval_pairs - count)
            if args.task_balanced:
                def assign_task(source):
                    if isinstance(source,MixedEpisodeDataset):
                        for child in source.sources: assign_task(child)
                    else:
                        tasks = (sorted({source.episodes[i]["task_id"] for i in source.ids})
                                 if args.robot_profile == "fr3" else list(range(10)))
                        source.validation_task_id=tasks[(count // args.targets_per_snapshot) % len(tasks)]
                assign_task(dataset)
            batch = dataset.sample_pairs(rng, 1, n)
            per_pair = {}
            _, metrics = compute_losses(updater, bridge, batch, noise_generator=generator,
                                       action_weight=args.action_weight, grip_weight=args.grip_weight,
                                       kv_weight=args.kv_weight, with_stale=True, pair_metrics=per_pair)
            for name, value in metrics.items():
                totals[name] = totals.get(name, 0.) + float(value) * n
            histogram += np.bincount(batch.ages, minlength=args.history + 1)
            pair_values = {name: values.detach().float().cpu().numpy() for name, values in per_pair.items()}
            if hasattr(batch,'target_observed'):
                for real in (True,False):
                    chosen=batch.target_observed==real;record=observed_totals[real];record['count']+=int(chosen.sum())
                    for name,values in pair_values.items():record['sums'][name]=record['sums'].get(name,0.)+float(values[chosen].sum())
            if args.robot_profile == "fr3":
                pair_tasks = np.array([dataset.episodes[int(eid)]["task_id"] for eid in batch.episode_ids])
                for task in np.unique(pair_tasks):
                    chosen_task = pair_tasks == task
                    task_record = task_totals.setdefault(int(task), {"count": 0, "sums": {}})
                    task_record["count"] += int(chosen_task.sum())
                    for name, values in pair_values.items():
                        task_record["sums"][name] = task_record["sums"].get(name, 0.) + float(values[chosen_task].sum())
            for lo, hi in buckets:
                chosen = (batch.ages >= lo) & (batch.ages <= hi)
                record = bucket_totals[(lo, hi)]
                record["count"] += int(chosen.sum())
                for name, values in pair_values.items():
                    record["sums"][name] = record["sums"].get(name, 0.) + float(values[chosen].sum())
            count += n
    updater.train()
    if not count:
        return {}
    result = {"val/" + name: value / count for name, value in totals.items()}
    result.update({f"val/age_count/age_{age:02d}": int(value) for age, value in enumerate(histogram)})
    for (lo, hi), record in bucket_totals.items():
        prefix = f"val/ages_{lo:02d}_{hi:02d}/"
        result[prefix + "pairs"] = record["count"]
        if record["count"]:
            result.update({prefix + name: value / record["count"] for name, value in record["sums"].items()})
    for real,record in observed_totals.items():
        prefix='val/'+('recorded_target/' if real else 'held_frame_target/')
        result[prefix+'pairs']=record['count']
        if record['count']:result.update({prefix+name:value/record['count'] for name,value in record['sums'].items()})
    for task, record in sorted(task_totals.items()):
        result[f"val/task_{task:02d}/pairs"] = record["count"]
        result.update({f"val/task_{task:02d}/{name}": value / record["count"] for name, value in record["sums"].items()})
    return result


def accumulate_microbatch_gradients(updater, bridge, dataset, rng, noise_generator, args):
    """Accumulate independent microbatches into one optimizer-update gradient.

    Caller zeroes once before this function and clips/steps once afterwards.
    The objective is the mean of microbatch objectives, matching equal-weight
    DDP rank means. KV normalization remains local to each microbatch, exactly
    as it is local to each DDP rank; it is not redefined as one concatenated
    token-weighted KV loss. Teacher/student noise remains paired inside each
    microbatch, and different microbatches draw fresh pairs and fresh noise.
    """
    metrics = {}
    histogram = np.zeros(args.history + 1, dtype=np.int64)
    for micro_step in range(args.grad_accum_steps):
        batch = dataset.sample_pairs(rng, args.batch_snapshots, args.targets_per_snapshot)
        histogram += np.bincount(batch.ages, minlength=args.history + 1)
        # DDP requires both forward and backward to be inside no_sync.
        defer_sync = isinstance(updater, DistributedDataParallel) and micro_step + 1 < args.grad_accum_steps
        context = updater.no_sync() if defer_sync else nullcontext()
        with context:
            loss, micro_metrics = compute_losses(
                updater, bridge, batch, noise_generator=noise_generator,
                action_weight=args.action_weight, grip_weight=args.grip_weight, kv_weight=args.kv_weight)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite microbatch training loss")
            (loss / args.grad_accum_steps).backward()
        for name, value in micro_metrics.items():
            contribution = value.detach() / args.grad_accum_steps
            metrics[name] = metrics.get(name, 0.) + contribution
        # Do not retain an expert computation graph across microbatches.
        del loss, micro_metrics, batch
    return metrics, histogram


def pair_exposure_plan(args, prior, world):
    """Count processed training pairs, keeping unknown older lineage explicit."""
    effective = args.batch_snapshots * args.targets_per_snapshot * args.grad_accum_steps * world
    before, basis = 0, "new training lineage"
    if prior is not None:
        previous = prior.get("pair_exposure")
        if previous is not None:
            before = previous.get("before_cycle" if args.resume else "completed_cumulative")
            basis = "recorded checkpoint pair exposure"
        elif prior["global_step"] == prior["cycle_step"]:
            previous_args = prior["train_args"]
            previous_effective = (previous_args["batch_snapshots"] * previous_args["targets_per_snapshot"]
                                  * previous_args.get("grad_accum_steps", 1) * prior["world_size"])
            before = 0 if args.resume else prior["global_step"] * previous_effective
            basis = "exact legacy first-cycle count from completed optimizer steps, batch and world size"
        else:
            before, basis = None, "unknown older multi-cycle lineage; do not infer exposures from current batch size"
    return {"before_cycle": before, "before_cycle_count_basis": basis,
            "pairs_per_optimizer_step": effective,
            "planned_cycle": args.steps * effective,
            "planned_cumulative": None if before is None else before + args.steps * effective,
            "scope": "model-lineage training pair exposures, including resampling; excludes discarded/replayed attempts, unique-pair claims and validation/diagnostic pairs"}


def completed_pair_exposure(plan, cycle_step):
    completed = cycle_step * plan["pairs_per_optimizer_step"]
    return {"before_cycle": plan["before_cycle"], "completed_in_cycle": completed,
            "completed_cumulative": (None if plan["before_cycle"] is None else plan["before_cycle"] + completed),
            "pairs_per_optimizer_step": plan["pairs_per_optimizer_step"],
            "before_cycle_count_basis": plan["before_cycle_count_basis"], "scope": plan["scope"]}


def _rng_state(rng, noise_generator):
    return {"numpy_pairs": rng.bit_generator.state, "noise": noise_generator.get_state().cpu(),
            "python": random.getstate(), "numpy_global": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(state, rng, noise_generator):
    rng.bit_generator.state = state["numpy_pairs"]
    noise_generator.set_state(state["noise"])
    random.setstate(state["python"])
    np.random.set_state(state["numpy_global"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def _save_checkpoint(path, updater, optimizer, rng, noise_generator, step, total_before, args, provenance, rank, world, selection=None):
    # All ranks wait before collectives, avoiding a rank-0-only storage stall.
    wait_for_space(path.parent, 1024**3,
                   event_path=path.parent / "storage_events.jsonl" if rank == 0 else None)
    state = _rng_state(rng, noise_generator)
    states = [None] * world if rank == 0 else None
    if world > 1:
        dist.gather_object(state, states, dst=0)
    else:
        states = [state]
    if rank == 0:
        payload = {
            "schema_version": 1, "updater_config": asdict(updater.config), "norm": updater.norm.to_dict(),
            "updater": updater.state_dict(), "optimizer": optimizer.state_dict(), "cycle_step": step,
            "global_step": total_before + step, "rng_by_rank": states, "world_size": world,
            "train_args": vars(args), "provenance": provenance, "selection": selection,
            "pair_exposure": completed_pair_exposure(provenance["pair_exposure_plan"], step),
        }
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    if world > 1:
        dist.barrier()


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--recorded-data", action="append", default=[],
                        help="append a dual eval NPZ directory with explicit train/validation roles; repeat for multiple rounds")
    parser.add_argument("--validation-recorded-data", action="append", default=None,
                        help="fixed validation sources, independent of changing training recordings")
    parser.add_argument("--lineage-id", required=True, help="explicit experiment lineage; inherited checkpoints must match")
    parser.add_argument("--initialize-only", action="store_true",
                        help="fit training-only statistics and save a zero-update updater with empty AdamW state")
    parser.add_argument("--preserve-training-rng", action="store_true",
                        help="continue pair/noise RNG across data-collection boundaries")
    parser.add_argument("--lr-cycle-steps", type=int, default=0,
                        help="warmup/cosine period in optimizer updates; 0 uses this invocation's steps")
    parser.add_argument("--milestone-steps", default="",
                        help="comma-separated global optimizer steps to retain for equal-budget evaluation")
    parser.add_argument("--robot-profile", choices=["libero", "fr3"], default="libero")
    parser.add_argument("--frame-root", help="FR3 decoded RGB frame directory")
    parser.add_argument("--frame-source-stride",type=int,default=1)
    parser.add_argument("--base-checkpoint", default="models/pi05_libero_pytorch")
    parser.add_argument("--out", required=True)
    parser.add_argument("--prefix-backend", choices=["compact","stock"], default="stock")
    parser.add_argument("--source-weights", help="demo followed by each recorded source, comma-separated")
    parser.add_argument("--validation-source-weights")
    parser.add_argument("--task-balanced", action="store_true")
    parser.add_argument("--selection-metric", choices=["val/pred","val/action_weighted","val/recorded_target/action_weighted"], default="val/action_weighted")
    parser.add_argument("--cache-probability", type=float, default=.5)
    parser.add_argument("--cache-restart-time", type=float, default=.5)
    parser.add_argument("--cache-replan-steps", type=int, default=5)
    parser.add_argument("--task-ids", default=",".join(map(str, range(10))), help="verified ROSE LIBERO10 task ids; 'all' for all40")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=.1)
    parser.add_argument("--age-lo", type=int, default=1)
    parser.add_argument("--age-hi", type=int, default=45)
    parser.add_argument("--age-sampling", choices=["legacy", "age_first"], default="age_first",
                        help="age_first draws all target ages before selecting a supporting episode/anchor")
    parser.add_argument("--age-core-lo", type=int, default=1)
    parser.add_argument("--age-core-hi", type=int, default=10)
    parser.add_argument("--age-core-prob", type=float, default=.8,
                        help="optional probability mass on the core age interval; remaining mass is uniform outside it")
    parser.add_argument("--split-manifest", help="pin the previous LeRobot split.json when changing age support")
    parser.add_argument("--history", type=int, default=48)
    parser.add_argument("--text-tokens", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--batch-snapshots", type=int, default=2)
    parser.add_argument("--targets-per-snapshot", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=2,
                        help="independent microbatches per optimizer update; steps/warmup/save cadence remain optimizer updates")
    parser.add_argument("--steps", type=int, default=6_000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--weight-decay", type=float, default=.01)
    parser.add_argument("--grad-clip", type=float, default=1.)
    parser.add_argument("--kv-weight", type=float, default=1.)
    parser.add_argument("--action-weight", type=float, default=2.)
    parser.add_argument("--grip-weight", type=float, default=.1)
    parser.add_argument("--norm-pairs", type=int, default=16)
    parser.add_argument("--norm-age-lo", type=int, default=20)
    parser.add_argument("--norm-age-hi", type=int, default=39)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--keep-best-checkpoint", action="store_true",
                        help="retain latest.pt and best.pt by fixed held-out val/pred; compare the initial model before updates")
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-pairs", type=int, default=320)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--resume", help="exact continuation of the same cycle/data/world size")
    modes.add_argument("--init-ckpt", help="new cosine cycle; inherit updater, norms and optimizer moments")
    parser.add_argument("--wandb-project", help="enables online W&B logging; otherwise JSONL only")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-id", help="explicit W&B run id; fresh runs reject existing ids, resumes require the prior id")
    parser.add_argument("--feedback-data", help="timestamped updater-only sidecars")
    return parser


def main(argv=None):
    args = parser().parse_args(argv)
    if args.initialize_only and (args.resume or args.init_ckpt):
        raise ValueError("zero-update initialization cannot inherit a checkpoint")
    if args.lr_cycle_steps < 0:
        raise ValueError("lr-cycle-steps must be nonnegative")
    milestones = {int(x) for x in args.milestone_steps.split(',') if x}
    if any(getattr(args, key) < 1 for key in ("steps", "batch_snapshots", "targets_per_snapshot", "grad_accum_steps", "save_every", "eval_every", "log_every", "norm_pairs")):
        raise ValueError("step/batch/save/eval intervals and norm-pairs must be positive")
    if args.keep_best_checkpoint and args.eval_pairs < 1:
        raise ValueError("best checkpoint selection requires positive eval-pairs")
    if args.warmup < 0 or args.eval_pairs < 0 or args.lr <= 0 or args.grad_clip <= 0:
        raise ValueError("invalid optimizer/evaluation settings")
    if not args.age_lo <= args.norm_age_lo <= args.norm_age_hi <= args.age_hi:
        raise ValueError("normalization age range must lie within the training age range")
    if min(args.kv_weight, args.action_weight, args.grip_weight, args.weight_decay) < 0 or args.kv_weight + args.action_weight <= 0:
        raise ValueError("loss/decay weights must be nonnegative and total loss nonzero")
    if not all(math.isfinite(getattr(args, field)) for field in ("lr", "grad_clip", "kv_weight", "action_weight", "grip_weight", "weight_decay")):
        raise ValueError("optimizer/loss parameters must be finite")
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if not args.device.startswith("cuda"):
        raise ValueError("training the frozen base is server CUDA only; use dual.validate for CPU data checks")
    device = torch.device(f"cuda:{local_rank}" if world > 1 else args.device)
    if device.index is None:
        device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl")
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    rng = np.random.default_rng(args.seed + rank)
    noise_generator = torch.Generator(device=device).manual_seed(args.seed + rank + 10_000)
    task_ids = None if args.task_ids == "all" else [int(part) for part in args.task_ids.split(",")]
    dataset_options = dict(seed=args.seed, val_fraction=args.val_fraction, task_ids=task_ids,
                           age_lo=args.age_lo, age_hi=args.age_hi, history=args.history,
                           age_sampling=args.age_sampling, age_core_lo=args.age_core_lo,
                           age_core_hi=args.age_core_hi, age_core_prob=args.age_core_prob)
    if args.split_manifest:
        dataset_options["partition_manifest"] = json.loads(Path(args.split_manifest).read_text())
    if args.robot_profile == "fr3":
        if args.recorded_data or args.prefix_backend != "stock" or args.text_tokens != 200:
            raise ValueError("FR3 uses its own 7D data, stock prefix and all 200 text slots")
        dataset_options.update(robot_profile="fr3", frame_root=args.frame_root, frame_source_stride=args.frame_source_stride)
    dataset_class=RawEpisodeDataset
    if args.feedback_data:
        if args.robot_profile!='fr3':raise ValueError('Timestamped feedback is FR3-only')
        from openpi.dual.feedback_data import FeedbackDataset
        dataset_class=FeedbackDataset;dataset_options['feedback_data']=args.feedback_data
    train = dataset_class(args.data, split="train", **dataset_options)
    val = dataset_class(args.data, split="val", **dataset_options)
    if args.recorded_data:
        train = MixedEpisodeDataset([train, *(RecordedEpisodeDataset(path, split="train", **dataset_options)
                                             for path in args.recorded_data)])
    validation_roots = args.recorded_data if args.validation_recorded_data is None else args.validation_recorded_data
    if validation_roots:
        val = MixedEpisodeDataset([val, *(RecordedEpisodeDataset(path, split="val", **dataset_options)
                                         for path in validation_roots)])
    def leaves(dataset):
        return [leaf for source in dataset.sources for leaf in leaves(source)] if isinstance(dataset,MixedEpisodeDataset) else [dataset]
    if args.source_weights:
        train=MixedEpisodeDataset(leaves(train), [float(x) for x in args.source_weights.split(',')])
    if args.validation_source_weights:
        val=MixedEpisodeDataset(leaves(val), [float(x) for x in args.validation_source_weights.split(',')])
    for dataset in leaves(train)+leaves(val):
        dataset.task_balanced=args.task_balanced
        dataset.cache_replan_steps=args.cache_replan_steps if args.cache_probability else 0
    if not 0 <= args.cache_probability <= 1 or not 0 < args.cache_restart_time <= 1:
        raise ValueError("invalid cache training boundary")
    config = UpdaterConfig(history=args.history, text_tokens=args.text_tokens, hidden_dim=args.hidden_dim,
                           depth=args.depth, heads=args.heads, state_dim=7 if args.robot_profile == "fr3" else 8, time_conditioned=bool(args.feedback_data))
    output = Path(args.out).resolve()
    latest = output / "latest.pt"
    if latest.exists():
        if not args.resume:
            raise FileExistsError("output already contains a checkpoint; use --resume or a fresh output directory")
        if Path(args.resume).resolve() != latest.resolve() and _sha256(Path(args.resume)) != _sha256(latest):
            raise ValueError("output latest.pt differs from requested resume checkpoint; choose a fresh output directory")
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    if world > 1:
        dist.barrier()
    prior = None
    if args.resume or args.init_ckpt:
        prior = torch.load(args.resume or args.init_ckpt, map_location="cpu", weights_only=False)
        if args.lineage_id and prior.get("provenance", {}).get("lineage_id") != args.lineage_id:
            raise ValueError("checkpoint belongs to a different training lineage")
        if prior.get("schema_version") != 1 or asdict(UpdaterConfig(**prior["updater_config"])) != asdict(config):
            raise ValueError("updater architecture mismatch; implicit partial loading is forbidden")
        if args.resume and prior["world_size"] != world:
            raise ValueError("exact resume requires the same world size")
    lineage = merge_split_lineage({"sources": [train.split_manifest, val.split_manifest]},
                                 prior["provenance"] if prior else None)
    training_sources = {name: _sha256(Path(__file__).parent / name) for name in
                        ("bridge.py", "config.py", "data.py", "feedback_data.py", "history.py", "model.py", "rose_tfws.py", "train.py")}
    if args.resume:
        previous_sources = prior["provenance"].get("training_source_sha256")
        if previous_sources is None:
            previous_sources = {key: prior["provenance"]["source_sha256"].get(key) for key in training_sources}
        if previous_sources != training_sources:
            raise ValueError("training source changed since checkpoint; exact resume forbidden, use --init-ckpt for a new cycle")
    from openpi.dual.bridge import FrozenPi05

    if args.prefix_backend == "stock":
        from openpi.dual.stock_training import StockTrainingBridge
        FrozenPi05 = StockTrainingBridge
    bridge = FrozenPi05(args.base_checkpoint, str(device), text_tokens=config.text_tokens, robot_profile=args.robot_profile)
    bridge.cache_probability=args.cache_probability
    bridge.cache_restart_time=args.cache_restart_time
    bridge.cache_replan_steps=args.cache_replan_steps
    bridge.model.eval().requires_grad_(False)
    identity_list = [_checkpoint_identity(args.base_checkpoint) if rank == 0 else None]
    if world > 1:
        dist.broadcast_object_list(identity_list, src=0)
    identity = identity_list[0]
    same_base = (not prior or (Path(prior["provenance"]["base_checkpoint"]["path"]).resolve() == Path(args.base_checkpoint).resolve()
                 if os.environ.get("PI_FS_SKIP_HASH_VERIFICATION") == "1"
                 else prior["provenance"]["base_checkpoint"] == identity))
    if not same_base:
        raise ValueError("base weights or normalization assets changed since updater checkpoint")
    if args.resume:
        if prior["provenance"]["data"] != train.split_manifest:
            raise ValueError("exact resume requires identical dataset metadata and episode split")
        immutable = ("steps", "lr", "warmup", "weight_decay", "grad_clip", "kv_weight", "action_weight", "grip_weight",
                     "seed", "batch_snapshots", "targets_per_snapshot", "age_lo", "age_hi", "eval_pairs")
        for key in immutable:
            if prior["train_args"][key] != getattr(args, key):
                raise ValueError(f"exact resume cannot change {key}; use --init-ckpt for a new cycle")
        new_options = {"age_sampling": "legacy", "age_core_lo": 20, "age_core_hi": 39,
                       "age_core_prob": None, "keep_best_checkpoint": False, "grad_accum_steps": 1}
        for key, default in new_options.items():
            if prior["train_args"].get(key, default) != getattr(args, key):
                raise ValueError(f"exact resume cannot change {key}; use --init-ckpt for a new cycle")
    if prior:
        norm = UpdaterNorm.from_dict(prior["norm"])
    else:
        norms = [estimate_norm(train, bridge, config, args.norm_pairs, args.seed + 300_000,
                               args.norm_age_lo, args.norm_age_hi).to_dict() if rank == 0 else None]
        if world > 1:
            dist.broadcast_object_list(norms, src=0)
        norm = UpdaterNorm.from_dict(norms[0])
    module = SelfUpdater(config, norm).to(device)
    if prior:
        module.load_state_dict(prior["updater"], strict=True)
        if any(not torch.equal(value, prior["norm"][key].float()) for key, value in module.norm.to_dict().items()):
            raise ValueError("checkpoint normalization payload differs from updater buffers")
    optimizer = torch.optim.AdamW(module.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if prior:
        optimizer.load_state_dict(prior["optimizer"])
        for group in optimizer.param_groups:
            group["weight_decay"] = args.weight_decay
    updater = DistributedDataParallel(module, device_ids=[device.index]) if world > 1 else module
    start = prior["cycle_step"] if args.resume else 0
    total_before = prior["global_step"] - start if prior else 0
    if args.resume or (prior and args.preserve_training_rng):
        if prior["world_size"] != world:
            raise ValueError("preserving training RNG requires the same world size")
        _restore_rng(prior["rng_by_rank"][rank], rng, noise_generator)
    selection = prior.get("selection") if args.resume and prior else None
    if args.keep_best_checkpoint and args.resume:
        if not selection or selection.get("metric") != args.selection_metric or not (output / "best.pt").is_file():
            raise ValueError("exact resume with best retention requires this cycle's selection record and best.pt")
        # best may have advanced after the most recent periodic latest save.
        # Restore its actual receipt so a restart cannot overwrite a better
        # checkpoint using the stale score embedded in an older latest.pt.
        saved_best = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
        best_selection = saved_best.get("selection")
        best_provenance = saved_best.get("provenance", {})
        if (not best_selection or best_selection.get("metric") != args.selection_metric
                or best_selection.get("protocol") != selection.get("protocol")
                or not math.isfinite(best_selection.get("value", float("nan")))
                or saved_best.get("updater_config") != asdict(config)
                or best_provenance.get("training_source_sha256") != training_sources
                or best_provenance.get("base_checkpoint") != identity
                or best_provenance.get("data") != train.split_manifest
                or best_provenance.get("wandb") != prior["provenance"].get("wandb")
                or saved_best.get("global_step") != best_selection.get("global_step")
                or saved_best.get("cycle_step") != best_selection.get("cycle_step")
                or saved_best["global_step"] - saved_best["cycle_step"] != total_before):
            raise ValueError("best.pt belongs to a different training cycle or validation protocol")
        for key in immutable:
            if saved_best["train_args"][key] != getattr(args, key):
                raise ValueError(f"best.pt has a different {key}")
        for key, default in new_options.items():
            if saved_best["train_args"].get(key, default) != getattr(args, key):
                raise ValueError(f"best.pt has a different {key}")
        selection = best_selection
        del saved_best
    if not args.keep_best_checkpoint:
        selection = None
    exposure = pair_exposure_plan(args, prior, world)
    properties = torch.cuda.get_device_properties(device)
    device_record = {"rank": rank, "hostname": socket.gethostname(), "device": str(device),
                     "name": properties.name, "uuid": str(properties.uuid),
                     "total_memory_bytes": properties.total_memory, "torch": torch.__version__,
                     "cuda_runtime": torch.version.cuda, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                     "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    devices = [None] * world
    if world > 1:
        dist.all_gather_object(devices, device_record)
    else:
        devices = [device_record]
    provenance = {
        "created_at": datetime.now(timezone.utc).isoformat(), "base_checkpoint": identity,
        "lineage_id": args.lineage_id,
        "validation_data": val.split_manifest,
        "initialization_only": args.initialize_only,
        "base_sha256": identity["files"]["model.safetensors"],
        "data": train.split_manifest, "data_lineage": lineage, "updater_config": asdict(config), "world_size": world,
        "robot_profile": args.robot_profile,
        "base_state_input": args.robot_profile == "fr3", "base_frozen": True, "action_teacher_nfe": 1,
        "cache_training_enabled": False, "cache_training_parameters": 0,
        "updater_cache_boundary_probability": args.cache_probability,
        "cache_boundary_context": "one fresh teacher endpoint at max(0,t-C); shared shifted/renoised source; proxy for recursive deployed cache",
        "prefix_backend": args.prefix_backend, "task_balanced": args.task_balanced,
        "training_pair_sampling": "independent same-episode windows; every retained history row; no cache across optimizer steps",
        "training_action_boundary": "paired fresh/warm boundary mixture" if args.cache_probability else "paired fresh Gaussian t=1",
        "training_replan_steps_dependency": None,
        "optimization_batch": {"snapshot_groups_per_microbatch_per_rank": args.batch_snapshots,
                               "targets_per_snapshot": args.targets_per_snapshot,
                               "pairs_per_microbatch_per_rank": args.batch_snapshots * args.targets_per_snapshot,
                               "gradient_accumulation_steps": args.grad_accum_steps, "world_size": world,
                               "effective_pairs_per_optimizer_step": exposure["pairs_per_optimizer_step"],
                               "step_unit": "optimizer updates; scheduler, validation and checkpoint cadence use this unit",
                               "gradient_reduction": "mean of local microbatch objectives, then DDP mean across ranks",
                               "gradient_clipping": "once on the total accumulated gradient before optimizer.step",
                               "ddp_synchronization": "no_sync on earlier microbatch forward/backward; final microbatch synchronizes",
                               "report_comparison": "matching effective batch alone does not reproduce report data coverage, pair count or model quality"},
        "pair_exposure_plan": exposure,
        "age_sampling": train.sampling_manifest,
        "validation_protocol": {"pair_seed": args.seed + 100_000, "noise_seed": args.seed + 200_000,
                                "pairs": args.eval_pairs, "sampler": val.sampling_manifest,
                                "comparability": "compare only within identical fixed pair/noise/sampler protocol; new age sampling differs from old validation"},
        "checkpoint_retention": {"latest": True, "best": args.keep_best_checkpoint,
                                 "best_metric": args.selection_metric if args.keep_best_checkpoint else None,
                                 "max_checkpoint_files": 2 if args.keep_best_checkpoint else 1,
                                 "atomic_save_peak_files": 3 if args.keep_best_checkpoint else 2},
        "objective": {"kv_weight": args.kv_weight, "action_weight": args.action_weight, "grip_weight": args.grip_weight},
        "devices": devices, "gpu_exclusivity": "scheduler allocation; external GPU sharing not independently verified",
        "trainable_parameters": sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad),
        "action_horizon": 10, "model_action_dim": 32, "supervised_action_dims": 7,
        "history_alignment": "states[t0+1:t1+1], executed_actions[t0:t1]",
        "source_sha256": {item.name: _sha256(item) for item in sorted(Path(__file__).parent.glob("*.py"))},
        "training_source_sha256": training_sources,
        "initialization_mode": "exact_resume" if args.resume else ("new_cycle_from_checkpoint" if args.init_ckpt else "scratch"),
        "training_rng": "restored checkpoint state" if args.resume or (prior and args.preserve_training_rng) else "reset from configured seed; validation RNG independent",
        "age_histogram_scope": "counts since previous training log in this process lifetime; summed across ranks",
        "parent_checkpoint": str(Path(args.resume or args.init_ckpt).resolve()) if prior else None,
        "parent_checkpoint_sha256": _sha256(Path(args.resume or args.init_ckpt)) if prior else None,
        "timing": "CUDA-synchronized train-step wall time; DDP includes collective wait; no closed-loop SR",
    }
    wandb_run = None
    try:
        if rank == 0:
            _json_save(output / "run.json", dict(provenance=provenance, args=vars(args)))
            _json_save(output / "split.json", train.split_manifest)
            if args.keep_best_checkpoint and selection is not None:
                _json_save(output / "best.json", selection)
            if args.wandb_project:
                import wandb
                resume_kwargs = _wandb_resume_kwargs(args, output, prior)
                wandb_run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_name,
                                       config=dict(args=vars(args), provenance=provenance), mode="online",
                                       job_type="updater-training", **resume_kwargs)
                if wandb_run is None or wandb_run.settings.mode != "online":
                    raise RuntimeError("requested online W&B run did not initialize")
                wandb_run.define_metric("train_step")
                for pattern in ("train/*", "val/*", "lr", "cycle_step"):
                    wandb_run.define_metric(pattern, step_metric="train_step")
                wandb_run.summary["training_status"] = "running"
                provenance["wandb"] = {"id": wandb_run.id, "entity": wandb_run.entity,
                                        "project": wandb_run.project, "url": wandb_run.url}
                _json_save(output / "wandb.json", provenance["wandb"])
                _json_save(output / "run.json", dict(provenance=provenance, args=vars(args)))
            _json_save(output / "status.json", {"status": "running", "start_step": total_before + start,
                                                 "target_step": total_before + args.steps})
        def validate_and_select(cycle_step):
            nonlocal selection
            if world > 1:
                dist.barrier()
            decision = [None]
            if rank == 0 and args.eval_pairs:
                metrics = evaluate(module, bridge, val, args)
                score = metrics[args.selection_metric]
                if not math.isfinite(score):
                    raise FloatingPointError("nonfinite held-out action error")
                better = args.keep_best_checkpoint and (selection is None or score < selection["value"])
                if better:
                    decision[0] = {"metric": args.selection_metric, "value": score, "cycle_step": cycle_step,
                                   "global_step": total_before + cycle_step,
                                   "checkpoint": str(output / "best.pt"),
                                   "protocol": provenance["validation_protocol"]}
                record = {"step": total_before + cycle_step, "cycle_step": cycle_step, **metrics}
                if args.keep_best_checkpoint:
                    chosen = decision[0] or selection
                    record.update({"val/selected_as_best": int(better), "val/best_pred": chosen["value"],
                                   "val/best_global_step": chosen["global_step"]})
                emit_record(output, record, wandb_run)
            if world > 1:
                dist.broadcast_object_list(decision, src=0)
            if decision[0] is not None:
                selection = decision[0]
                _save_checkpoint(output / "best.pt", module, optimizer, rng, noise_generator,
                                 cycle_step, total_before, args, provenance, rank, world, selection)
                if rank == 0:
                    _json_save(output / "best.json", selection)
            if world > 1:
                dist.barrier()

        if args.initialize_only:
            validate_and_select(0)
            if optimizer.state:
                raise RuntimeError("initial updater acquired optimizer state")
            _save_checkpoint(output / "initial.pt", module, optimizer, rng, noise_generator,
                             0, 0, args, provenance, rank, world, selection)
            if rank == 0:
                _json_save(output / "status.json", {"status": "initialized", "completed_step": 0,
                    "checkpoint": str(output / "initial.pt"), "optimizer_state_entries": 0,
                    "parent_checkpoint": None, "lineage_id": args.lineage_id,
                    "pair_exposure": completed_pair_exposure(exposure, 0)})
            if wandb_run:
                wandb_run.summary["training_status"] = "initialized"
                wandb_run.summary["completed_train_steps"] = 0
                wandb_run.finish(exit_code=0)
            return
        updater.train()
        if args.keep_best_checkpoint and not args.resume:
            # Establish the unmodified parent under the new fixed protocol.
            # Validation has independent RNGs and cannot advance training pairs.
            validate_and_select(0)
        age_histogram = np.zeros(args.history + 1, dtype=np.int64)
        for step in range(start, args.steps):
            # Step is zero-based here; checkpoint cycle_step is the number completed.
            period = args.lr_cycle_steps or args.steps
            lr_step = (total_before + step) % period if args.lr_cycle_steps else step
            if args.warmup and lr_step < args.warmup:
                multiplier = (lr_step + 1) / args.warmup
            else:
                progress = (lr_step - args.warmup) / max(1, period - args.warmup - 1)
                multiplier = .5 * (1 + math.cos(math.pi * min(1., max(0., progress))))
            for group in optimizer.param_groups:
                group["lr"] = args.lr * multiplier
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            metrics, microbatch_histogram = accumulate_microbatch_gradients(
                updater, bridge, train, rng, noise_generator, args)
            age_histogram += microbatch_histogram
            norm_value = torch.nn.utils.clip_grad_norm_(module.parameters(), args.grad_clip, error_if_nonfinite=True)
            if step == start and any(parameter.grad is not None for parameter in bridge.model.parameters()):
                raise RuntimeError("frozen base acquired gradients")
            optimizer.step()
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            if (step + 1) % args.log_every == 0 or step == start or step + 1 == args.steps:
                metrics.update(grad_norm=norm_value.detach(), step_seconds=torch.tensor(elapsed, device=device))
                values = torch.stack([value.float() for value in metrics.values()])
                if world > 1:
                    dist.all_reduce(values)
                    values /= world
                age_counts = torch.as_tensor(age_histogram, device=device).clone()
                if world > 1:
                    dist.all_reduce(age_counts)
                if rank == 0:
                    record = {"step": total_before + step + 1, "cycle_step": step + 1, "lr": args.lr * multiplier,
                              **{"train/" + name: float(value) for name, value in zip(metrics, values, strict=True)},
                              "train/gpu_allocated_bytes_rank0": torch.cuda.memory_allocated(device),
                              "train/gpu_peak_allocated_bytes_rank0": torch.cuda.max_memory_allocated(device),
                              "train/effective_pairs_per_optimizer_step": exposure["pairs_per_optimizer_step"],
                              "train/grad_accum_steps": args.grad_accum_steps,
                              "train/pairs_seen_in_cycle": (step + 1) * exposure["pairs_per_optimizer_step"]}
                    if exposure["before_cycle"] is not None:
                        record["train/pairs_seen_cumulative"] = exposure["before_cycle"] + record["train/pairs_seen_in_cycle"]
                    counts = age_counts.cpu().tolist()
                    record.update({f"train/age_count/age_{age:02d}": int(value) for age, value in enumerate(counts)})
                    record["train/age_count_total"] = sum(counts)
                    record["train/age_mean_interval"] = sum(age * value for age, value in enumerate(counts)) / sum(counts)
                    record.update(storage_metrics(output))
                    emit_record(output, record, wandb_run)
                age_histogram.fill(0)
            if step == start or (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
                validate_and_select(step + 1)
            if step == start or (step + 1) % args.save_every == 0 or step + 1 == args.steps:
                _save_checkpoint(output / "latest.pt", module, optimizer, rng, noise_generator,
                                 step + 1, total_before, args, provenance, rank, world, selection)
            if total_before + step + 1 in milestones:
                _save_checkpoint(output / f"step_{total_before + step + 1:06d}.pt", module, optimizer,
                                 rng, noise_generator, step + 1, total_before, args, provenance, rank, world, selection)
        if wandb_run:
            wandb_run.summary["training_status"] = "completed"
            wandb_run.summary["completed_train_steps"] = total_before + args.steps
            wandb_run.summary["completed_train_pairs_in_cycle"] = args.steps * exposure["pairs_per_optimizer_step"]
            if exposure["planned_cumulative"] is not None:
                wandb_run.summary["completed_train_pairs_cumulative"] = exposure["planned_cumulative"]
            wandb_run.finish(exit_code=0)
        if rank == 0:
            _json_save(output / "status.json", {"status": "completed", "completed_step": total_before + args.steps,
                                                 "checkpoint": str(output / "latest.pt"), "selection": selection,
                                                 "pair_exposure": completed_pair_exposure(exposure, args.steps)})
    except BaseException as error:
        if rank == 0:
            with suppress(Exception):
                _json_save(output / "status.json", {"status": "failed", "error_type": type(error).__name__})
            if wandb_run is not None:
                with suppress(Exception):
                    wandb_run.summary["training_status"] = "failed"
                    wandb_run.summary["failure_type"] = type(error).__name__
                    wandb_run.finish(exit_code=1)
        raise
    finally:
        if world > 1 and dist.is_initialized():
            with suppress(Exception):
                dist.destroy_process_group()



if __name__ == "__main__":
    main()
