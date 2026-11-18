"""Episode-safe causal pairs from the existing LIBERO LeRobot v2 dataset.

The input is the published image/state/action parquet data, not a shuffled
OpenPI action-chunk loader.  Age is measured in consecutive *dataset rows*;
the upstream no-noops conversion does not preserve every simulator tick.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass
import hashlib
import os
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_DATA = "data/libero"


@dataclass
class PairBatch:
    snapshots: list[dict]
    current: list[dict]
    snapshot_for_pair: np.ndarray
    snapshot_state: np.ndarray
    current_state: np.ndarray
    states_after: np.ndarray
    executed_actions: np.ndarray
    ages: np.ndarray
    episode_ids: np.ndarray
    t0: np.ndarray
    t1: np.ndarray
    previous: list[dict] | None = None


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class RawEpisodeDataset:
    """A bounded parquet LRU with deterministic episode-disjoint task splits."""

    def __init__(
        self, root: str | Path = DEFAULT_DATA, *, split: str = "train", seed: int = 42,
        val_fraction: float = .1, task_ids: list[int] | None = None,
        age_lo: int = 10, age_hi: int = 45, history: int = 48, cache_episodes: int = 8,
        age_sampling: str = "legacy", age_core_lo: int = 20, age_core_hi: int = 39,
        age_core_prob: float | None = None, partition_manifest: dict | None = None,
        robot_profile: str = "libero", frame_root: str | None = None, frame_source_stride: int = 1,
    ):
        if split not in {"train", "val"}:
            raise ValueError("split must be train or val")
        if not 0 < val_fraction < 1 or not 1 <= age_lo <= age_hi <= history:
            raise ValueError("require 0 < val_fraction < 1 and 1 <= age_lo <= age_hi <= history")
        if cache_episodes < 1:
            raise ValueError("cache_episodes must be positive")
        self.root = Path(root).resolve()
        self.robot_profile = robot_profile
        self.state_dim = 7 if robot_profile == "fr3" else 8
        self.state_key = "observation.state" if robot_profile == "fr3" else "state"
        self.action_key = "action" if robot_profile == "fr3" else "actions"
        self.frame_root = Path(frame_root).resolve() if frame_root else None
        self.frame_source_stride=int(frame_source_stride)
        if self.frame_source_stride not in (1,20):raise ValueError("Unsupported image source stride")
        self.split, self.age_lo, self.age_hi, self.history = split, age_lo, age_hi, history
        self.cache_episodes = cache_episodes
        self.info = json.loads((self.root / "meta/info.json").read_text())
        if self.info.get("codebase_version") not in {"v2.0", "v2.1"}:
            raise ValueError("this reader requires a LeRobot v2 parquet episode layout")
        for key, dimension in ((self.state_key, self.state_dim), (self.action_key, 7)):
            if self.info["features"][key]["shape"] != [dimension]:
                raise ValueError(f"LIBERO {key} must have dimension {dimension}")
        self.tasks = {int(row["task_index"]): row["task"] for row in _read_jsonl(self.root / "meta/tasks.jsonl")}
        inverse = {name: key for key, name in self.tasks.items()}
        if robot_profile != "fr3" and len(inverse) != len(self.tasks):
            raise ValueError("task text must uniquely identify a task")
        selected = set(self.tasks) if task_ids is None else set(task_ids)
        if not selected or not selected.issubset(self.tasks):
            raise ValueError("task_ids must select existing task ids")
        self.episodes = {}
        by_task = defaultdict(list)
        for row in _read_jsonl(self.root / "meta/episodes.jsonl"):
            names = row["tasks"]
            if len(names) != 1 or names[0] not in inverse:
                raise ValueError("each episode must have exactly one immutable known prompt")
            eid = int(row["episode_index"])
            if robot_profile == "fr3":
                import pyarrow.parquet as pq
                task = int(pq.read_table(self.episode_path(eid), columns=["task_index"])["task_index"][0].as_py())
                if self.tasks.get(task) != names[0]:
                    raise ValueError("episode task_index and prompt disagree")
            else:
                task = inverse[names[0]]
            if task in selected and int(row["length"]) > age_lo:
                if eid in self.episodes:
                    raise ValueError(f"duplicate episode id {eid}")
                self.episodes[eid] = dict(row, task_id=task)
                by_task[task].append(eid)
        train, val = [], []
        for task in sorted(selected):
            ids = np.array(sorted(by_task[task]), dtype=np.int64)
            if robot_profile == "fr3" and len(ids) == 1:
                train.extend(map(int, ids))
                continue
            if len(ids) < 2:
                raise ValueError(f"task {task} needs at least two eligible episodes for a disjoint split")
            split_rng = np.random.default_rng(np.random.SeedSequence([seed, task]))
            split_rng.shuffle(ids)
            nval = min(len(ids) - 1, max(1, round(len(ids) * val_fraction)))
            val.extend(map(int, ids[:nval]))
            train.extend(map(int, ids[nval:]))
        if partition_manifest is not None:
            # A lower age bound must not reshuffle the existing held-out split.
            # Pin identities, then revalidate source metadata rather than trust a
            # path/name alone. Newly eligible short episodes stay excluded.
            required = {"root", "seed", "val_fraction", "task_ids", "train_episode_ids", "val_episode_ids", "metadata_sha256"}
            if not required.issubset(partition_manifest) or "sources" in partition_manifest:
                raise ValueError("partition_manifest must describe one LeRobot demonstration source")
            if (Path(partition_manifest["root"]).resolve() != self.root
                    or partition_manifest["seed"] != seed
                    or partition_manifest["val_fraction"] != val_fraction
                    or partition_manifest["task_ids"] != sorted(selected)):
                raise ValueError("pinned partition source/seed/tasks/validation fraction differs")
            train = list(map(int, partition_manifest["train_episode_ids"]))
            val = list(map(int, partition_manifest["val_episode_ids"]))
            if (not train or not val or len(set(train)) != len(train) or len(set(val)) != len(val)
                    or set(train) & set(val) or not set(train + val).issubset(self.episodes)):
                raise ValueError("pinned episode split is duplicated, overlapping or ineligible")
            self.episodes = {eid: self.episodes[eid] for eid in sorted(train + val)}
        self.ids = sorted(train if split == "train" else val)
        self.split_manifest = {
            "root": str(self.root), "seed": seed, "val_fraction": val_fraction,
            "task_ids": sorted(selected), "train_episode_ids": sorted(train),
            "val_episode_ids": sorted(val), "temporal_unit": "consecutive_dataset_rows",
            "source_may_remove_noops": True,
            "metadata_sha256": {
                name: hashlib.sha256((self.root / "meta" / name).read_bytes()).hexdigest()
                for name in ("info.json", "episodes.jsonl", "tasks.jsonl")
            },
            "parquet_identity": {
                str(eid): {"path": str(self.episode_path(eid).relative_to(self.root)),
                           "size": self.episode_path(eid).stat().st_size,
                           "mtime_ns": self.episode_path(eid).stat().st_mtime_ns}
                for eid in sorted(self.episodes)
            },
            "data_identity_kind": "metadata SHA256 and parquet size/mtime; not full parquet content hashing",
        }
        if os.environ.get("PI_FS_SKIP_HASH_VERIFICATION") != "1" and partition_manifest is not None and self.split_manifest != partition_manifest:
            raise ValueError("pinned partition source metadata or parquet identity changed")
        if robot_profile == "fr3":
            self.split_manifest.update(robot_profile="fr3", fps=self.info["fps"],
                state_dim=7, action_dim=7, source_may_remove_noops=None,
                temporal_unit=f"rows_at_{self.info['fps']}Hz", frame_root=str(self.frame_root), frame_source_stride=self.frame_source_stride,
                inherited_base_has_seen_dataset=True)
        self._configure_sampling(age_sampling, age_core_lo, age_core_hi, age_core_prob)
        self._cache: OrderedDict[int, dict] = OrderedDict()

    def _configure_sampling(self, strategy, core_lo, core_hi, core_probability):
        if strategy not in {"legacy", "age_first"}:
            raise ValueError("age_sampling must be legacy or age_first")
        if strategy == "legacy" and core_probability is not None:
            raise ValueError("age_core_prob only applies to age_first sampling")
        self.age_sampling = strategy
        self._age_values = np.arange(self.age_lo, self.age_hi + 1)
        self._age_probabilities = np.full(len(self._age_values), 1. / len(self._age_values))
        if core_probability is not None:
            if not np.isfinite(core_probability) or not 0 < core_probability < 1:
                raise ValueError("age_core_prob must be finite and strictly between zero and one")
            if not self.age_lo <= core_lo <= core_hi <= self.age_hi:
                raise ValueError("core age interval must be inside the supported age interval")
            core = (self._age_values >= core_lo) & (self._age_values <= core_hi)
            if core.all():
                raise ValueError("age mixture requires both core and non-core ages")
            self._age_probabilities = np.where(core, core_probability / core.sum(),
                                               (1. - core_probability) / (~core).sum())
        self._eligible_by_max_age = {
            int(age): [eid for eid in self.ids if self.episodes[eid]["length"] > age]
            for age in self._age_values
        }
        if strategy == "age_first" and self.ids and not self._eligible_by_max_age[self.age_hi]:
            raise ValueError("selected split cannot support the full configured age range")
        self.sampling_manifest = {
            "strategy": strategy, "age_lo": self.age_lo, "age_hi": self.age_hi,
            "core_lo": core_lo if core_probability is not None else None,
            "core_hi": core_hi if core_probability is not None else None,
            "core_probability": core_probability,
            "requested_age_probabilities": ({str(int(age)): float(prob) for age, prob in
                                               zip(self._age_values, self._age_probabilities, strict=True)}
                                              if strategy == "age_first" else None),
            "order": ("target ages first; episode uniform among those supporting max(target ages); "
                      "anchor uniform over [0, length-1-max_age]; all targets share this real anchor"
                      if strategy == "age_first" else "episode, anchor, then conditionally truncated uniform age"),
            "temporal_unit": self.split_manifest["temporal_unit"],
        }

    def __len__(self):
        return len(self.ids)

    def episode_path(self, eid: int) -> Path:
        template = self.info["data_path"]
        path = self.root / template.format(
            episode_chunk=eid // int(self.info.get("chunks_size", 1000)), episode_index=eid,
        )
        if not path.resolve().is_relative_to(self.root):
            raise ValueError("parquet path escapes dataset root")
        return path

    def load_episode(self, eid: int) -> dict:
        if eid not in self.episodes:
            raise KeyError(f"episode {eid} is not selected")
        if eid in self._cache:
            self._cache.move_to_end(eid)
            return self._cache[eid]
        import pyarrow.parquet as pq

        columns = [self.state_key, self.action_key, "frame_index", "episode_index", "task_index"]
        if self.robot_profile != "fr3":
            columns += ["image", "wrist_image"]
        if self.frame_source_stride>1:columns.append("source_frame_index")
        raw = pq.read_table(self.episode_path(eid), columns=columns).to_pydict()
        if self.robot_profile == "fr3":
            raw["state"] = raw.pop(self.state_key)
            raw["actions"] = raw.pop(self.action_key)
            if self.frame_root is None:
                raise ValueError("FR3 requires prepared RGB frame_root")
            frames = np.load(self.frame_root / f"episode_{eid:06d}.npy", mmap_mode="r")
            expected=(len(raw["state"])-1)//self.frame_source_stride+1
            if frames.shape != (expected, 2, 224, 224, 3) or frames.dtype != np.uint8:
                raise ValueError("FR3 source camera frames differ from episode mapping")
            if self.frame_source_stride>1:
                mapping=np.asarray(raw["source_frame_index"],dtype=np.int64)
                if not np.array_equal(mapping,np.arange(len(raw["state"]))//self.frame_source_stride):raise ValueError("Noncausal source frame mapping")
                raw["source_image_index"]=mapping
            raw["image"], raw["wrist_image"] = frames[:, 0], frames[:, 1]
        n = len(raw["state"])
        if n != self.episodes[eid]["length"]:
            raise ValueError(f"episode {eid} length differs from metadata")
        if not np.array_equal(np.asarray(raw["frame_index"]).reshape(-1), np.arange(n)):
            raise ValueError(f"episode {eid} has nonconsecutive frame_index; cannot infer causal history")
        if not np.all(np.asarray(raw["episode_index"]) == eid):
            raise ValueError(f"episode {eid} contains another episode")
        if not np.all(np.asarray(raw["task_index"]) == self.episodes[eid]["task_id"]):
            raise ValueError(f"episode {eid} changes task within a trajectory")
        raw["state"] = np.asarray(raw["state"], dtype=np.float32)
        raw["actions"] = np.asarray(raw["actions"], dtype=np.float32)
        if raw["state"].shape != (n, self.state_dim) or raw["actions"].shape != (n, 7):
            raise ValueError(f"invalid state/action shape in episode {eid}")
        if not np.isfinite(raw["state"]).all() or not np.isfinite(raw["actions"]).all():
            raise ValueError(f"nonfinite state/action in episode {eid}")
        self._cache[eid] = raw
        while len(self._cache) > self.cache_episodes:
            self._cache.popitem(last=False)
        return raw

    def _decode_image(self, value) -> np.ndarray:
        if isinstance(value, dict):
            if value.get("bytes") is not None:
                source = io.BytesIO(value["bytes"])
            elif value.get("path"):
                path = (self.root / value["path"]).resolve()
                if not path.is_relative_to(self.root):
                    raise ValueError("image path escapes dataset root")
                source = path
            else:
                raise ValueError("image record has neither bytes nor path")
            with Image.open(source) as image:
                result = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        else:
            result = np.asarray(value)
        if result.ndim != 3 or result.shape[-1] != 3 or result.dtype != np.uint8:
            raise ValueError("LIBERO images must be uint8 HWC RGB")
        # Published RLDS/LeRobot images already have the model's orientation.
        # Only live simulator observations need the 180-degree flip.
        return result

    def observation(self, eid: int, frame: int) -> dict:
        episode = self.load_episode(eid)
        if not 0 <= frame < len(episode["state"]):
            raise IndexError(frame)
        image_index=int(episode["source_image_index"][frame]) if "source_image_index" in episode else frame
        return {
            "observation/state": episode["state"][frame].copy(),
            "observation/image": self._decode_image(episode["image"][image_index]),
            "observation/wrist_image": self._decode_image(episode["wrist_image"][image_index]),
            "prompt": self.tasks[self.episodes[eid]["task_id"]],
        }

    def sample_pairs(self, rng: np.random.Generator, batch_snapshots: int = 4, targets_per_snapshot: int = 4) -> PairBatch:
        if batch_snapshots < 1 or targets_per_snapshot < 1:
            raise ValueError("batch sizes must be positive")
        p = batch_snapshots * targets_per_snapshot
        snapshots, current = [], []
        snap_state = np.empty((p, getattr(self, "state_dim", 8)), np.float32)
        now_state = np.empty_like(snap_state)
        states = np.zeros((p, self.history, getattr(self, "state_dim", 8)), np.float32)
        actions = np.zeros((p, self.history, 7), np.float32)
        indices = {key: np.empty(p, np.int64) for key in ("ages", "episode_ids", "t0", "t1", "snapshot_for_pair")}
        pair = 0
        for snapshot in range(batch_snapshots):
            if self.age_sampling == "age_first":
                ages = rng.choice(self._age_values, size=targets_per_snapshot, p=self._age_probabilities)
                maximum = int(ages.max())
                candidates = self._eligible_by_max_age[maximum]
                if getattr(self, "task_balanced", False):
                    groups = defaultdict(list)
                    for candidate in candidates:
                        record = self.episodes[candidate]
                        groups[record.get("benchmark_task_id", record["task_id"])].append(candidate)
                    task = getattr(self, "validation_task_id", None)
                    if task is None: task = int(rng.choice(sorted(groups)))
                    candidates = groups[task]
                eid = int(rng.choice(candidates))
                episode = self.load_episode(eid)
                n = len(episode["state"])
                t0 = int(rng.integers(0, n - maximum))
            else:
                eid = int(rng.choice(self.ids))
                episode = self.load_episode(eid)
                n = len(episode["state"])
                t0 = int(rng.integers(0, n - self.age_lo))
                ages = [int(rng.integers(self.age_lo, min(self.age_hi, n - 1 - t0) + 1))
                        for _ in range(targets_per_snapshot)]
            snapshots.append(self.observation(eid, t0))
            for sampled_age in ages:
                age = int(sampled_age)
                t1 = t0 + age
                current.append(self.observation(eid, t1))
                snap_state[pair], now_state[pair] = episode["state"][[t0, t1]]
                states[pair, :age] = episode["state"][t0 + 1:t1 + 1]
                actions[pair, :age] = episode["actions"][t0:t1]
                for key, value in (("ages", age), ("episode_ids", eid), ("t0", t0), ("t1", t1), ("snapshot_for_pair", snapshot)):
                    indices[key][pair] = value
                pair += 1
        shift = getattr(self, "cache_replan_steps", 0)
        previous = ([self.observation(int(eid), max(0, int(t)-shift))
                     for eid,t in zip(indices["episode_ids"],indices["t1"])] if shift else None)
        return PairBatch(snapshots, current, indices["snapshot_for_pair"], snap_state, now_state, states, actions,
                         indices["ages"], indices["episode_ids"], indices["t0"], indices["t1"], previous)

    def state_statistics(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute statistics only over this split, without decoding images."""
        import pyarrow.parquet as pq

        count = 0
        total = np.zeros(self.state_dim, np.float64)
        squared = np.zeros(self.state_dim, np.float64)
        for eid in self.ids:
            states = np.asarray(pq.read_table(self.episode_path(eid), columns=[self.state_key])[self.state_key].to_pylist(), np.float64)
            if states.ndim != 2 or states.shape[1] != self.state_dim or not np.isfinite(states).all():
                raise ValueError(f"invalid states in episode {eid}")
            count += len(states)
            total += states.sum(0)
            squared += np.square(states).sum(0)
        mean = total / count
        sd = np.sqrt(np.maximum(squared / count - mean**2, 0)) + 1e-3
        return mean.astype(np.float32), sd.astype(np.float32)


class RecordedEpisodeDataset(RawEpisodeDataset):
    """Causal online trajectories recorded by dual.eval_libero.

    Roles are explicit, never inferred from filename or a successful outcome.
    Evaluation recordings cannot become updater training data by accident.
    """

    def __init__(self, root, *, split="train", age_lo=10, age_hi=45, history=48, cache_episodes=2,
                 age_sampling="legacy", age_core_lo=20, age_core_hi=39, age_core_prob=None, **unused):
        if split not in {"train", "val"} or not 1 <= age_lo <= age_hi <= history:
            raise ValueError("invalid recorded trajectory split or age range")
        self.root = Path(root).resolve()
        self.split, self.age_lo, self.age_hi, self.history = split, age_lo, age_hi, history
        self.cache_episodes = cache_episodes
        self._cache = OrderedDict()
        self.episodes, self.tasks, self.paths = {}, {}, {}
        self._content_hashes = {}
        self._verified_content = {}
        self._file_identities = {}
        train, val, manifest = [], [], []
        scenes = {"train": set(), "validation": set()}
        physical_scenes = {"train": set(), "validation": set()}
        paths = sorted(path for path in self.root.glob("*.npz") if not path.name.endswith('.tmp.npz'))
        if not paths:
            raise ValueError(f"no recorded episodes in {self.root}")
        for eid, path in enumerate(paths):
            with np.load(path, allow_pickle=False) as archive:
                if "meta_json" not in archive:
                    raise ValueError(f"recording {path} lacks explicit data_role; do not infer train/test from filename")
                metadata = json.loads(str(archive["meta_json"].item()))
                role = metadata.get("data_role")
                if role == "evaluation":
                    raise ValueError(f"evaluation-only recording cannot enter updater dataset: {path}")
                if role not in {"train", "validation"}:
                    raise ValueError(f"unknown recording data_role in {path}")
                prompt = str(archive["prompt"].item())
                n = len(archive["state"])
                if n <= age_lo:
                    continue
                if not all(key in metadata for key in ("task_id", "init_index", "seed", "benchmark")):
                    raise ValueError(f"recording lacks scene identity: {path}")
                scene = f"{metadata['benchmark']}:{metadata['task_id']}:{metadata['init_index']}"
                content_sha256 = None
                if metadata.get('schema_version') == 2:
                    required = {'initial_state_kind', 'initial_state_seed', 'initial_state_sha256', 'scene_key',
                                'dataset_scope', 'temporal_unit', 'removes_noops', 'action_semantics', 'success'}
                    if not required <= metadata.keys() or metadata['temporal_unit'] != 'environment_step' or metadata['removes_noops']:
                        raise ValueError(f'incomplete causal recording metadata: {path}')
                    kind = metadata['initial_state_kind']
                    if kind == 'fresh_reset':
                        if (metadata['benchmark'] != 'native' or metadata['init_index'] < 1000
                                or not 0 <= metadata['task_id'] < 10
                                or metadata['initial_state_seed'] != 1000 * metadata['init_index'] + metadata['task_id']):
                            raise ValueError(f'invalid fresh-scene identity: {path}')
                        scene = f"native:fresh_reset:{metadata['task_id']}:{metadata['init_index']}"
                    elif kind != 'fixed_initial_state' or metadata['initial_state_seed'] is not None:
                        raise ValueError(f'invalid initial-state kind: {path}')
                    if metadata['scene_key'] != scene:
                        raise ValueError(f'inconsistent scene key: {path}')
                    state_hash = hashlib.sha256(np.asarray(archive['initial_state']).tobytes()).hexdigest()
                    if os.environ.get('PI_FS_SKIP_HASH_VERIFICATION') != '1' and state_hash != metadata['initial_state_sha256']:
                        raise ValueError(f'initial state does not match its hash: {path}')
                    physical_scenes[role].add(f"{metadata['benchmark']}:{metadata['task_id']}:{state_hash}")
                    receipt_path = path.with_suffix('.receipt.json')
                    if not receipt_path.is_file():
                        raise ValueError(f'committed recording lacks content receipt: {path}')
                    receipt = json.loads(receipt_path.read_text())
                    if (receipt.get('schema_version') != 2 or receipt.get('bytes') != path.stat().st_size
                            or receipt.get('frames') != n or Path(receipt.get('path', '')).resolve() != path.resolve()
                            or (os.environ.get('PI_FS_SKIP_HASH_VERIFICATION') != '1' and
                                (not isinstance(receipt.get('sha256'), str) or len(receipt['sha256']) != 64))):
                        raise ValueError(f'invalid recording content receipt: {path}')
                    content_sha256 = receipt['sha256']
                    self._content_hashes[eid] = content_sha256
                scenes[role].add(scene)
                self.paths[eid] = path
                self._file_identities[eid] = (path.stat().st_size, path.stat().st_mtime_ns)
                # A unique text key per recorded episode permits legitimate
                # language variants without assuming LeRobot task numbering.
                self.tasks[eid] = prompt
                self.episodes[eid] = {"length": n, "task_id": eid, "tasks": [prompt], "scene_key": scene, "benchmark_task_id": metadata["task_id"]}
                (train if role == "train" else val).append(eid)
                manifest.append({"episode_id": eid, "file": path.name, "size": path.stat().st_size,
                                 "mtime_ns": path.stat().st_mtime_ns, "length": n, "metadata": metadata,
                                 "prompt": prompt, **({'sha256': content_sha256} if content_sha256 else {})})
        if scenes["train"] & scenes["validation"] or physical_scenes['train'] & physical_scenes['validation']:
            raise ValueError("same recorded task/init scene appears in train and validation, even across noise seeds")
        self.ids = train if split == "train" else val
        self.split_manifest = {"root": str(self.root), "source": "dual_recordings", "train_episode_ids": train,
                               "val_episode_ids": val, "train_scene_keys": sorted(scenes["train"]),
                               "val_scene_keys": sorted(scenes["validation"]), "episodes": manifest,
                               "temporal_unit": "environment_step", "source_may_remove_noops": False}
        if any(physical_scenes.values()):
            self.split_manifest.update(train_initial_state_keys=sorted(physical_scenes['train']),
                                       val_initial_state_keys=sorted(physical_scenes['validation']),
                                       recording_identity='schema2 content SHA256 from atomic receipt, verified when loading each episode')
        self._configure_sampling(age_sampling, age_core_lo, age_core_hi, age_core_prob)

    def load_episode(self, eid):
        if eid not in self.episodes:
            raise KeyError(f'episode {eid} is not selected')
        if eid in self._cache:
            self._cache.move_to_end(eid)
            return self._cache[eid]
        expected = self._content_hashes.get(eid)
        identity = (self.paths[eid].stat().st_size, self.paths[eid].stat().st_mtime_ns)
        if identity != self._file_identities[eid]:
            raise ValueError(f'recording size/mtime changed after dataset initialization: {self.paths[eid]}')
        if os.environ.get('PI_FS_SKIP_HASH_VERIFICATION') != '1' and expected and self._verified_content.get(eid) != identity:
            digest = hashlib.sha256()
            with self.paths[eid].open('rb') as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    digest.update(block)
            if digest.hexdigest() != expected:
                raise ValueError(f'recording content changed: {self.paths[eid]}')
            self._verified_content[eid] = identity
        with np.load(self.paths[eid], allow_pickle=False) as archive:
            raw = {"state": np.asarray(archive["state"], np.float32),
                   "actions": np.asarray(archive["action"], np.float32),
                   "image": np.asarray(archive["img"]), "wrist_image": np.asarray(archive["wrist"])}
        n = self.episodes[eid]["length"]
        if raw["state"].shape != (n, 8) or raw["actions"].shape != (n, 7):
            raise ValueError("recording must contain one pre-action state and one executed action per step")
        for name in ("image", "wrist_image"):
            if raw[name].shape[0] != n or raw[name].ndim != 4 or raw[name].shape[-1] != 3 or raw[name].dtype != np.uint8:
                raise ValueError("recorded image/state time axes differ or image format is invalid")
        if not np.isfinite(raw["state"]).all() or not np.isfinite(raw["actions"]).all():
            raise ValueError("nonfinite recorded state/action")
        self._cache[eid] = raw
        while len(self._cache) > self.cache_episodes:
            self._cache.popitem(last=False)
        return raw

    def state_statistics(self):
        total, squared, count = np.zeros(8, np.float64), np.zeros(8, np.float64), 0
        for eid in self.ids:
            with np.load(self.paths[eid], allow_pickle=False) as archive:
                states = np.asarray(archive["state"], np.float64)
            if states.ndim != 2 or states.shape[1] != 8 or not np.isfinite(states).all():
                raise ValueError("invalid recorded states")
            count += len(states)
            total += states.sum(0)
            squared += np.square(states).sum(0)
        if not count:
            raise ValueError("no states in selected recorded split")
        mean = total / count
        return mean.astype(np.float32), (np.sqrt(np.maximum(squared / count - mean**2, 0)) + 1e-3).astype(np.float32)


class MixedEpisodeDataset:
    """Append train-role on-policy rollouts without relabeling evaluation data."""

    def __init__(self, sources, weights=None):
        manifests = [source.split_manifest for source in sources]
        train_scenes = set().union(*(set(m.get("train_scene_keys", [])) for m in manifests))
        val_scenes = set().union(*(set(m.get("val_scene_keys", [])) for m in manifests))
        train_states = set().union(*(set(m.get('train_initial_state_keys', [])) for m in manifests))
        val_states = set().union(*(set(m.get('val_initial_state_keys', [])) for m in manifests))
        if train_scenes & val_scenes or train_states & val_states:
            raise ValueError("recorded scenes overlap train/validation across source directories")
        if weights is not None and (len(weights) != len(sources) or any(w < 0 for w in weights)):
            raise ValueError("one nonnegative mixture weight per source required")
        chosen = [(s, weights[i] if weights is not None else len(s)) for i,s in enumerate(sources) if len(s)]
        self.sources = [s for s,w in chosen]
        chosen_weights = [w for s,w in chosen]
        if not self.sources:
            raise ValueError("mixed split contains no eligible episodes")
        self.split_manifest = {"sources": manifests, "sampling": "uniform_episode_then_snapshot_then_age",
                               "temporal_unit": "source_specific; retained demonstration rows or recorded environment steps"}
        self.sampling_manifest = {"strategy": "mixed_sources", "source_weighting": "number_of_episodes",
                                  "sources": [source.sampling_manifest for source in self.sources]}
        if any(source.age_sampling != "legacy" for source in self.sources):
            self.split_manifest["sampling"] = "episode-weighted source, then explicit source age sampler"
        self.weights = np.asarray(chosen_weights, dtype=float)
        if not self.weights.sum() > 0: raise ValueError("empty mixture probability mass")
        self.sampling_manifest["source_weighting"] = "explicit_probabilities" if weights is not None else "number_of_episodes"
        self.weights /= self.weights.sum()
        self.sampling_manifest["source_probabilities"] = self.weights.tolist()

    def __len__(self):
        return sum(len(source) for source in self.sources)

    def sample_pairs(self, rng, batch_snapshots=4, targets_per_snapshot=4):
        batches = [self.sources[int(rng.choice(len(self.sources), p=self.weights))].sample_pairs(rng, 1, targets_per_snapshot)
                   for _ in range(batch_snapshots)]
        snapshots = [obs for batch in batches for obs in batch.snapshots]
        current = [obs for batch in batches for obs in batch.current]
        fields = {name: np.concatenate([getattr(batch, name) for batch in batches]) for name in
                  ("snapshot_state", "current_state", "states_after", "executed_actions", "ages", "episode_ids", "t0", "t1")}
        index = np.repeat(np.arange(batch_snapshots), targets_per_snapshot)
        previous = None if batches[0].previous is None else [obs for batch in batches for obs in batch.previous]
        return PairBatch(snapshots=snapshots, current=current, snapshot_for_pair=index, previous=previous, **fields)

    def state_statistics(self):
        total, second, count = np.zeros(8, np.float64), np.zeros(8, np.float64), 0
        for source in self.sources:
            mean, sd = source.state_statistics()
            n = sum(source.episodes[eid]["length"] for eid in source.ids)
            total += mean * n
            second += ((sd - 1e-3)**2 + mean**2) * n
            count += n
        mean = total / count
        return mean.astype(np.float32), (np.sqrt(np.maximum(second / count - mean**2, 0)) + 1e-3).astype(np.float32)


def split_identities(manifest: dict) -> dict[str, set[str]]:
    """Stable identities across repeated runs and relocated recording rounds.

    Demonstrations have no simulator initial-state id in the public metadata,
    so their identity is canonical dataset root + episode id. Recorded scenes
    use benchmark/task/init, independent of trajectory noise seed or directory.
    """
    result = {"train": set(), "validation": set()}
    if "sources" in manifest:
        for source in manifest["sources"]:
            identities = split_identities(source)
            for role in result:
                result[role].update(identities[role])
    elif manifest.get("source") == "dual_recordings":
        for role, key in (("train", "train_scene_keys"), ("validation", "val_scene_keys")):
            result[role].update("recorded:" + scene for scene in manifest[key])
        for role, key in (("train", "train_initial_state_keys"), ("validation", "val_initial_state_keys")):
            result[role].update("recorded-state:" + scene for scene in manifest.get(key, []))
    else:
        root = str(Path(manifest["root"]).resolve())
        for role, key in (("train", "train_episode_ids"), ("validation", "val_episode_ids")):
            result[role].update(json.dumps(["lerobot", root, int(eid)], separators=(",", ":")) for eid in manifest[key])
    if result["train"] & result["validation"]:
        raise ValueError("data identity appears in both training and validation")
    return result


def merge_split_lineage(manifest: dict, previous_provenance: dict | None = None) -> dict:
    """Forbid train/validation reassignment across the entire checkpoint lineage."""
    current = split_identities(manifest)
    previous = {"train": set(), "validation": set()}
    if previous_provenance:
        if "data_lineage" in previous_provenance:
            previous = {role: set(previous_provenance["data_lineage"][role]) for role in previous}
        else:
            previous = split_identities(previous_provenance["data"])
    if previous["train"] & previous["validation"]:
        raise ValueError("checkpoint contains conflicting train/validation lineage")
    overlap = (previous["train"] & current["validation"]) | (previous["validation"] & current["train"])
    if overlap:
        raise ValueError(f"data split reassigned across checkpoint lineage ({len(overlap)} identities); preserve prior held-out data")
    return {role: sorted(previous[role] | current[role]) for role in previous}
