"""Per-environment-step dual system with independent action endpoint reuse.

Virtual mode reproduces the report's publication clock, not measured latency.
Threaded mode captures available observations in a background CUDA stream and
publishes only completed work. Both use actual executed-action feedback.
"""
from __future__ import annotations
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
import threading
import time

import numpy as np
import torch

from openpi.dual.bridge import FrozenPi05, PrefixKV, file_sha256
from openpi.dual.config import SystemConfig, UpdaterConfig
from openpi.dual.history import condition_features
from openpi.dual.model import SelfUpdater, UpdaterNorm, masked_kv_loss
from openpi.dual.rose_tfws import integration_schedule, resolve_initial_latent


@dataclass
class Snapshot:
    step: int
    state: np.ndarray
    prefix: PrefixKV


def load_updater(path, config, device):
    from openpi.dual.release_weights import load_adapter
    return load_adapter(path, config, device)


class DualPolicy:
    """One episode/session; call act at every environment step, never per chunk.

    For n>0, executed_action must be the action actually passed to env.step at
    n-1 (after clipping or controller processing, if applicable). New prompts
    require reset; hidden inference history is never carried between episodes.
    """
    def __init__(self, bridge, config: SystemConfig, updater=None, *, shadow_every_replans=0):
        self.bridge, self.config, self.updater = bridge, config, updater
        self.device = torch.device(bridge.device)
        if type(shadow_every_replans) is not int or shadow_every_replans < 0:
            raise ValueError('shadow_every_replans must be a nonnegative integer')
        if shadow_every_replans and not (config.dual.enabled and config.dual.updater_enabled
                                         and config.dual.mode == 'virtual' and not config.cache.enabled
                                         and config.num_steps == 1):
            raise ValueError('Shadow requires virtual dual updater, cache off and NFE1; it cannot perturb threaded timing')
        self.shadow_every_replans = shadow_every_replans
        self._fused = bool(getattr(bridge, 'fused_fast', False))
        if self._fused and shadow_every_replans:
            raise ValueError('Shadow monkeypatch diagnostics are not supported inside compiled graphs')
        self._compiled_warm = False
        self.warmup_ms = 0.
        # The shared history encoder validates CPU observations before a single
        # transfer. No repeated device scalar reads are needed in the fast path.
        self._history_norm = updater.norm.to('cpu') if self._fused and updater is not None else None
        if config.dual.enabled and config.dual.updater_enabled and updater is None:
            raise ValueError('Dual updater requires a trained checkpoint; random weights are not deployed')
        self._pool = ThreadPoolExecutor(max_workers=1) if config.dual.enabled and config.dual.mode == 'threaded' else None
        self._stream = torch.cuda.Stream(device=self.device) if self._pool and self.device.type == 'cuda' else None
        self._future = None
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        if self._lock.locked():
            raise RuntimeError('Cannot reset during an active inference')
        if self._future is not None:
            try:
                self._future.result()  # drain old episode before recycling shared model
            except Exception:
                # An error is surfaced by act() when publishing that future;
                # reset must still be able to recover a poisoned old session.
                pass
        self._future = None
        self.step = 0
        self.prompt = None
        self.snapshot = None
        self.endpoint = None
        self.states, self.actions = {}, {}
        self.chunk = None
        self._jobs = []
        self._next_pass = 1
        self._last_meta = {}
        self._failed = False

    def close(self):
        try:
            self.reset()
        finally:
            if self._pool:
                self._pool.shutdown(wait=True)

    def _sync(self):
        if self.device.type == 'cuda':
            # Only wait for this stream; global synchronize would serialize the
            # background brain and invalidate threaded execution semantics.
            torch.cuda.current_stream(self.device).synchronize()

    def _capture(self, obs, step):
        prefix = self.bridge.prefix([obs])
        return Snapshot(step, np.asarray(obs['observation/state'], dtype=np.float32).copy(), prefix)

    def _background_capture(self, obs, step):
        if self._stream is None:
            return self._capture(obs, step)
        with torch.cuda.device(self.device), torch.cuda.stream(self._stream):
            result = self._capture(obs, step)
            self._stream.synchronize()
        return result

    def _advance_brain(self, obs):
        cfg = self.config.dual
        if not cfg.enabled:
            return
        if self.snapshot is None:
            self.snapshot = self._capture(obs, self.step)
        if cfg.mode == 'virtual':
            ratio = cfg.brain_ms / cfg.action_ms
            # Capture at floor(start time / action slot), publish at ceil(end).
            while math.floor((self._next_pass - 1) * ratio + 1e-10) <= self.step:
                capture_step = math.floor((self._next_pass - 1) * ratio + 1e-10)
                if capture_step != self.step:
                    raise RuntimeError('Missing per-step observation in virtual brain clock')
                snap = self.snapshot if capture_step == 0 else self._capture(obs, capture_step)
                ready = math.ceil(self._next_pass * ratio - 1e-10)
                self._jobs.append((ready, snap))
                self._next_pass += 1
            while self._jobs and self._jobs[0][0] <= self.step:
                _, snap = self._jobs.pop(0)
                if snap.step >= self.snapshot.step:
                    self.snapshot = snap
        else:
            if self._future is not None and self._future.done():
                snap = self._future.result()
                if snap.step >= self.snapshot.step:
                    self.snapshot = snap
                self._future = None
            if self._future is None:
                frozen_obs = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in obs.items()}
                self._future = self._pool.submit(self._background_capture, frozen_obs, self.step)

    def _condition(self, snapshot):
        age = self.step - snapshot.step
        length = self.config.updater.history
        states = np.zeros((1, length, 8), dtype=np.float32)
        actions = np.zeros((1, length, 7), dtype=np.float32)
        for offset in range(age):
            states[0, offset] = self.states[snapshot.step + offset + 1]
            actions[0, offset] = self.actions[snapshot.step + offset]
        device = 'cpu' if self._fused else self.device
        tensor = lambda value: torch.as_tensor(value, device=device)
        norm = self._history_norm if self._fused else self.updater.norm
        condition = condition_features(tensor(snapshot.state[None]), tensor(self.states[self.step][None]),
                                       tensor(states), tensor(actions), tensor([age]), norm, length)
        return condition.to(self.device)

    def _shadow_diagnostic(self, obs, snap, pred_k, pred_v, endpoint, innovation, updated):
        """Read-only same-observation oracle comparisons; never publish teacher KV.

        Virtual-clock-only: extra work must not affect when a real brain snapshot
        becomes available. All branches share the same new innovation, and only
        the already selected policy action is executed.
        """
        self._sync()
        begin = time.perf_counter()
        age = self.step - snap.step
        model = self.bridge.model
        original = model.denoise_step
        extra_calls = 0
        def counted(*values, **kwargs):
            nonlocal extra_calls
            extra_calls += 1
            return original(*values, **kwargs)
        model.denoise_step = counted
        try:
            teacher = self.bridge.prefix([obs])
            if not torch.equal(teacher.mask, snap.prefix.mask):
                raise ValueError('Shadow teacher changed prefix mask inside an episode')
            teacher_actions = self.bridge.hand(teacher.k, teacher.v, teacher.mask, teacher.state,
                                               innovation, num_steps=1)
            if updated:
                predicted_actions = endpoint
                stale_actions = self.bridge.hand(snap.prefix.k, snap.prefix.v, snap.prefix.mask,
                                                 snap.prefix.state, innovation, num_steps=1)
            else:
                # The gated real branch already used stale KV. Predict only for
                # diagnosis; its output cannot replace the chosen stale chunk.
                stale_actions = endpoint
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                                    enabled=self.device.type == 'cuda'):
                    pred_k, pred_v = self.updater(snap.prefix.k, snap.prefix.v, snap.prefix.mask,
                                                  self._condition(snap))
                predicted_actions = self.bridge.hand(pred_k, pred_v, snap.prefix.mask, snap.prefix.state,
                                                      innovation, num_steps=1)
        finally:
            model.denoise_step = original
        if extra_calls != 2:
            raise RuntimeError(f'Shadow expected two additional actual denoiser calls, got {extra_calls}')
        pred_error = (predicted_actions[..., :7].float() - teacher_actions[..., :7].float()).square()
        stale_error = (stale_actions[..., :7].float() - teacher_actions[..., :7].float()).square()
        metrics = {
            'kv_pred': masked_kv_loss(pred_k, pred_v, teacher.k, teacher.v, teacher.mask, self.updater.norm),
            'kv_stale': masked_kv_loss(snap.prefix.k, snap.prefix.v, teacher.k, teacher.v,
                                     teacher.mask, self.updater.norm),
        }
        c = self.config.replan_steps
        for label, error in [('pred', pred_error), ('stale', stale_error)]:
            metrics.update({label: error.mean(), label + '_arm': error[..., :6].mean(),
                            label + '_grip': error[..., 6].mean(), label + '_first_c': error[:, :c].mean(),
                            label + '_first_c_arm': error[:, :c, :6].mean(),
                            label + '_first_c_grip': error[:, :c, 6].mean()})
        if not all(bool(torch.isfinite(value).all()) for value in metrics.values()):
            raise ValueError('Nonfinite shadow prediction metric')
        result = {name: float(value) for name, value in metrics.items()}
        teacher_command = self.bridge.output_actions(teacher_actions, teacher.state)
        for label, actions in [('pred', predicted_actions), ('stale', stale_actions)]:
            command = self.bridge.output_actions(actions, teacher.state)
            disagreement = np.sign(command[:, 6]) != np.sign(teacher_command[:, 6])
            result[label + '_gripper_sign_disagreement'] = float(disagreement.mean())
            result[label + '_first_c_gripper_sign_disagreement'] = float(disagreement[:c].mean())
        raw_states = torch.as_tensor(np.stack([snap.state, self.states[self.step]]), device=self.device)
        standardized = ((raw_states - self.updater.norm.state_mu) / self.updater.norm.state_sd).abs()
        result['state_norm_absmax'] = float(standardized.max())
        result['state_norm_over3sigma_fraction'] = float((standardized > 3).float().mean())
        self._sync()
        result.update(age=age, replan_index=self.step // c,
                      executed_branch='updater' if updated else 'stale_low_age_gate',
                      extra_actual_denoiser_calls=extra_calls, extra_updater_calls=int(not updated),
                      teacher_prefix_computations=1, teacher_prefix_published=False,
                      shadow_ms=(time.perf_counter() - begin) * 1000,
                      scope='same-observation same-innovation counterfactual; teacher never executed or published',
                      timing_scope='virtual diagnostic extra work included in act_wall_ms; not a performance measurement')
        return result

    @torch.no_grad()
    def act(self, obs, *, executed_action=None, noise=None):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('Concurrent act calls are not supported; use one policy per session')
        try:
            if self._failed:
                raise RuntimeError('Previous inference failed; reset the session before continuing')
            return self._act(obs, executed_action=executed_action, noise=noise)
        except Exception:
            # Invalidate action endpoints/history after any failed inference.
            self.endpoint = None
            self.chunk = None
            self._failed = True
            raise
        finally:
            self._lock.release()

    def _act(self, obs, *, executed_action, noise):
        prompt = obs.get('prompt')
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError('A nonempty task prompt is required')
        if self.prompt is not None and prompt != self.prompt:
            raise ValueError('Prompt changed; reset the dual session before a new task')
        self.prompt = prompt
        state = np.asarray(obs['observation/state'], dtype=np.float32)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError('Raw LIBERO state must be finite [8]')
        if self.step == 0:
            if executed_action is not None:
                raise ValueError('First frame cannot have a previous executed action')
        else:
            action = np.asarray(executed_action, dtype=np.float32)
            if action.shape != (7,) or not np.isfinite(action).all():
                raise ValueError('Supply previous actually executed action [7] at every frame')
            self.actions[self.step - 1] = action.copy()
        self.states[self.step] = state.copy()
        if self._fused and not self._compiled_warm:
            warmup_start = time.perf_counter()
            prefix = self.bridge.prefix([obs])
            self.bridge.warmup(prefix, self.updater, self.config)
            self.warmup_ms = (time.perf_counter() - warmup_start) * 1000
            self._compiled_warm = True
        self._sync()
        begin = time.perf_counter()
        self._advance_brain(obs)
        replanned = self.step % self.config.replan_steps == 0
        timing = {}
        shadow = None
        if replanned:
            fallback = False
            if not self.config.dual.enabled:
                snap = self._capture(obs, self.step)
            else:
                snap = self.snapshot
                if self.step - snap.step > self.config.dual.max_age:
                    # Explicit synchronous refresh, never silently drop history.
                    snap = self.snapshot = self._capture(obs, self.step)
                    fallback = True
            age = self.step - snap.step
            mask = snap.prefix.mask
            k, v = (None, None) if self._fused else (snap.prefix.k, snap.prefix.v)
            self._sync()
            updater_start = time.perf_counter()
            skipped_low_age = (self.config.dual.enabled and self.config.dual.updater_enabled
                               and 0 < age < self.config.dual.updater_min_age)
            updated = (self.config.dual.enabled and self.config.dual.updater_enabled
                       and age >= self.config.dual.updater_min_age)
            condition = self._condition(snap) if updated else None
            if updated and not self._fused:
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == 'cuda'):
                    k, v = self.updater(k, v, mask, condition)
            if not self._fused:
                self._sync()
                timing['updater_ms'] = (time.perf_counter() - updater_start) * 1000
            else:
                timing['history_prepare_ms'] = (time.perf_counter() - updater_start) * 1000
            if noise is None:
                innovation = torch.randn((1, 10, 32), device=self.device)
            else:
                innovation = torch.as_tensor(noise, dtype=torch.float32, device=self.device)
                if innovation.ndim == 2:
                    innovation = innovation[None]
                if innovation.shape != (1, 10, 32) or not bool(torch.isfinite(innovation).all()):
                    raise ValueError('Innovation must be finite [10,32] or [1,10,32]')
            latent, start, warm = resolve_initial_latent(
                innovation, self.endpoint, enabled=self.config.cache.enabled,
                restart_time=self.config.cache.restart_time, replan_steps=self.config.replan_steps)
            nfe, _ = integration_schedule(self.config.num_steps, start)
            if warm and self.config.cache.warm_nfe is not None:
                nfe = self.config.cache.warm_nfe
            hand_start = time.perf_counter()
            if updated and self._fused:
                endpoint = self.bridge.fast(self.updater, snap.prefix, condition, latent,
                                            num_steps=nfe, start_time=start)
            elif self._fused:
                endpoint = self.bridge.expert(snap.prefix, latent, num_steps=nfe, start_time=start)
            else:
                endpoint = self.bridge.hand(k, v, mask, snap.prefix.state, latent, num_steps=nfe, start_time=start)
            self._sync()
            timing['fast_graph_ms' if self._fused else 'hand_ms'] = (time.perf_counter() - hand_start) * 1000
            chunk = self.bridge.output_actions(endpoint, snap.prefix.state)
            self.chunk = chunk
            self.endpoint = endpoint.detach().clone() if self.config.cache.enabled else None
            if (self.shadow_every_replans and age > 0
                    and (age < 20 or (self.step // self.config.replan_steps) % self.shadow_every_replans == 0)):
                shadow = self._shadow_diagnostic(obs, snap, k, v, endpoint, innovation, bool(updated))
            self._last_meta = {'snapshot_step': snap.step, 'age': age, 'updated': bool(updated),
                               'updater_skipped_low_age': bool(skipped_low_age),
                               'synchronous_age_fallback': fallback, 'warm_start': warm, 'start_time': start,
                               'actual_nfe': nfe, 'mode': self.config.dual.mode if self.config.dual.enabled else 'synchronous',
                               'timing_scope': 'current_stream_synchronized; threaded brain may overlap',
                               'virtual_clock_models_gpu_contention': False,
                               'shadow_diagnostic_extra_work': shadow is not None,
                               'inference_backend': self.config.inference.backend,
                               'compile_mode': self.config.inference.compile_mode,
                               'cold_warmup_ms': self.warmup_ms}
        self._sync()
        timing['act_wall_ms'] = (time.perf_counter() - begin) * 1000
        result = {'action': self.chunk[self.step % self.config.replan_steps].copy(), 'replanned': replanned,
                  'policy_timing': timing, 'dual': dict(self._last_meta), 'step': self.step, 'shadow': shadow}
        self.step += 1
        oldest = self.step - self.config.updater.history - 1
        self.states = {n: value for n, value in self.states.items() if n >= oldest}
        self.actions = {n: value for n, value in self.actions.items() if n >= oldest}
        return result


def create_policy(config: SystemConfig, device='cuda', *, shadow_every_replans=0):
    if config.dual.enabled and config.dual.updater_enabled and not config.dual.updater_checkpoint:
        raise ValueError('Set dual.updater_checkpoint to a trained updater checkpoint')
    if config.inference.backend == 'stock':
        from openpi.dual.stock_compiled import StockCompiledBridge
        bridge = StockCompiledBridge(config.base_checkpoint, device=device,
                                     text_tokens=config.updater.text_tokens,
                                     compile_mode=config.inference.compile_mode)
    else:
        bridge = FrozenPi05(config.base_checkpoint, device=device, text_tokens=config.updater.text_tokens)
    updater = None
    if config.dual.enabled and config.dual.updater_enabled:
        updater, provenance = load_updater(config.dual.updater_checkpoint, config, device)
        updater.requires_grad_(False)
    return DualPolicy(bridge, config, updater, shadow_every_replans=shadow_every_replans)
