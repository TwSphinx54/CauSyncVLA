"""Per-step adapter for the original full-prefix Pi0.5 policy and TFWS.

This module never constructs a compact prefix or a learned updater. Both cache
arms use the same stock input/output transforms and the verified ROSE sampler.
"""
from __future__ import annotations

from dataclasses import replace
import threading
import time

import numpy as np
import torch

from openpi.dual.config import SystemConfig
from openpi.dual.rose_tfws import TrainingFreeTFWS, integration_schedule
from openpi.policies.policy_config import create_trained_policy
from openpi.training.config import get_config


class _StockPolicyView:
    """Expose the wrapper's model alias while forwarding sampler mutations.

    Upstream Policy stores _model and has no public model property. Forwarding
    assignments is essential: its bound infer() must see the replaced sampler.
    """
    def __init__(self, policy):
        object.__setattr__(self, '_raw', policy)

    def __getattr__(self, name):
        return getattr(self._raw, '_model' if name == 'model' else name)

    def __setattr__(self, name, value):
        setattr(self._raw, '_model' if name == 'model' else name, value)


class StockPolicy:
    def __init__(self, config: SystemConfig, device='cuda:0'):
        if config.dual.enabled:
            raise ValueError('stock backend cannot enable dual; use compact for the updater')
        self.config = config
        self.device = torch.device(device)
        cfg = get_config('pi05_libero')
        cfg = replace(cfg, model=replace(cfg.model, discrete_state_input=False, pytorch_compile_mode=None))
        self._raw = create_trained_policy(cfg, config.base_checkpoint,
                                         sample_kwargs={'num_steps': config.num_steps}, pytorch_device=str(self.device))
        self._raw._model.requires_grad_(False).eval()
        self._wrapper = TrainingFreeTFWS(
            _StockPolicyView(self._raw), enabled=config.cache.enabled, configured_steps=config.num_steps,
            restart_time=config.cache.restart_time, replan_steps=config.replan_steps,
            warm_nfe=config.cache.warm_nfe if config.cache.enabled else None)
        self._original_denoise = self._raw._model.denoise_step
        self._denoise_calls = 0

        def counted_denoise(*args, **kwargs):
            self._denoise_calls += 1
            return self._original_denoise(*args, **kwargs)

        self._raw._model.denoise_step = counted_denoise
        self._lock = threading.Lock()
        self._closed = False
        self.reset()

    def reset(self):
        self._wrapper.reset()
        self.step = 0
        self.prompt = None
        self.chunk = None
        self._last_meta = {}
        self._failed = False

    def close(self):
        if self._closed:
            return
        self.reset()
        self._raw._model.denoise_step = self._original_denoise
        self._raw._sample_actions = self._wrapper._stock_sample_actions
        self._raw._sample_kwargs = self._wrapper._stock_sample_kwargs
        self._raw._training_free_tfws_owner = None
        self._closed = True

    def _sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    @torch.no_grad()
    def act(self, obs, *, executed_action=None, noise=None):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('Concurrent stock act calls are unsupported')
        try:
            if self._closed or self._failed:
                raise RuntimeError('Stock policy is closed or failed; reset failed episodes')
            prompt = obs.get('prompt')
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError('Nonempty task prompt required')
            if self.prompt is not None and prompt != self.prompt:
                raise ValueError('Reset before changing task prompt')
            state = np.asarray(obs['observation/state'])
            if state.shape != (8,) or not np.isfinite(state).all():
                raise ValueError('Expected finite raw LIBERO state [8]')
            if self.step == 0:
                if executed_action is not None:
                    raise ValueError('First frame has no previous action')
            else:
                previous = np.asarray(executed_action)
                if previous.shape != (7,) or not np.isfinite(previous).all():
                    raise ValueError('Supply the previous actual finite action [7] each step')
            self.prompt = prompt
            self._sync()
            started = time.perf_counter()
            replanned = self.step % self.config.replan_steps == 0
            timing = {}
            if replanned:
                before = self._denoise_calls
                result = self._wrapper.infer(obs, noise=noise,
                                            executed_steps=0 if self.step == 0 else self.config.replan_steps)
                actual_nfe = self._denoise_calls - before
                meta = result['tfws']
                expected_nfe, _ = integration_schedule(self.config.num_steps, meta['start_time'])
                if meta['warm_start'] and self.config.cache.warm_nfe is not None:
                    expected_nfe = self.config.cache.warm_nfe
                if actual_nfe != expected_nfe or meta['effective_nfe'] != actual_nfe:
                    raise RuntimeError(f'Actual denoiser calls {actual_nfe} differ from requested {expected_nfe}')
                self.chunk = np.asarray(result['actions']).copy()
                if self.chunk.shape != (10, 7) or not np.isfinite(self.chunk).all():
                    raise ValueError('Stock output must be finite action chunk [10,7]')
                timing = dict(result['policy_timing'])
                self._last_meta = {
                    'snapshot_step': self.step, 'age': 0, 'updated': False,
                    'synchronous_age_fallback': False, 'warm_start': bool(meta['warm_start']),
                    'start_time': meta['start_time'], 'actual_nfe': actual_nfe, 'mode': 'synchronous',
                    'policy_backend': 'stock', 'prefix_layout': 'full',
                    'timing_scope': 'full_device_synchronized_policy_infer',
                    'virtual_clock_models_gpu_contention': False,
                }
            self._sync()
            timing['act_wall_ms'] = (time.perf_counter() - started) * 1000
            result = {'action': self.chunk[self.step % self.config.replan_steps].copy(),
                      'replanned': replanned, 'dual': dict(self._last_meta), 'policy_timing': timing, 'step': self.step}
            self.step += 1
            return result
        except Exception:
            self._failed = True
            self.chunk = None
            self._wrapper.reset()
            raise
        finally:
            self._lock.release()
