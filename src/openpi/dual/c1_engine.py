"""Stock compiled PI0.5 inference with causal feedback from a 20 Hz C1 controller."""
import time
import torch

from openpi.dual.history import condition_features
from openpi.dual.policy import create_policy
from openpi.dual.realtime import AsyncBrain, PublishedPrefix, RealtimeEngine
from openpi.dual.rose_tfws import resolve_initial_latent


class CompiledC1Engine(RealtimeEngine):
    def __init__(self, config, device='cuda:0'):
        if not (config.replan_steps in (1,5) and config.inference.backend == 'stock'
                and config.inference.compile_mode == 'max-autotune-no-cudagraphs'):
            raise ValueError('Expected C1 compiled stock inference')
        if config.dual.enabled:
            if not (config.dual.mode == 'threaded' and config.dual.updater_enabled
                    and config.cache.enabled and config.num_steps == config.cache.warm_nfe == 1
                    and config.cache.restart_time == .5):
                raise ValueError('Expected threaded final M3+TFWS configuration')
        elif config.num_steps not in (1,10) or config.cache.enabled:
            raise ValueError('Expected original synchronous 10-step comparison arm')
        policy = create_policy(config, device=device)
        self.bridge, self.updater = policy.bridge, policy.updater
        self.history_norm = policy._history_norm
        policy.close()
        self.config, self.device = config, torch.device(device)
        self.stream = torch.cuda.Stream(device=self.device)
        self.brain, self.last_brain_events = None, []

    def warmup(self, obs):
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream), torch.no_grad():
            prefix = self.bridge.prefix([obs])
            self.bridge.warmup(prefix, self.updater, self.config)
            self.stream.synchronize()

    def plan(self, frames, innovation, plan_id, previous_endpoint):
        start = time.monotonic_ns()
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream), torch.no_grad():
            frame = self.history.ingest(frames)
            snap = self.brain.available(frame.step) if self.brain is not None else None
            fallback = self.brain is not None and (snap is None or frame.step-snap.frame.step > self.config.dual.max_age)
            if snap is None or fallback:
                capture = time.monotonic_ns()
                prefix = self.bridge.prefix([frame.observation])
                self.stream.synchronize()
                snap = PublishedPrefix(frame, prefix, capture, time.monotonic_ns())
                if fallback:
                    self.brain.publish_sync(snap)
            selected_ns = time.monotonic_ns()
            p = snap.prefix
            if self.brain is not None:
                # Prefix storage can originate on the background CUDA stream.
                # Protect the native DynamicCache (avoid stacking or copying it).
                for i in range(len(p.cache)):
                    for tensor in p.cache[i]:
                        tensor.record_stream(self.stream)
                p.mask.record_stream(self.stream)
                p.state.record_stream(self.stream)
            age = frame.step-snap.frame.step
            update = self.updater is not None and age >= self.config.dual.updater_min_age
            noise = torch.as_tensor(innovation[None], device=self.device)
            latent, flow_time, warm = resolve_initial_latent(noise, previous_endpoint,
                enabled=self.config.cache.enabled, replan_steps=self.config.replan_steps, restart_time=self.config.cache.restart_time)
            nfe = self.config.cache.warm_nfe if warm else self.config.num_steps
            if update:
                states, actions, _ = self.history.arrays(snap.frame.step)
                condition = condition_features(torch.from_numpy(snap.frame.observation['observation/state'][None]),
                    torch.from_numpy(frame.observation['observation/state'][None]), torch.from_numpy(states),
                    torch.from_numpy(actions), torch.tensor([age]), self.history_norm, self.config.updater.history)
                endpoint = self.bridge.fast(self.updater, p, condition.to(self.device), latent,
                                            num_steps=nfe, start_time=flow_time)
            else:
                endpoint = self.bridge.expert(p, latent, num_steps=nfe, start_time=flow_time)
            actions = self.bridge.output_actions(endpoint, p.state)
            next_endpoint = endpoint.detach().clone() if self.config.cache.enabled else None
            self.stream.synchronize()
            ready = time.monotonic_ns()
            if plan_id == 0 and self.config.dual.enabled:
                self.brain = AsyncBrain(self.bridge, snap, self.device)
            return dict(plan_id=plan_id, decision_step=frame.step, observation_ns=frame.observed_ns,
                        started_ns=start, ready_ns=ready, plan_wall_ms=(ready-start)/1e6,
                        actions=actions, endpoint=next_endpoint, snapshot_step=snap.frame.step,
                        snapshot_published_ns=snap.published_ns, anchor_selected_ns=selected_ns,
                        age=age, actual_nfe=nfe, updated=update, warm_start=warm,
                        synchronous_age_fallback=fallback)
