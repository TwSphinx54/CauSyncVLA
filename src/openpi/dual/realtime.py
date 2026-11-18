"""Causal inference worker for a continuously ticking simulator.

This opt-in adapter shares the canonical frozen bridge and history encoder.
It never changes legacy virtual evaluation or training. Cache is deliberately
excluded from this two-arm experiment. CPU scheduler/feedback code is model-free.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import hashlib
import threading
import time
import numpy as np

@dataclass(frozen=True)
class FeedbackFrame:
    step: int
    observation: dict
    previous_action: object
    observed_ns: int

class FeedbackHistory:
    def __init__(self, capacity=48):
        self.capacity = capacity
        self.step = -1
        self.states, self.actions = {}, {}
        self.prompt = None

    def ingest(self, frames):
        if not frames:
            raise ValueError('Every plan requires new observed feedback')
        for frame in frames:
            if frame.step != self.step + 1:
                raise ValueError('Feedback must retain every executed environment tick in order')
            prompt = frame.observation['prompt']
            if self.prompt is not None and prompt != self.prompt:
                raise ValueError('Prompt changed inside an episode')
            self.prompt = prompt
            state = np.asarray(frame.observation['observation/state'], np.float32)
            if state.shape != (8,) or not np.isfinite(state).all():
                raise ValueError('Invalid measured state')
            if frame.step == 0:
                if frame.previous_action is not None:
                    raise ValueError('Cold observation has no previous executed action')
            else:
                action = np.asarray(frame.previous_action, np.float32)
                if action.shape != (7,) or not np.isfinite(action).all():
                    raise ValueError('Missing actual executed action feedback')
                self.actions[frame.step-1] = action.copy()
            self.states[frame.step] = state.copy()
            self.step = frame.step
        oldest = self.step-self.capacity
        self.states = {k:v for k,v in self.states.items() if k >= oldest}
        self.actions = {k:v for k,v in self.actions.items() if k >= oldest}
        return frames[-1]

    def arrays(self, anchor):
        age = self.step-anchor
        if not 0 <= age <= self.capacity:
            raise ValueError('Unsupported anchor age')
        states = np.zeros((1,self.capacity,8),np.float32)
        actions = np.zeros((1,self.capacity,7),np.float32)
        for i in range(age):
            states[0,i] = self.states[anchor+i+1]
            actions[0,i] = self.actions[anchor+i]
        return states, actions, age

class ActionConsumer:
    """Consume exactly C commands per accepted plan, then hold until ready.

    Holding means zero arm command and unchanged last gripper command. It is
    recorded and returned through the same causal feedback path as every action.
    """
    def __init__(self, consumption):
        if consumption < 1 or consumption > 10:
            raise ValueError('Consumption must be in [1,10]')
        self.consumption = consumption
        self.plan = None
        self.used = consumption
        self.last_gripper = -1.

    @property
    def exhausted(self):
        return self.used >= self.consumption

    def adopt(self, plan):
        if not self.exhausted:
            raise ValueError('Do not overwrite an unconsumed plan')
        chunk = np.asarray(plan['actions'])
        if chunk.shape != (10,7) or not np.isfinite(chunk).all():
            raise ValueError('Invalid predicted chunk')
        self.plan = plan
        self.used = 0

    def action(self):
        if self.exhausted:
            return np.array([0.]*6+[self.last_gripper],np.float32), True, None
        offset = self.used
        action = np.array(self.plan['actions'][offset],copy=True)
        self.last_gripper = float(action[6])
        self.used += 1
        return action, False, offset

@dataclass(frozen=True)
class PublishedPrefix:
    frame: FeedbackFrame
    prefix: object
    capture_started_ns: int
    published_ns: int

class AsyncBrain:
    """A single worker always captures the newest *observed* frame available.

    Completed real prefixes alone enter the publication deque. The action worker
    takes a fixed snapshot of published entries at its own decision start.
    """
    def __init__(self, bridge, initial, device):
        self.bridge, self.device = bridge, device
        self.condition = threading.Condition()
        self.latest = initial.frame
        self.published = deque([initial],maxlen=64)
        self.last_captured = initial.frame.step
        self.events = []
        self.stopped = False
        self.error = None
        self.thread = threading.Thread(target=self._run,name='real-prefix-worker',daemon=False)
        self.thread.start()

    def observe(self, frame):
        with self.condition:
            if self.error is not None:
                raise RuntimeError('Background VLM failed') from self.error
            if frame.step <= self.latest.step:
                raise ValueError('Brain observations must advance')
            self.latest = frame
            self.condition.notify()

    def available(self, decision_step):
        with self.condition:
            if self.error is not None:
                raise RuntimeError('Background VLM failed') from self.error
            candidates = [x for x in self.published if x.frame.step <= decision_step]
            return max(candidates,key=lambda x:x.frame.step) if candidates else None

    def publish_sync(self, snapshot):
        with self.condition:
            self.published.append(snapshot)
            self.events.append({'capture_step':snapshot.frame.step,'observation_ns':snapshot.frame.observed_ns,
                                'capture_started_ns':snapshot.capture_started_ns,'published_ns':snapshot.published_ns,
                                'synchronous_fallback':True})

    def _run(self):
        import torch
        try:
            with torch.cuda.device(self.device):
                stream = torch.cuda.Stream(device=self.device)
                while True:
                    with self.condition:
                        self.condition.wait_for(lambda:self.stopped or self.latest.step > self.last_captured)
                        if self.stopped:
                            return
                        frame = self.latest
                        self.last_captured = frame.step
                    start = time.monotonic_ns()
                    with torch.cuda.stream(stream), torch.no_grad():
                        prefix = self.bridge.prefix([frame.observation])
                        stream.synchronize()
                    done = time.monotonic_ns()
                    snap = PublishedPrefix(frame,prefix,start,done)
                    with self.condition:
                        self.published.append(snap)
                        self.events.append({'capture_step':frame.step,'observation_ns':frame.observed_ns,
                                            'capture_started_ns':start,'published_ns':done})
        except BaseException as error:
            with self.condition:
                self.error = error
                self.condition.notify_all()

    def close(self):
        with self.condition:
            self.stopped = True
            self.condition.notify_all()
        self.thread.join()
        if self.error is not None:
            raise RuntimeError('Background VLM failed') from self.error

class RealtimeEngine:
    """One action worker plus an independent real VLM publication worker."""
    def __init__(self, config, device='cuda:0'):
        import torch
        from openpi.dual.bridge import FrozenPi05,file_sha256
        from openpi.dual.policy import load_updater
        if config.inference.backend != 'compact' or config.inference.compile_mode is not None:
            raise ValueError('Fixed-20Hz experiment adapter is frozen to compact eager; use DualPolicy for stock compiled inference')
        if config.cache.enabled or config.num_steps != 1 or config.replan_steps != 5:
            raise ValueError('Matched realtime experiment fixes cache off, NFE1, H10/C5')
        if config.dual.enabled and config.dual.mode != 'threaded':
            raise ValueError('Realtime publication cannot use the virtual clock')
        self.config, self.device = config, torch.device(device)
        self.bridge = FrozenPi05(config.base_checkpoint,device=device,text_tokens=config.updater.text_tokens)
        self.updater = None
        if config.dual.enabled:
            if not config.dual.updater_enabled:
                raise ValueError('This experiment compares fresh and updater only')
            self.updater, provenance = load_updater(config.dual.updater_checkpoint,config,self.device)
            if provenance.get('base_sha256') != file_sha256(config.base_checkpoint+'/model.safetensors'):
                raise ValueError('Updater and base checkpoint identities differ')
        self.stream = torch.cuda.Stream(device=self.device)
        self.brain = None
        self.last_brain_events = []
        self.history = None
        self.initial = None

    def warmup(self, obs, repeats=100):
        """Compute-only warmup at one real initial observation, age zero.

        No invented execution enters policy history; all outputs are discarded.
        It is excluded from both success and latency samples.
        """
        import torch
        from openpi.dual.history import condition_features
        with torch.cuda.device(self.device),torch.cuda.stream(self.stream),torch.no_grad():
            for _ in range(repeats):
                p=self.bridge.prefix([obs])
                if self.updater is not None:
                    state=torch.as_tensor(obs['observation/state'][None],device=self.device)
                    condition=condition_features(state,state,torch.zeros((1,48,8),device=self.device),
                        torch.zeros((1,48,7),device=self.device),torch.zeros(1,device=self.device,dtype=torch.long),self.updater.norm,48)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        self.updater(p.k,p.v,p.mask,condition)
                self.bridge.hand(p.k,p.v,p.mask,p.state,torch.zeros((1,10,32),device=self.device),num_steps=1)
            self.stream.synchronize()

    def start_episode(self):
        self.close_episode()
        self.history=FeedbackHistory(self.config.updater.history)
        self.initial=None
        self.last_brain_events=[]

    def observe(self, frame):
        if self.brain is not None:
            self.brain.observe(frame)

    def plan(self, frames, innovation, plan_id):
        import torch
        from openpi.dual.history import condition_features
        with torch.cuda.device(self.device),torch.cuda.stream(self.stream),torch.no_grad():
            self.stream.synchronize()
            start=time.monotonic_ns()
            frame=self.history.ingest(frames)
            fallback=False
            prefix_start=time.monotonic_ns()
            if not self.config.dual.enabled or plan_id==0:
                prefix=self.bridge.prefix([frame.observation]);self.stream.synchronize()
                snap=PublishedPrefix(frame,prefix,prefix_start,time.monotonic_ns())
                if plan_id==0:self.initial=snap
            else:
                snap=self.brain.available(frame.step)
                if snap is None or frame.step-snap.frame.step > self.config.dual.max_age:
                    prefix=self.bridge.prefix([frame.observation]);self.stream.synchronize()
                    snap=PublishedPrefix(frame,prefix,prefix_start,time.monotonic_ns());fallback=True
                    self.brain.publish_sync(snap)
            selection_ns=time.monotonic_ns()
            prefix_ms=(selection_ns-prefix_start)/1e6
            if snap.published_ns > time.monotonic_ns() or snap.frame.step > frame.step:
                raise ValueError('Unpublished or future anchor')
            p=snap.prefix
            for tensor in (p.k,p.v,p.mask,p.state):tensor.record_stream(self.stream)
            k,v=p.k,p.v
            age=frame.step-snap.frame.step
            updated=bool(self.config.dual.enabled and age>=self.config.dual.updater_min_age)
            update_start=time.monotonic_ns()
            if updated:
                states,actions,encoded_age=self.history.arrays(snap.frame.step)
                to=lambda value:torch.as_tensor(value,device=self.device)
                condition=condition_features(to(snap.frame.observation['observation/state'][None]),
                    to(frame.observation['observation/state'][None]),to(states),to(actions),to([encoded_age]),
                    self.updater.norm,self.config.updater.history)
                with torch.autocast('cuda',dtype=torch.bfloat16):k,v=self.updater(k,v,p.mask,condition)
            self.stream.synchronize();update_ms=(time.monotonic_ns()-update_start)/1e6
            hand_start=time.monotonic_ns()
            noise=torch.as_tensor(innovation[None],dtype=torch.float32,device=self.device)
            endpoint=self.bridge.hand(k,v,p.mask,p.state,noise,num_steps=1)
            self.stream.synchronize();hand_ms=(time.monotonic_ns()-hand_start)/1e6
            actions=self.bridge.output_actions(endpoint,p.state)
            self.stream.synchronize();ready=time.monotonic_ns()
            result={'plan_id':plan_id,'decision_step':frame.step,'observation_ns':frame.observed_ns,
                    'innovation_sha256':hashlib.sha256(np.ascontiguousarray(innovation).tobytes()).hexdigest(),
                    'started_ns':start,'ready_ns':ready,'actions':actions,
                    'snapshot_step':snap.frame.step,'snapshot_observation_ns':snap.frame.observed_ns,
                    'snapshot_published_ns':snap.published_ns,'anchor_selected_ns':selection_ns,'age':age,'updated':updated,
                    'synchronous_age_fallback':fallback,'actual_nfe':1,'prefix_ms':prefix_ms,
                    'updater_ms':update_ms,'hand_ms':hand_ms,'plan_wall_ms':(ready-start)/1e6,
                    'history_age_outside_training':bool(updated and not 5<=age<=45)}
            # No background task is live during cold prefix/expert initialization.
            if plan_id==0 and self.config.dual.enabled:
                self.brain=AsyncBrain(self.bridge,self.initial,self.device)
            return result

    def close_episode(self):
        if self.brain is not None:
            brain,self.brain=self.brain,None
            brain.close()
            self.last_brain_events=list(brain.events)

    def close(self):
        self.close_episode()
