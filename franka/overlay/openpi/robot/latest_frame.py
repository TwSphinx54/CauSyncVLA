"""One real-prefix worker, always taking the latest available observation."""
import threading
import time
import numpy as np


class LatestFrameWorker:
    def __init__(self, capture, initial):
        self.capture = capture
        self.completed = initial
        self.published_history = [initial]
        self.last_captured = initial.step
        self.latest = None
        self.observation_version = 0
        self.captured_version = 0
        self.condition = threading.Condition()
        self.stopped = False
        self.paused = False
        self.in_flight = False
        self.epoch = 0
        self.error = None
        self.capture_count = 0
        self.last_event = None
        self.events = []
        self.thread = threading.Thread(target=self._run, name='fr3-real-prefix', daemon=False)
        self.thread.start()

    def observe(self, observation, step):
        frozen = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in observation.items()}
        with self.condition:
            if self.error is not None:
                raise RuntimeError('Real-prefix worker failed') from self.error
            if self.paused:return
            self.observation_version += 1
            self.latest = (step, frozen)
            self.condition.notify_all()

    def published(self, decision_step):
        with self.condition:
            if self.error is not None:
                raise RuntimeError('Real-prefix worker failed') from self.error
            eligible=[s for s in self.published_history if s.step<=decision_step]
            if not eligible:raise RuntimeError('No causal published visual anchor')
            return eligible[-1]

    def telemetry(self):
        with self.condition:
            events, self.events = self.events, []
            return dict(capture_count=self.capture_count, last_completed=self.last_event,
                        events=events, clock='server_monotonic_ns')

    def _run(self):
        try:
            while True:
                with self.condition:
                    self.condition.wait_for(lambda: self.stopped or (
                        not self.paused and self.latest is not None and self.observation_version > self.captured_version))
                    if self.stopped:
                        return
                    step, observation = self.latest
                    self.last_captured = step
                    self.captured_version = self.observation_version
                    window_id = self.captured_version
                    epoch = self.epoch
                    self.in_flight = True
                started = time.monotonic_ns()
                with self.condition:
                    self.events.append(dict(kind='slow_start', window_id=window_id,
                        capture_step=step, timestamp_ns=started))
                snapshot = self.capture(observation, step)
                # capture synchronizes its own CUDA stream before returning.
                done = time.monotonic_ns()
                with self.condition:
                    self.in_flight = False
                    if self.paused or epoch != self.epoch:
                        self.events.append(dict(kind="slow_discard",window_id=window_id,capture_step=step,timestamp_ns=done))
                        self.condition.notify_all()
                        continue
                    if snapshot.step >= self.completed.step:
                        self.completed = snapshot
                        self.published_history.append(snapshot)
                        self.published_history=self.published_history[-4:]
                    self.capture_count += 1
                    self.last_event = dict(capture_step=step, started_ns=started, published_ns=done)
                    self.events.append(dict(kind='slow_publish', window_id=window_id,
                        capture_step=step, timestamp_ns=done))
                    self.condition.notify_all()
                # No timer or cadence gate: immediately take the newest observed frame.
        except BaseException as exc:
            with self.condition:
                self.in_flight = False
                self.error = exc
                self.condition.notify_all()

    def pause(self):
        with self.condition:
            self.paused=True;self.epoch+=1;self.latest=None
            self.condition.wait_for(lambda:not self.in_flight or self.error is not None)
            if self.error is not None:raise RuntimeError('Real-prefix worker failed') from self.error
    def reanchor(self,observation,step):
        with self.condition:
            if not self.paused or self.in_flight:raise RuntimeError('Pause before real reanchor')
            self.observation_version+=1;wid=self.observation_version
            self.captured_version=wid;self.latest=None
            started=time.monotonic_ns()
            self.events.append(dict(kind='slow_start',window_id=wid,capture_step=step,timestamp_ns=started))
        snapshot=self.capture(observation,step)
        done=time.monotonic_ns()
        with self.condition:
            self.completed=snapshot;self.published_history=[snapshot]
            self.capture_count+=1;self.last_captured=step
            self.last_event=dict(capture_step=step,started_ns=started,published_ns=done)
            self.events.append(dict(kind='slow_publish',window_id=wid,capture_step=step,timestamp_ns=done))
            self.paused=False;self.condition.notify_all()
        return snapshot,done

    def close(self):
        with self.condition:
            self.stopped = True
            self.condition.notify_all()
        self.thread.join()
