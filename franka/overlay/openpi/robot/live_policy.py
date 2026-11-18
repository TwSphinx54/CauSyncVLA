"""Scheduling-only extension of the reviewed dual policy; model graphs unchanged."""
import time
import torch
from openpi.dual.policy import DualPolicy, load_updater
from openpi.dual.stock_compiled import StockCompiledBridge
from openpi.robot.latest_frame import LatestFrameWorker


class RobotPolicy(DualPolicy):
    def __init__(self, *args, **kwargs):
        self._live_brain = None
        self._frame_feed = None
        self.visual_mailbox = None
        super().__init__(*args, **kwargs)

    def reset(self):
        if getattr(self, '_frame_feed', None) is not None:
            self._frame_feed.close();self._frame_feed=None
        self.visual_mailbox=None
        if self._live_brain is not None:
            self._live_brain.close()
            self._live_brain = None
        super().reset()
        self._episode_generator = None
        self._gripper_paused=False
        self._resume_snapshot=None

    def set_episode_seed(self, seed):
        self._episode_generator = torch.Generator(device=self.device).manual_seed(seed)

    def _advance_brain(self, obs):
        if not self.config.dual.enabled:
            return
        if self.snapshot is None:
            self.snapshot = self._capture(obs, self.step)
            self._live_brain = LatestFrameWorker(self._background_capture, self.snapshot)
            if self.visual_mailbox:
                from openpi.robot.frame_mailbox import FrameMailbox
                self._frame_feed=FrameMailbox(self.visual_mailbox,self._live_brain)
        else:
            if not self.visual_mailbox:self._live_brain.observe(obs, self.step)
            candidate = self._live_brain.published(self.step)
            if candidate.step >= self.snapshot.step:
                self.snapshot = candidate

    def pause_gripper(self):
        self._gripper_paused=True
        if self._frame_feed is not None:self._frame_feed.close();self._frame_feed=None
        if self._live_brain is not None:self._live_brain.pause()
        return self._live_brain.telemetry() if self._live_brain else None

    def resume_gripper(self,obs,step):
        if not self._gripper_paused or step!=self.step:raise ValueError('Invalid gripper resume sequence')
        self.chunk=None;self.endpoint=None;self.endpoint_state=None
        self.states={step:obs['observation/state'].copy()};self.actions={}
        self.feedback_times={}  # Exclude the gripper pause from the next temporal history.
        self._last_meta={};self._chunk_start_step=step
        if self._live_brain is not None:
            self.snapshot,done=self._live_brain.reanchor(obs,step)
        else:
            self.snapshot=self._capture(obs,step);done=time.monotonic_ns()
        self._resume_snapshot=self.snapshot if not self.config.dual.enabled else None
        self._gripper_paused=False
        if self._live_brain is not None and self.visual_mailbox:
            from openpi.robot.frame_mailbox import FrameMailbox
            self._frame_feed=FrameMailbox(self.visual_mailbox,self._live_brain,skip_current=True,initial_frame_key=str([(k,v.get("seq")) for k,v in sorted(obs.get("camera_metadata",{}).items())]))
        return done

    def _fresh_action_snapshot(self,obs):
        if self._resume_snapshot is not None:
            snapshot=self._resume_snapshot;self._resume_snapshot=None;return snapshot
        return super()._fresh_action_snapshot(obs)

    def finish_telemetry(self):
        if self._live_brain is None:
            return None
        self._live_brain.close()
        return self._live_brain.telemetry()

    def act(self, *args, **kwargs):
        if self._gripper_paused:raise RuntimeError("Policy paused for gripper synchronization")
        if self._episode_generator is not None and self._needs_replan() and kwargs.get('noise') is None:
            kwargs['noise'] = torch.randn((1,10,32), device=self.device, generator=self._episode_generator)
        result = super().act(*args, **kwargs)
        result['slow_worker'] = self._live_brain.telemetry() if self._live_brain else None
        return result


def create_robot_policy(config, *, source_base_checkpoint, device='cuda:0'):
    if config.robot_profile != 'fr3' or config.inference.backend != 'stock':
        raise ValueError('This entrypoint requires the FR3 state-on stock profile')
    if config.dual.mode != 'threaded':
        raise ValueError('Physical deployment uses measured threaded execution')
    updater = None
    if config.dual.enabled and config.dual.updater_enabled:
        if not config.dual.updater_checkpoint:
            raise ValueError('Supply the trained FR3 updater checkpoint')
        updater, provenance = load_updater(config.dual.updater_checkpoint, config, device)
        updater.requires_grad_(False)
    # The supplied local path may differ from the original ROSE training path.
    # Relocation changes no weight or normalization contents; no model hashing.
    bridge = StockCompiledBridge(config.base_checkpoint, device=device,
        text_tokens=config.updater.text_tokens, compile_mode=config.inference.compile_mode,
        robot_profile='fr3')
    return RobotPolicy(bridge, config, updater)
