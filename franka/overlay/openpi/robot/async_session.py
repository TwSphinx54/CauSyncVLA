"""10 Hz model ticks with independent camera publication and servo transport."""
from pathlib import Path
from openpi.robot.session import SessionPolicy, observation_from

class AsyncSessionPolicy(SessionPolicy):
    def infer(self, request):
        op = request.get('op')
        if op == 'drain':
            if self.episode is None or request.get('episode_id') != self.episode:
                raise ValueError('No matching episode to drain')
            out = self.policy.finish_telemetry()
            count = self.next_step
            self.reset()
            return dict(finished=True, feedback_rows=count, slow_worker=out)
        if op == 'gripper_pause':
            if self.episode is None or request.get('episode_id')!=self.episode or request.get('step_index')!=self.next_step:
                raise ValueError('Pause requires the current episode and next feedback index')
            return dict(paused=True,slow_worker=self.policy.pause_gripper())
        if op == 'gripper_resume':
            resumed=self._resume(request)
            return dict(reanchored=True,gripper_reanchor_ready_ns=resumed,
                        slow_worker=self.policy._live_brain.telemetry() if getattr(self.policy,'_live_brain',None) else None)
        if op == 'compatible_step':
            if self.episode is None or request.get('episode_id') != self.episode:
                raise ValueError('Reset before compatible inference')
            mailbox = Path(request['visual_mailbox']).resolve()
            if mailbox.parent != Path('/dev/shm/chronovla-fr3').resolve():
                raise ValueError('Invalid local visual mailbox')
            self.policy.visual_mailbox = str(mailbox)
            resumed=None
            if request.get('resume_after_gripper'):
                resumed=self._resume(request)
            result=super().infer(dict(request, op='step'))
            if resumed is not None:result['gripper_reanchor_ready_ns']=resumed
            return result
        # Old high-rate batches and partial-time adapters are not active protocols.
        return super().infer(request)

    def _resume(self,request):
        obs=observation_from(request);self._feedback(request,obs)
        complete=request['gripper_completed_ns'];meta=request.get('camera_metadata') or {}
        if request.get('observed_ns',0)<=complete or set(meta)!={'view1','hand'} or any(v.get('t_arrival_ns',0)<=complete for v in meta.values()):
            raise ValueError('Resume requires fresh state and both frames after gripper completion')
        obs['camera_metadata']=meta
        return self.policy.resume_gripper(obs,self.next_step)
