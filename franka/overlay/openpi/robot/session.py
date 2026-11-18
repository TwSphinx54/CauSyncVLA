"""Per-tick OpenPI wire contract. No ML import is needed to test the protocol."""
import time
import numpy as np

PROTOCOL = 'chronovla.fr3.tick.v1'
CAMERAS = ('observation/image', 'observation/wrist_image')


def observation_from(request):
    obs = {key: request[key] for key in (*CAMERAS, 'observation/state', 'prompt')}
    state = np.asarray(obs['observation/state'], dtype=np.float32)
    if state.shape != (7,) or not np.isfinite(state).all():
        raise ValueError('State must be finite absolute xyz/Euler/commanded-gripper [7]')
    if not isinstance(obs['prompt'], str) or not obs['prompt'].strip():
        raise ValueError('Use the task instruction from the training dataset')
    obs['observation/state'] = state.copy()
    for key in CAMERAS:
        image = np.asarray(obs[key])
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f'{key} must be HWC uint8 RGB')
        obs[key] = image.copy()
    stamp=request.get('observed_ns',request.get('feedback_observed_ns'))
    if stamp is not None:
        if type(stamp) is not int or stamp<=0:raise ValueError('Feedback timestamp must be positive monotonic nanoseconds')
        obs['feedback_observed_ns']=stamp
    return obs


class SessionPolicy:
    """One active episode, one returned action per executed control interval."""
    def __init__(self, policy, metadata):
        self.policy, self.metadata = policy, metadata
        self.episode = None
        self.next_step = 0
        self.plan_counter = 0
        self.plan_id = None
        self.failed = False
        self.prompt = None

    def reset(self):
        self.policy.reset()
        self.episode, self.prompt = None, None
        self.next_step = 0
        self.plan_counter = 0
        self.plan_id = None
        self.failed = False

    def _feedback(self, request, obs):
        if request.get('episode_id') != self.episode or self.episode is None:
            raise ValueError('Start a new episode before inference')
        step = request.get('step_index')
        if type(step) is not int or step != self.next_step:
            raise ValueError(f'Expected step_index={self.next_step}; feedback cannot skip or repeat ticks')
        if self.prompt is not None and obs['prompt'] != self.prompt:
            raise ValueError('Reset before changing task instruction')
        action = request.get('executed_action')
        if step == 0:
            if action is not None:
                raise ValueError('The first observation has no previous executed action')
        else:
            action = np.asarray(action, dtype=np.float32)
            if action.shape != (7,) or not np.isfinite(action).all():
                raise ValueError('Return the actual applied command, after clipping/controller processing')
            if not np.isclose(obs['observation/state'][6], action[6], atol=1e-5):
                raise ValueError('state[6] must be the previous commanded gripper, not measured opening')
        return action

    def infer(self, request):
        op = request.get('op', 'step')
        if op == 'reset':
            episode = request.get('episode_id')
            if not isinstance(episode, str) or not episode:
                raise ValueError('episode_id is required')
            seed=request.get('seed')
            if seed is not None and (type(seed) is not int or not 0<=seed<2**63):
                raise ValueError('seed must be an integer in [0, 2**63)')
            self.reset()
            if seed is not None:
                self.policy.set_episode_seed(seed)
            self.episode = episode
            return dict(ready=True, episode_id=episode)
        if op == 'warmup':
            if self.episode is not None:
                raise ValueError('Warm up before starting an episode')
            try:
                result = self.policy.act(observation_from(request), executed_action=None)
                return dict(ready=True, cold_warmup_ms=result['dual'].get('cold_warmup_ms', 0.))
            finally:
                self.reset()  # Discard every warmup action/history/cache.
        if self.failed:
            raise RuntimeError('Reset after the failed request')
        if op not in ('step', 'finish'):
            raise ValueError('Unknown protocol operation')
        obs = observation_from(request)
        previous = self._feedback(request, obs)
        if op == 'finish':
            count = self.next_step
            finish = getattr(self.policy, 'finish_telemetry', None)
            slow = finish() if finish else None
            self.reset()
            return dict(finished=True, executed_transitions=count, slow_worker=slow)
        self.prompt = obs['prompt']
        started = time.monotonic_ns()
        try:
            result = self.policy.act(obs, executed_action=previous)
            action = np.asarray(result.pop('action'), dtype=np.float32)
            if action.shape != (7,) or not np.isfinite(action).all():
                raise ValueError('Invalid model command')
        except Exception:
            self.failed = True
            raise
        if result.get('replanned'):
            self.plan_id = f'{self.episode}:p{self.plan_counter}'
            self.plan_counter += 1
            self.plan_start_step = request['step_index']
        self.next_step += 1
        return dict(plan_id=self.plan_id,
            action_index_in_plan=request['step_index']-getattr(self, 'plan_start_step', request['step_index']),
            server_clock='monotonic_ns', actions=action[None], episode_id=self.episode,
            decision_step=request['step_index'], received_ns=started, ready_ns=time.monotonic_ns(),
            **result)
