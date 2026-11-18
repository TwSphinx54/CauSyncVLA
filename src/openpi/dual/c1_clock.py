"""C1 controller with one in-flight plan and absolute wall-clock command deadlines."""
from concurrent.futures import ThreadPoolExecutor
import time
import os
from openpi.dual.deadline_dispatch import Dispatch

import numpy as np

from openpi.dual.realtime import ActionConsumer, FeedbackFrame

PERIOD_NS = 50_000_000


def _record(plan, submitted_ns, deadline):
    row = {k: v for k, v in plan.items() if k not in {'actions', 'endpoint'}}
    row.update(submitted_ns=submitted_ns, target_deadline_ns=deadline,
               observation_to_ready_ms=(row['ready_ns'] - row['observation_ns']) / 1e6)
    return row


def _distribution(values):
    if not values:
        return dict(n=0, mean=None, p50=None, p95=None, max=None)
    return dict(n=len(values), mean=float(np.mean(values)), p50=float(np.median(values)),
                p95=float(np.percentile(values, 95)), max=float(np.max(values)))


def run_episode(env, engine, initial_observation, prompt, rng, max_steps, observation_element,
                *, period_ns=PERIOD_NS, now=time.monotonic_ns, sleep=time.sleep, executor=None):
    """Release a request after env.step; adopt results only at scheduled boundaries.

    The first command is available before the timed clock. A missed slot holds the
    arm/gripper; a late result can supply one command at the next eligible tick.
    """
    engine.start_episode()
    consumer = Dispatch(engine.config.replan_steps,os.environ['CHRONO_MISS']=='tail')
    pending_frames = []
    ticks, plans = [], []
    future = None
    pool = executor or ThreadPoolExecutor(max_workers=1, thread_name_prefix='c1-action')
    first = FeedbackFrame(0, observation_element(initial_observation, prompt), None, now())
    try:
        issued = now()
        cold = pool.submit(engine.plan, [first], rng.standard_normal((10, 32), dtype=np.float32),
                           0, None).result()
        plans.append(_record(cold, issued, None))
        consumer.adopt(cold,0)
        selected = cold
        cached_endpoint = None
        origin = now()
        for step in range(max_steps):
            deadline = origin + step * period_ns
            remaining = (deadline - now()) / 1e9
            if remaining > 0:
                sleep(remaining)
            if future is not None and future.done():
                candidate = future.result()
                # The thread may wake late. A result completed during that
                # overshoot is eligible only at the following absolute boundary.
                if candidate['ready_ns'] <= deadline:
                    if consumer.adopt(candidate,step):
                        selected = candidate
                    plans.append(_record(candidate, issued, target))
                    future = None
            values,source,offset=consumer.command(step)
            action=np.asarray(values,dtype=np.float32);hold=(source=='hold')
            apply_ns = now()
            obs, _, done, _ = env.step(action.tolist())
            observed_ns = now()
            if not hold:
                # Publish only after the command was actually sent to env.step.
                cached_endpoint = selected['endpoint']
            ticks.append(dict(step=step, scheduled_ns=deadline, applied_ns=apply_ns,
                              observed_ns=observed_ns, hold=hold, source=source, action_offset=offset,
                              action=action.tolist(), plan_id=None if hold else selected['plan_id'],
                              decision_step=None if hold else selected['decision_step']))
            if done or step + 1 == max_steps:
                break
            frame = FeedbackFrame(step + 1, observation_element(obs, prompt), action.copy(), observed_ns)
            ticks[-1]['feedback_prepared_ns'] = now()
            engine.observe(frame)
            pending_frames.append(frame)
            # Noise is keyed to the observation tick, even if a busy worker
            # skips that request, so arms share innovations for matching ticks.
            innovation = rng.standard_normal((10, 32), dtype=np.float32)
            ticks[-1]['decision_due_next']=consumer.need(step+1)
            ticks[-1]['due_busy']=ticks[-1]['decision_due_next'] and future is not None
            if future is None and consumer.need(step+1):
                issued, target = now(), origin + (step + 1) * period_ns
                future = pool.submit(engine.plan, pending_frames, innovation,
                                     len(plans), cached_endpoint)
                pending_frames = []
            ticks[-1]['feedback_dispatched_ns'] = now()
        if future is not None:
            plans.append(_record(future.result(), issued, target))
        engine.close_episode()
    finally:
        if executor is None:
            pool.shutdown(wait=True, cancel_futures=False)
        engine.close_episode()

    warm, measured = plans[1:], ticks[1:]
    fresh = sum(t['source'] == 'fresh' for t in measured)
    late = sum(t['source'] == 'late' for t in measured)
    jitter = [(t['applied_ns'] - t['scheduled_ns']) / 1e6 for t in measured]
    tids = {t['plan_id']: t['step'] for t in ticks if t['source'] in ('cold','fresh','late')}
    for plan in plans:
        plan['adopted_step'] = tids.get(plan['plan_id'])
    span = (ticks[-1]['applied_ns'] - ticks[0]['applied_ns']) / 1e9
    result = dict(success=bool(done), error=None, actions_executed=len(ticks),
                  eligible_ticks=len(measured), fresh_on_time_ticks=fresh,
                  late_adopt_ticks=late, new_action_ticks=fresh+late,
                  hold_ticks=sum(t['hold'] for t in measured),
                  request_count=len(warm), busy_skip_ticks=len(measured)-len(warm),
                  warm_requests_ready_50ms=sum(p['ready_ns'] - p['observation_ns'] <= 50_000_000 for p in warm),
                  warm_requests_ready_within_period=sum(p['ready_ns'] - p['observation_ns'] <= period_ns for p in warm),
                  warm_requests_ready_by_target=sum(p['ready_ns'] <= p['target_deadline_ns'] for p in warm),
                  deadline_jitter_ms=_distribution(jitter),
                  clock_overrun_ticks=sum(t['applied_ns'] - t['scheduled_ns'] >= period_ns for t in measured),
                  command_span_seconds=span, command_execution_hz=len(measured)/span if span else None,
                  duration_seconds=(ticks[-1]['observed_ns']-origin)/1e9,
                  environment_step_ms=_distribution([(t['observed_ns']-t['applied_ns'])/1e6 for t in ticks]),
                  cold_start_ms=plans[0]['plan_wall_ms'],
                  observation_to_ready_ms=_distribution([p['observation_to_ready_ms'] for p in warm]))
    result.update(committed_ticks=sum(t['source']=='committed' for t in measured),
                  tail_ticks=sum(t['source']=='tail' for t in measured),
                  consumption=consumer.c,miss_policy=os.environ['CHRONO_MISS'],
                  request_opportunities=sum(t.get('decision_due_next',False) for t in ticks))
    result['busy_skip_ticks']=sum(t.get('due_busy',False) for t in ticks)
    assert sum(result[k] for k in ('fresh_on_time_ticks','late_adopt_ticks','hold_ticks','committed_ticks','tail_ticks'))==len(measured)
    return result, dict(origin_ns=origin, ticks=ticks, plans=plans, real_prefix_publications=engine.last_brain_events)
