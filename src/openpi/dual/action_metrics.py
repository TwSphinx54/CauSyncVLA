"""Lightweight command continuity receipts, independent of Torch and LIBERO.

These are differences in the seven commands passed to env.step(), after policy
output transforms and before any clipping/scaling inside the environment. They
are not Cartesian motion, acceleration, jerk, or executed actuator feedback.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import uuid


SCHEMA_VERSION = 2
GRIPPER_DEADBAND = 1e-6


def contract(replan_steps, gripper_deadband=GRIPPER_DEADBAND):
    if type(replan_steps) is not int or replan_steps < 1:
        raise ValueError('replan_steps must be a positive integer')
    if not math.isfinite(gripper_deadband) or gripper_deadband < 0:
        raise ValueError('gripper_deadband must be finite and nonnegative')
    return {
        'schema_version': SCHEMA_VERSION,
        'space': '7D commands passed to env.step, after policy output transforms; before environment internal clipping/scaling',
        'units': 'controller command units; translation and rotation reported separately, not physical displacement/jerk',
        'replan_steps': replan_steps,
        'boundary': 't > 0 and t % replan_steps == 0; delta = command[t] - command[t-1]',
        'internal': 't > 0 and t % replan_steps != 0',
        'settling_actions_included': False,
        'first_action_transition_included': False,
        'gripper_deadband': gripper_deadband,
        'gripper_flip': 'adjacent commands on opposite sides of +/- deadband; neutral transitions do not count as flips',
        'percentile': 'linear interpolation at (n-1)*0.95 within each episode/group',
        'recording': 'gzip JSON command_actions and replanned; no images',
        'input_bounds': {'low': [-1.] * 7, 'high': [1.] * 7,
                         'validation': 'When enabled, actual robosuite env.env.action_spec must match these seven bounds.'},
        'effective_control_proxy': {
            'transform': 'clip(command[:6], -1, 1), followed by exact sign(command[6]); sign(0)=0',
            'assumption': 'ROSE LIBERO OSC arm input_min/max +/-1 and PandaGripper sign mapping; downstream scaling/dynamics not modeled',
            'units': 'normalized controller-input proxy; not physical executed action, displacement or jerk',
            'gripper_deadband': 0.,
            'changes_policy_or_environment': False,
        },
    }


def distribution(values):
    values = [float(x) for x in values]
    if not all(math.isfinite(x) for x in values):
        raise ValueError('Non-finite action statistic')
    count = len(values)
    if not count:
        return {'count': 0, 'mean': None, 'rms': None, 'p95': None, 'max': None}
    ordered = sorted(values)
    position = (count - 1) * .95
    left = int(position)
    right = min(left + 1, count - 1)
    p95 = ordered[left] + (position - left) * (ordered[right] - ordered[left])
    return {'count': count, 'mean': math.fsum(values) / count,
            'rms': math.sqrt(math.fsum(x*x for x in values) / count),
            'p95': p95, 'max': max(values)}


def summarize(actions, replanned, replan_steps, gripper_deadband=GRIPPER_DEADBAND):
    metadata = contract(replan_steps, gripper_deadband)
    if len(actions) != len(replanned) or not actions:
        raise ValueError('A nonempty command sequence and one replan flag per command are required')
    checked = []
    for step, (action, flag) in enumerate(zip(actions, replanned)):
        if type(flag) is not bool or flag != (step % replan_steps == 0):
            raise ValueError(f'Replan flag disagrees with configured cadence at step {step}')
        if len(action) != 7:
            raise ValueError('Expected seven command dimensions')
        command = [float(x) for x in action]
        if not all(math.isfinite(x) for x in command):
            raise ValueError('Non-finite command')
        checked.append(command)
    output = {'schema_version': SCHEMA_VERSION, 'contract': metadata, 'commands': len(checked)}
    output.update(_transitions(checked, replanned, gripper_deadband))
    outside = [sum(abs(command[dimension]) > 1. for command in checked) for dimension in range(7)]
    output['input_range'] = {
        'commands': len(checked), 'low': [-1.] * 7, 'high': [1.] * 7,
        'outside_count_per_dim': outside,
        'outside_fraction_per_dim': [count/len(checked) for count in outside],
        'arm_any_outside_count': sum(any(abs(x)>1. for x in command[:6]) for command in checked),
        'arm_any_outside_fraction': sum(any(abs(x)>1. for x in command[:6]) for command in checked)/len(checked),
        'any_outside_count': sum(any(abs(x)>1. for x in command) for command in checked),
        'any_outside_fraction': sum(any(abs(x)>1. for x in command) for command in checked)/len(checked),
    }
    proxy = [[max(-1., min(1., x)) for x in command[:6]] +
             [float((command[6] > 0.) - (command[6] < 0.))] for command in checked]
    output['effective_control_proxy'] = {'commands': len(checked), 'contract': metadata['effective_control_proxy'],
                                         **_transitions(proxy, replanned, 0.)}
    return output


def validate_action_spec(action_spec):
    """Fail closed if the audited LIBERO/Panda proxy does not match action bounds."""
    if len(action_spec) != 2:
        raise ValueError('Expected action_spec low/high pair')
    low, high = ([float(x) for x in values] for values in action_spec)
    if len(low) != 7 or len(high) != 7 or any(x != -1. for x in low) or any(x != 1. for x in high):
        raise ValueError('Effective-control proxy requires seven action_spec bounds of exactly [-1,1]')
    return {'low': low, 'high': high, 'bounds_verified': True, 'source': 'env.env.action_spec'}


def _transitions(checked, replanned, gripper_deadband):
    groups = {name: {'translation': [], 'rotation': [], 'gripper': [], 'flips': 0,
                     'eligible': 0, 'neutral': 0} for name in ('boundary', 'internal', 'all')}
    def direction(value):
        return 1 if value > gripper_deadband else -1 if value < -gripper_deadband else 0
    for step in range(1, len(checked)):
        previous, current = checked[step-1], checked[step]
        delta = [b-a for a, b in zip(previous, current)]
        translation = math.sqrt(math.fsum(x*x for x in delta[:3]))
        rotation = math.sqrt(math.fsum(x*x for x in delta[3:6]))
        a, b = direction(previous[6]), direction(current[6])
        for key in ('boundary' if replanned[step] else 'internal', 'all'):
            group = groups[key]
            group['translation'].append(translation)
            group['rotation'].append(rotation)
            group['gripper'].append(abs(delta[6]))
            group['flips'] += int(a*b == -1)
            group['eligible'] += int(a != 0 and b != 0)
            group['neutral'] += int(a == 0 or b == 0)
    output = {}
    for key, group in groups.items():
        count = len(group['translation'])
        output[key] = {
            'transitions': count,
            'translation_l2_delta': distribution(group['translation']),
            'rotation_l2_delta': distribution(group['rotation']),
            'gripper_abs_delta': distribution(group['gripper']),
            'gripper_flips': group['flips'], 'gripper_non_neutral_pairs': group['eligible'],
            'gripper_neutral_pairs': group['neutral'],
            'gripper_flip_rate': group['flips']/count if count else None,
            'gripper_flip_rate_non_neutral': group['flips']/group['eligible'] if group['eligible'] else None,
        }
    return output


def save_trace(directory, *, task_id, init_index, seed, actions, replanned, replan_steps):
    """Publish an atomic trace; preserve any orphan from an interrupted attempt."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    name = f't{task_id}_i{init_index}_s{seed}.json.gz'
    path = directory / name
    if path.exists():
        path = directory / f't{task_id}_i{init_index}_s{seed}_{uuid.uuid4().hex[:12]}.json.gz'
    metrics = summarize(actions, replanned, replan_steps)
    payload = {'schema_version': SCHEMA_VERSION, 'task_id': task_id, 'init_index': init_index,
               'seed': seed, 'contract': metrics['contract'], 'command_actions': actions,
               'replanned': replanned}
    encoded = (json.dumps(payload, separators=(',', ':'), allow_nan=False) + '\n').encode()
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        with gzip.GzipFile(fileobj=stream, mode='wb', mtime=0, filename='') as compressed:
            compressed.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return metrics, {'path': str(path.relative_to(directory.parent)),
                     'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                     'bytes': path.stat().st_size, 'schema_version': SCHEMA_VERSION}


def verify_trace(root, row, replan_steps):
    """Verify a logged trace and recompute its metrics without a model or simulator."""
    root = Path(root).resolve()
    reference = row['action_trace']
    relative = Path(reference['path'])
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root):
        raise ValueError('Action trace reference escapes the result directory')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != reference['sha256'] or len(raw) != reference['bytes']:
        raise ValueError('Action trace hash/size does not match the episode receipt')
    payload = json.loads(gzip.decompress(raw))
    if payload['schema_version'] != SCHEMA_VERSION:
        raise ValueError('Unsupported action trace schema')
    for field in ('task_id', 'init_index', 'seed'):
        if payload[field] != row[field]:
            raise ValueError(f'Action trace {field} differs from episode')
    measured = summarize(payload['command_actions'], payload['replanned'], replan_steps)
    if measured != row['action_metrics'] or measured['contract'] != payload['contract']:
        raise ValueError('Action metrics differ from recomputed command trace')
    if measured['commands'] != row['actions_executed']:
        raise ValueError('Action trace length differs from actual env.step calls')
    return measured


def episode_weighted(rows, space='raw_command'):
    """Each episode with a defined statistic gets equal weight, not each timestep."""
    if space not in ('raw_command', 'effective_control_proxy'):
        raise ValueError('Unknown action metric space')
    series = [row['action_metrics'] if space == 'raw_command' else row['action_metrics']['effective_control_proxy'] for row in rows]
    output = {'episodes': len(rows), 'space': space, 'weighting': 'equal weight per episode with a defined statistic',
              'caution': 'Success ends episodes early; groups may differ in duration and state distribution.'}
    for group in ('boundary', 'internal', 'all'):
        result = {'episodes_with_transitions': sum(item[group]['transitions'] > 0 for item in series),
                  'total_transitions': sum(item[group]['transitions'] for item in series)}
        for component in ('translation_l2_delta', 'rotation_l2_delta', 'gripper_abs_delta'):
            result[component] = {}
            for statistic in ('mean', 'rms', 'p95', 'max'):
                values = [item[group][component][statistic] for item in series]
                values = [value for value in values if value is not None]
                result[component][f'episode_mean_{statistic}'] = math.fsum(values)/len(values) if values else None
        for statistic in ('gripper_flip_rate', 'gripper_flip_rate_non_neutral'):
            values = [item[group][statistic] for item in series]
            values = [value for value in values if value is not None]
            result[f'episode_mean_{statistic}'] = math.fsum(values)/len(values) if values else None
        result['gripper_flips_total'] = sum(item[group]['gripper_flips'] for item in series)
        output[group] = result
    if space == 'raw_command':
        output['input_range'] = {
            'outside_count_per_dim': [sum(row['action_metrics']['input_range']['outside_count_per_dim'][i] for row in rows) for i in range(7)],
            'episode_mean_outside_fraction_per_dim': [math.fsum(row['action_metrics']['input_range']['outside_fraction_per_dim'][i] for row in rows)/len(rows) if rows else None for i in range(7)],
            'episode_mean_arm_any_outside_fraction': math.fsum(row['action_metrics']['input_range']['arm_any_outside_fraction'] for row in rows)/len(rows) if rows else None,
            'episode_mean_any_outside_fraction': math.fsum(row['action_metrics']['input_range']['any_outside_fraction'] for row in rows)/len(rows) if rows else None,
        }
        output['effective_control_proxy'] = episode_weighted(rows, space='effective_control_proxy')
    return output


def wandb_values(metrics):
    result = {'eval/action_commands': metrics['commands'],
              'eval/actions_arm_clip_fraction': metrics['input_range']['arm_any_outside_fraction'],
              'eval/actions_any_outside_fraction': metrics['input_range']['any_outside_fraction']}
    for i, value in enumerate(metrics['input_range']['outside_fraction_per_dim']):
        result[f'eval/actions_outside_fraction_dim{i}'] = value
    for group in ('boundary', 'internal', 'all'):
        proxy = metrics['effective_control_proxy'][group]
        for component in ('translation_l2_delta', 'rotation_l2_delta', 'gripper_abs_delta'):
            if proxy[component]['mean'] is not None:
                result[f'eval/actions_proxy_{group}_{component}_mean'] = proxy[component]['mean']
        if proxy['gripper_flip_rate'] is not None:
            result[f'eval/actions_proxy_{group}_gripper_flip_rate'] = proxy['gripper_flip_rate']
    for group in ('boundary', 'internal', 'all'):
        values = metrics[group]
        prefix = f'eval/actions_{group}_'
        result[prefix+'transitions'] = values['transitions']
        result[prefix+'gripper_flips'] = values['gripper_flips']
        if values['gripper_flip_rate'] is not None:
            result[prefix+'gripper_flip_rate'] = values['gripper_flip_rate']
        for component in ('translation_l2_delta', 'rotation_l2_delta', 'gripper_abs_delta'):
            for statistic in ('mean', 'rms', 'p95', 'max'):
                value = values[component][statistic]
                if value is not None:
                    result[prefix+component+'_'+statistic] = value
    return result
