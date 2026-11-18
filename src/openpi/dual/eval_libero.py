"""Native/Plus LIBERO evaluation and non-privileged on-policy data recording."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import inspect
import importlib.metadata
import json
import os
from pathlib import Path
import random
import socket
import sys

import numpy as np
import torch

from openpi.dual.action_metrics import contract as action_contract, save_trace, validate_action_spec, verify_trace
from openpi.dual.bridge import file_sha256
from openpi.dual.config import load_config
from openpi.dual.policy import create_policy
from openpi.dual.resume import resume_manifest, record_attempt
from openpi.dual.storage import wait_for_space, metrics as storage_metrics


def observation_element(obs, prompt):
    from openpi_client import image_tools
    quat = np.asarray(obs['robot0_eef_quat'])
    w = np.clip(quat[3], -1., 1.)
    denominator = np.sqrt(max(0., 1. - w * w))
    rot = np.zeros(3) if denominator < 1e-8 else quat[:3] * 2 * np.arccos(w) / denominator
    result = {'prompt': prompt, 'observation/state': np.concatenate(
        [obs['robot0_eef_pos'], rot, obs['robot0_gripper_qpos']]).astype(np.float32)}
    for key, camera in [('observation/image', 'agentview_image'), ('observation/wrist_image', 'robot0_eye_in_hand_image')]:
        image = np.ascontiguousarray(obs[camera][::-1, ::-1])
        result[key] = image_tools.convert_to_uint8(image_tools.resize_with_pad(image, 224, 224))
    return result


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def fresh_scene_seed(task_id, init_index):
    """A scene seed independent of policy noise, round, worker and GPU."""
    if not 0 <= task_id < 10 or init_index < 1000:
        raise ValueError('Fresh native scenes require task 0..9 and init_index >= 1000')
    seed = 1000 * init_index + task_id
    if seed >= 2**32:
        raise ValueError('Fresh scene seed exceeds NumPy uint32 range')
    return seed


def scene_identity(benchmark, task_id, init_index, fresh):
    return (f'{benchmark}:fresh_reset:{task_id}:{init_index}' if fresh
            else f'{benchmark}:{task_id}:{init_index}')


def read_episode_rows(path, recording_recovery=False):
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    rows = []
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if not recording_recovery or index + 1 != len(lines):
                raise
            # The NPZ already contains the complete episode receipt. Preserve
            # the interrupted tail, then recover that row from its NPZ below.
            write_json(path.with_suffix('.interrupted-tail.json'), {'line': line, 'valid_rows': len(rows)})
            temporary = path.with_suffix('.jsonl.tmp')
            with temporary.open('w') as stream:
                for row in rows:
                    stream.write(json.dumps(row, allow_nan=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
    return rows


def recover_recording(path, manifest):
    """Validate a committed recording, including a crash before the JSONL append.

    An episode is committed by the atomic NPZ rename. The small receipt can be
    rebuilt from that file. A temporary NPZ is never a training episode.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        if json.loads(str(archive['provenance'].item())) != manifest:
            raise ValueError(f'Recorded episode provenance differs: {path}')
        metadata = json.loads(str(archive['meta_json'].item()))
        row = json.loads(str(archive['episode_json'].item()))
        n = len(archive['state'])
        if (metadata.get('schema_version') != 2 or n != row['actions_executed']
                or archive['state'].shape != (n, 8) or archive['action'].shape != (n, 7)
                or n < 1 or row.get('error') is not None):
            raise ValueError(f'Incomplete recorded trajectory: {path}')
        state = np.asarray(archive['initial_state'])
        if (not np.isfinite(state).all() or not np.isfinite(archive['state']).all()
                or not np.isfinite(archive['action']).all()):
            raise ValueError(f'Nonfinite recorded trajectory: {path}')
        if os.environ.get('PI_FS_SKIP_HASH_VERIFICATION') != '1' and hashlib.sha256(state.tobytes()).hexdigest() != metadata['initial_state_sha256']:
            raise ValueError(f'Initial state identity changed: {path}')
        if metadata['initial_state_sha256'] != row['initial_state_sha256']:
            raise ValueError(f'Recording/episode initial state identities differ: {path}')
        for key in ('task_id', 'init_index', 'seed', 'success'):
            if metadata[key] != row[key]:
                raise ValueError(f'Recording/episode {key} differs: {path}')
        if metadata['data_role'] != manifest['record_role']:
            raise ValueError(f'Recording role differs: {path}')
        if path.name != f"t{row['task_id']}_i{row['init_index']}_s{row['seed']}.npz":
            raise ValueError(f'Recording identity differs from its file name: {path}')
    receipt = {'schema_version': 2, 'path': str(path.resolve()), 'sha256': file_sha256(path),
               'bytes': path.stat().st_size, 'frames': n}
    receipt_path = path.with_suffix('.receipt.json')
    if receipt_path.exists() and json.loads(receipt_path.read_text()) != receipt:
        raise ValueError(f'Recording content differs from its receipt: {path}')
    if not receipt_path.exists():
        write_json(receipt_path, receipt)
    row['recording'] = receipt
    return row


def save_recording(path, recorded, initial_state, metadata, row, manifest):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f'Refuse to overwrite committed recording {path}')
    # Include the atomic temporary file and the actual Ceph project quota.
    raw_bytes = sum(sum(value.nbytes for key, value in element.items() if key != 'prompt') + action.nbytes
                    for element, action in recorded)
    wait_for_space(path.parent, raw_bytes + 16 * 1024**2)
    temporary = path.with_suffix('.tmp.npz')
    try:
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, img=np.stack([x[0]['observation/image'] for x in recorded]),
                                wrist=np.stack([x[0]['observation/wrist_image'] for x in recorded]),
                                state=np.stack([x[0]['observation/state'] for x in recorded]),
                                action=np.stack([x[1] for x in recorded]), prompt=recorded[0][0]['prompt'],
                                initial_state=initial_state, success=row['success'], task_id=row['task_id'],
                                init_index=row['init_index'], seed=row['seed'],
                                meta_json=json.dumps(metadata, sort_keys=True),
                                episode_json=json.dumps(row, sort_keys=True, allow_nan=False),
                                temporal_unit='environment_step', provenance=json.dumps(manifest, sort_keys=True))
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return recover_recording(path, manifest)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--policy-backend', choices=['compact', 'stock'], default='compact')
    parser.add_argument('--shadow-updater-every', type=int, default=0,
                        help='Virtual dual diagnostic only: compare read-only current teacher every N replans; early ages always checked')
    parser.add_argument('--benchmark', choices=['native', 'plus'], default='native')
    parser.add_argument('--tasks', default='all')
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--init-offset', type=int, default=0)
    parser.add_argument('--fresh-inits', action='store_true',
                        help='Native fresh reset scenes: env.seed(1000*init_index+task_id), init-offset >= 1000')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--max-steps', type=int, default=520)
    parser.add_argument('--out', required=True)
    parser.add_argument('--record-dir')
    parser.add_argument('--resume-source', help='Explicit reviewed previous source path; protocol must remain identical')
    parser.add_argument('--wandb-project')
    parser.add_argument('--wandb-entity', default=None)
    parser.add_argument('--wandb-name')
    parser.add_argument('--action-metrics', action='store_true',
                        help='Record lightweight per-step command traces and boundary/internal continuity metrics')
    parser.add_argument('--record-role', choices=['train', 'validation', 'evaluation'], default='evaluation')
    parser.add_argument('--record-dataset-scope', default='unspecified',
                        help='Descriptive immutable collection lineage; never changes scene identity or data role')
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    if args.trials < 1 or args.init_offset < 0 or args.max_steps < 1:
        raise ValueError('Positive trials/max-steps and nonnegative init-offset required')
    if args.fresh_inits and (args.benchmark != 'native' or args.init_offset < 1000):
        raise ValueError('fresh-inits requires native LIBERO and init-offset >= 1000')
    config = load_config(args.config)
    if args.shadow_updater_every < 0:
        raise ValueError('shadow-updater-every must be nonnegative')
    if args.shadow_updater_every and not (args.policy_backend == 'compact' and config.dual.enabled
            and config.dual.updater_enabled and config.dual.mode == 'virtual'
            and not config.cache.enabled and config.num_steps == 1):
        raise ValueError('Shadow diagnosis requires compact virtual updater, fresh-noise NFE1, cache off')
    if args.policy_backend == 'stock' and config.dual.enabled:
        raise ValueError('Stock backend forbids dual; compact is required for the updater')
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.envs.bddl_utils import get_problem_info
    suite = benchmark.get_benchmark_dict()['libero_10']()
    expected = 2519 if args.benchmark == 'plus' else 10
    if suite.n_tasks != expected:
        raise ValueError(f'Wrong LIBERO loader: requested {args.benchmark}, got {suite.n_tasks} tasks from {benchmark.__file__}')
    ids = list(range(expected)) if args.tasks == 'all' else [int(item) for item in args.tasks.split(',')]
    if not ids or len(set(ids)) != len(ids) or any(i < 0 or i >= expected for i in ids):
        raise ValueError('Invalid task IDs')
    classification = {}
    if args.benchmark == 'plus':
        path = Path(benchmark.__file__).parent / 'task_classification.json'
        classification = {row['name']: row['category'] for row in json.loads(path.read_text())['libero_10']}
    tasks, init_states = [], {}
    safe = [np.core.multiarray._reconstruct, np.ndarray, np.dtype,
            type(np.dtype(np.float64)), type(np.dtype(np.float32))]
    for index in ids:
        task = suite.get_task(index)
        filename = task.bddl_file.split('_view_', 1)[0] + '.bddl' if '_view_' in task.bddl_file else task.bddl_file
        bddl = Path(get_libero_path('bddl_files')) / task.problem_folder / filename
        prompt = get_problem_info(str(bddl))['language_instruction']
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f'Missing actual BDDL instruction: {bddl}')
        if args.fresh_inits:
            selected = []
            seeds = [fresh_scene_seed(index, i) for i in range(args.init_offset, args.init_offset + args.trials)]
        else:
            with torch.serialization.safe_globals(safe):
                states = suite.get_task_init_states(index)
            if args.init_offset + args.trials > len(states):
                raise ValueError(f'Not enough initial states for task {index}')
            selected = [np.array(states[i], copy=True) for i in range(args.init_offset, args.init_offset + args.trials)]
        init_states[index] = selected
        tasks.append({'task_id': index, 'name': task.name, 'prompt': prompt,
                      'category': classification.get(task.name, 'Native'), 'bddl_sha256': file_sha256(bddl),
                      'initial_state_sha256': [hashlib.sha256(x.tobytes()).hexdigest() for x in selected]})
        if args.fresh_inits:
            tasks[-1].update(fresh_scene_seeds=seeds, initial_state_hash_location='per-episode row and recorded NPZ')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).parent
    manifest = {'schema_version': 1, 'config': config.to_dict(), 'benchmark': args.benchmark,
                'shadow_every_replans': args.shadow_updater_every,
                'shadow_scope': 'read-only current teacher; never published or executed; extra compute excluded from speed claims',
                'runtime': {'python': sys.version.split()[0], 'executable': sys.executable,
                            'environment': os.environ.get('ROSE_OPENPI_ENV', 'active Python environment'),
                            'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
                            'packages': {name: importlib.metadata.version(name) for name in
                                         ('transformers', 'robosuite', 'mujoco', 'wandb')}},
                'policy_backend': ('stock_compiled' if args.policy_backend == 'compact' and config.inference.backend == 'stock' else args.policy_backend),
                'gpu_sharing': os.environ.get('PI_FS_GPU_SHARING', 'not_independently_verified'),
                'seed': args.seed, 'trials': args.trials, 'init_offset': args.init_offset, 'tasks': tasks,
                'max_steps': args.max_steps, 'settling_steps': 10, 'resolution': 256,
                'base_sha256': file_sha256(Path(config.base_checkpoint) / 'model.safetensors'),
                'updater_sha256': file_sha256(config.dual.updater_checkpoint) if config.dual.enabled and config.dual.updater_enabled and config.dual.updater_checkpoint else None,
                'source_sha256': {p.name: file_sha256(p) for p in sorted(source.glob('*.py'))},
                'benchmark_module': benchmark.__file__, 'benchmark_sha256': file_sha256(benchmark.__file__),
                'environment_sha256': file_sha256(inspect.getfile(OffScreenRenderEnv)),
                'noise': 'PCG64 SeedSequence([seed,plus:1/native:0,task_id,init_index]); fresh innovation per replan',
                'timing': ('full-device synchronized stock policy.infer; includes prefix and output transforms; no speedup claim'
                           if args.policy_backend == 'stock' else
                           'synchronized current-stream component and wall times; virtual parameters are simulated; no speedup claim'),
                'state_input': False, 'compile': config.inference.compile_mode,
                'identity_mode': ('path_config_no_hash_verification' if os.environ.get('PI_FS_SKIP_HASH_VERIFICATION') == '1' else 'sha256'),
                'source_path': str(source.parent.parent.parent), 'job_id': os.environ.get('SLURM_JOB_ID'), 'record_dir': args.record_dir, 'record_role': args.record_role,
                'action_metrics': action_contract(config.replan_steps) if args.action_metrics else None}
    if args.fresh_inits or args.record_dir:
        manifest.update(initial_state_kind='fresh_reset' if args.fresh_inits else 'fixed_initial_state',
                        fresh_scene_seed_formula='1000*init_index+task_id' if args.fresh_inits else None,
                        recording_schema_version=2 if args.record_dir else None,
                        recording_compression='numpy savez_compressed lossless' if args.record_dir else None,
                        record_dataset_scope=args.record_dataset_scope)
    manifest_path = out / 'manifest.json'
    attempt_manifest = manifest
    manifest = resume_manifest(manifest_path, manifest, args.resume_source)
    write_json(manifest_path, manifest)
    print(json.dumps({'preflight': 'passed', 'tasks': len(tasks), 'benchmark': args.benchmark}), flush=True)
    if args.preflight_only:
        return
    if not torch.cuda.is_available():
        raise RuntimeError('LIBERO evaluation requires an allocated server GPU')
    execution_attempt = record_attempt(out, attempt_manifest, args.resume_source)
    records_file = out / 'episodes.jsonl'
    rows = read_episode_rows(records_file, recording_recovery=bool(args.record_dir))
    keys = {(row['task_id'], row['init_index']) for row in rows}
    expected_keys = {(task['task_id'], i) for task in tasks for i in range(args.init_offset, args.init_offset + args.trials)}
    if len(keys) != len(rows) or not keys <= expected_keys or any(row.get('error') and not (args.benchmark == 'plus' and row.get('outcome') == 'unsupported_instruction') for row in rows):
        raise ValueError('Invalid or duplicate resume episode records')
    if args.record_dir:
        by_key = {(row['task_id'], row['init_index']): row for row in rows}
        for index, initial_index in sorted(expected_keys):
            path = Path(args.record_dir) / f't{index}_i{initial_index}_s{args.seed}.npz'
            existing = by_key.get((index, initial_index))
            if existing is not None and not path.exists():
                raise ValueError(f'Completed episode has no recording: {path}')
            if not path.exists():
                continue
            recovered = recover_recording(path, manifest)
            if existing is not None:
                if recovered != existing:
                    raise ValueError(f'JSONL/recording episode receipt differs: {path}')
            else:
                if args.action_metrics:
                    verify_trace(out, recovered, config.replan_steps)
                with records_file.open('a') as stream:
                    stream.write(json.dumps(recovered, allow_nan=False) + '\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                rows.append(recovered)
                keys.add((index, initial_index))
    if args.action_metrics:
        for row in rows:
            if row.get('outcome') != 'unsupported_instruction':
                verify_trace(out, row, config.replan_steps)
    wandb_run = None
    if args.wandb_project:
        import wandb
        saved_run = out / 'wandb.json'
        resume_kwargs = ({'id': json.loads(saved_run.read_text())['id'], 'resume': 'must'}
                         if saved_run.exists() else {})
        wandb_run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_name,
                               config=manifest, dir=str(out), mode='online', **resume_kwargs)
        wandb_run.summary.update({'latest_job_id': execution_attempt['job_id'],
                                  'latest_source_path': execution_attempt['source_path']})
        write_json(out / 'wandb.json', {'id': wandb_run.id, 'url': wandb_run.url})
        wandb_run.log({'rollout/episodes': len(rows), 'rollout/status': 'loading_policy'})
    if args.policy_backend == 'stock':
        from openpi.dual.stock_policy import StockPolicy
        policy = StockPolicy(config)
    else:
        policy = create_policy(config, shadow_every_replans=args.shadow_updater_every)
    try:
        for task_record in tasks:
            index = task_record['task_id']
            task = suite.get_task(index)
            for trial in range(args.trials):
                initial_index = args.init_offset + trial
                if (index, initial_index) in keys:
                    continue
                wait_for_space(out, 512 * 1024**2, event_path=out / 'storage_events.jsonl',
                               on_event=wandb_run.log if wandb_run else None)
                random.seed(args.seed)
                np.random.seed(args.seed)
                rng = np.random.default_rng(np.random.SeedSequence([args.seed, int(args.benchmark == 'plus'), index, initial_index]))
                row = dict(task_id=index, init_index=initial_index, seed=args.seed, category=task_record['category'],
                           success=False, error=None, calls=[], hostname=socket.gethostname(),
                           gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
                           created_at=datetime.now(timezone.utc).isoformat(), execution_attempt=execution_attempt)
                if args.fresh_inits or args.record_dir:
                    row['gpu_binding'] = {key: os.environ.get(key) for key in
                                          ('CUDA_VISIBLE_DEVICES', 'MUJOCO_EGL_DEVICE_ID', 'MUJOCO_GL', 'PYOPENGL_PLATFORM')}
                    row['gpu_binding']['torch_device'] = 'cuda:0'
                env = None
                recorded = []
                command_actions, command_replans = [], []
                try:
                    env_path = Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
                    env = OffScreenRenderEnv(bddl_file_name=str(env_path), camera_heights=256, camera_widths=256)
                    if ' '.join(env.language_instruction.split()) != ' '.join(task_record['prompt'].split()):
                        raise ValueError('Actual environment instruction differs from BDDL prompt')
                    if args.action_metrics:
                        row['action_spec'] = validate_action_spec(env.env.action_spec)
                    if args.fresh_inits:
                        initial_seed = fresh_scene_seed(index, initial_index)
                        env.seed(initial_seed)
                        obs = env.reset()
                        initial_state = np.array(env.env.sim.get_state().flatten(), copy=True)
                    else:
                        initial_seed = None
                        env.seed(args.seed)
                        env.reset()
                        initial_state = init_states[index][trial]
                        obs = env.set_init_state(initial_state)
                    if args.fresh_inits or args.record_dir:
                        row.update(initial_state_kind='fresh_reset' if args.fresh_inits else 'fixed_initial_state',
                                   initial_state_seed=initial_seed,
                                   initial_state_sha256=hashlib.sha256(initial_state.tobytes()).hexdigest(),
                                   scene_key=scene_identity(args.benchmark, index, initial_index, args.fresh_inits))
                    for _ in range(10):
                        obs, _, _, _ = env.step([0.] * 6 + [-1.])
                    policy.reset()
                    previous_action = None
                    for step in range(args.max_steps):
                        element = observation_element(obs, task_record['prompt'])
                        noise = rng.standard_normal((10, 32), dtype=np.float32) if step % config.replan_steps == 0 else None
                        prediction = policy.act(element, executed_action=previous_action, noise=noise)
                        action = prediction['action']
                        if prediction['replanned']:
                            row['calls'].append({'step': step, **prediction['dual'], **prediction['policy_timing'],
                                                 **({'shadow': prediction['shadow']} if prediction.get('shadow') is not None else {}),
                                                 'innovation_sha256': hashlib.sha256(noise.tobytes()).hexdigest()})
                        command = action.tolist()
                        obs, _, done, _ = env.step(command)
                        if args.record_dir:
                            # Retain only actions actually submitted successfully;
                            # task failures/timeouts are data, environment errors are not.
                            recorded.append((element, action.copy()))
                        if args.action_metrics:
                            command_actions.append(action.tolist())
                            command_replans.append(bool(prediction['replanned']))
                        previous_action = action.copy()
                        row['actions_executed'] = step + 1
                        if done:
                            row['success'] = True
                            break
                    if args.action_metrics:
                        row['action_metrics'], row['action_trace'] = save_trace(
                            out / 'actions', task_id=index, init_index=initial_index, seed=args.seed,
                            actions=command_actions, replanned=command_replans, replan_steps=config.replan_steps)
                    if args.record_dir:
                        metadata = {key: row[key] for key in ('task_id', 'init_index', 'seed', 'success',
                                    'initial_state_kind', 'initial_state_seed', 'initial_state_sha256', 'scene_key')}
                        metadata.update(schema_version=2, data_role=args.record_role, benchmark=args.benchmark,
                                        dataset_scope=args.record_dataset_scope, temporal_unit='environment_step',
                                        removes_noops=False, action_semantics='command submitted to env.step',
                                        base_sha256=manifest['base_sha256'], updater_sha256=manifest['updater_sha256'])
                        path = Path(args.record_dir) / f't{index}_i{initial_index}_s{args.seed}.npz'
                        row = save_recording(path, recorded, initial_state, metadata, row, manifest)
                    with records_file.open('a') as stream:
                        stream.write(json.dumps(row, allow_nan=False) + '\n')
                        stream.flush()
                        os.fsync(stream.fileno())
                    rows.append(row)
                    if wandb_run:
                        wandb_run.log({'rollout/episodes':len(rows), 'rollout/task_id': index,
                                       'rollout/success':int(row['success']),
                                       'rollout/success_rate':sum(r['success'] for r in rows)/len(rows),
                                       'rollout/actions_executed':row['actions_executed'], **storage_metrics(out)})
                    print(json.dumps({'done': len(rows), 'expected': len(expected_keys), 'success': row['success']}), flush=True)
                except Exception as error:
                    row['error'] = repr(error)
                    write_json(out / f'error_t{index}_i{initial_index}.json', row)
                    supported_error = isinstance(error, ValueError) and str(error) in (
                        'Active text exceeds trained updater capacity',
                        'Active prefix content lies outside trained updater slots')
                    if args.benchmark != 'plus' or not supported_error or row['calls'] or row.get('actions_executed', 0):
                        raise
                    row.update(outcome='unsupported_instruction', actions_executed=0)
                    with records_file.open('a') as stream:
                        stream.write(json.dumps(row, allow_nan=False) + '\n')
                        stream.flush(); os.fsync(stream.fileno())
                    rows.append(row)
                    if wandb_run:
                        wandb_run.log({'rollout/episodes':len(rows),'rollout/task_id':index,
                                       'rollout/unsupported_instruction':1,'rollout/success':0})
                finally:
                    if env is not None:
                        env.close()
    finally:
        policy.close()
    categories = defaultdict(list)
    for row in rows:
        categories[row['category']].append(row['success'])
    if len(rows) != len(expected_keys):
        raise ValueError('Incomplete episode set')
    write_json(out / 'summary.json', {'episodes': len(rows), 'successes': sum(row['success'] for row in rows),
               'success_rate': sum(row['success'] for row in rows) / len(rows),
               'by_perturbation': {key: {'episodes': len(values), 'successes': sum(values), 'success_rate': sum(values)/len(values)}
                                  for key, values in categories.items()}, 'diagnostic_short_run': args.max_steps != 520})

    if wandb_run:
        wandb_run.summary.update(json.loads((out / 'summary.json').read_text()))
        wandb_run.finish()

if __name__ == '__main__':
    main()
