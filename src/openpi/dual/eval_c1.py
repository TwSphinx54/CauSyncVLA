"""Native LIBERO-10, C=1, compiled PI0.5 with a continuous command clock."""
import argparse
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import random
import socket

import numpy as np
import torch
import wandb

from openpi.dual.c1_clock import run_episode
from openpi.dual.c1_engine import CompiledC1Engine
from openpi.dual.config import load_config
from openpi.dual.eval_libero import observation_element, write_json
from openpi.dual.storage import wait_for_space


def summarize(rows):
    keys = ('eligible_ticks', 'fresh_on_time_ticks', 'late_adopt_ticks', 'new_action_ticks',
            'hold_ticks', 'busy_skip_ticks', 'request_count', 'warm_requests_ready_50ms',
            'warm_requests_ready_by_target', 'clock_overrun_ticks')
    totals = {key: sum(row[key] for row in rows) for key in keys}
    def ratio(numerator, denominator):
        return totals[numerator] / totals[denominator] if totals[denominator] else None
    successes = sum(bool(row['success']) for row in rows)
    return dict(episodes=len(rows), successes=successes, success_rate=successes/len(rows),
                fresh_on_time_fraction=ratio('fresh_on_time_ticks','eligible_ticks'),
                new_action_fraction=ratio('new_action_ticks','eligible_ticks'),
                observation_to_ready_50ms_fraction=ratio('warm_requests_ready_50ms','request_count'),
                hold_fraction=ratio('hold_ticks','eligible_ticks'),
                **totals)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tasks', default='all')
    parser.add_argument('--trials', type=int, default=50)
    parser.add_argument('--init-offset', type=int, default=0)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--command-hz', type=int, choices=(20, 50), default=20)
    parser.add_argument('--max-steps', type=int, help='defaults to 26 simulated seconds at command-hz')
    parser.add_argument('--wandb-project', default='causyncvla')
    parser.add_argument('--wandb-name', default='native-c1')
    parser.add_argument('--wandb-entity', default=None)
    args = parser.parse_args()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.envs.bddl_utils import get_problem_info
    ids = list(range(10)) if args.tasks=='all' else [int(t) for t in args.tasks.split(',')]
    if not ids or len(ids)!=len(set(ids)) or any(t not in range(10) for t in ids):
        raise ValueError('Expected distinct native LIBERO-10 tasks')
    hz = args.command_hz
    period_ns = round(1e9 / hz)
    max_steps = args.max_steps if args.max_steps is not None else 26 * hz
    settling_steps = hz // 2
    if args.init_offset<0 or args.trials<1 or args.init_offset+args.trials>50 or max_steps!=26*hz:
        raise ValueError('C1 evaluation uses official initial states 0..49 and a 26-second simulation budget')
    isolated=os.environ.get('CHRONO_C1_RENDER_ISOLATED')=='1'
    if hz == 50 and not isolated:
        raise ValueError('50 Hz experiment requires dedicated EGL and inference GPUs')
    if isolated:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    config=load_config(args.config)
    suite=benchmark.get_benchmark_dict()['libero_10']()
    if suite.n_tasks!=10: raise ValueError('Expected native 10-task suite')
    tasks={}; states={}
    safe=[np.core.multiarray._reconstruct,np.ndarray,np.dtype,type(np.dtype(np.float64)),type(np.dtype(np.float32))]
    for t in ids:
        task=suite.get_task(t)
        bddl=Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
        prompt=get_problem_info(str(bddl))['language_instruction']
        if not prompt.strip(): raise ValueError('Empty BDDL instruction')
        tasks[str(t)]=dict(name=task.name, bddl=str(bddl), prompt=prompt)
        with torch.serialization.safe_globals(safe):
            init_states=suite.get_task_init_states(t)
        for i in range(args.init_offset,args.init_offset+args.trials):
            states[t,i]=np.array(init_states[i],copy=True)
    out=args.out
    (out/'traces').mkdir(parents=True,exist_ok=True)
    gpu=torch.cuda.get_device_properties(0)
    manifest=dict(protocol=('native-c1-50hz-v1' if hz==50 else 'native-c1-deadline-v2-separated-render' if isolated else 'native-c1-deadline-v1'),source=str(Path(__file__).resolve().parents[3]),miss_policy=os.environ['CHRONO_MISS'],
        base=config.base_checkpoint,config=config.to_dict(),tasks=tasks,
        init_offset=args.init_offset,trials=args.trials,seed=args.seed,max_steps=max_steps,
        gpu_name=gpu.name,torch=str(torch.__version__),cuda=torch.version.cuda,simulator_control_hz=hz,replan_interval=config.replan_steps,
        command_hz=hz,settling_steps=settling_steps,noise='PCG64 SeedSequence([seed,0,task,init]); draw once per observation tick',
        primary='fresh plan from current observation adopted by its scheduled deadline / all noninitial slots',
        secondary='requests completed within 50ms of observation availability / all issued noninitial requests',
        fallback='zero arm/retain previous gripper, one in-flight plan; adopt late result once at next available deadline',
        cache='shift previous executed plan endpoint by C=1; holds are actual history feedback; cold first plan excluded',
        gpu_sharing=os.environ.get('PI_FS_GPU_SHARING','unspecified'))
    manifest.update(protocol='native-fixed-deadline-grid-v1',
                    primary='hold ticks / all noninitial slots; task success on the ten paired starts',
                    secondary='fresh/late/committed/tail slots; requests ready by their scheduled absolute deadline',
                    fallback='hold until new plan' if os.environ['CHRONO_MISS']=='hold' else 'continue old time-aligned H=10 tail, hold only when it runs out',
                    cache=f'if enabled: shift by C={config.replan_steps}, chronological feedback from executed commands only')
    if isolated:
        manifest.update(render_isolated=True, render_gpu_name=torch.cuda.get_device_properties(1).name,
                        torch_intraop_threads=1,torch_interop_threads=1)
    if (out/'manifest.json').exists() and json.loads((out/'manifest.json').read_text())!=manifest:
        raise ValueError('Resume protocol differs')
    write_json(out/'manifest.json',manifest)
    rows=[json.loads(line) for line in (out/'episodes.jsonl').read_text().splitlines()] if (out/'episodes.jsonl').exists() else []
    completed={(r['task_id'],r['init_index']) for r in rows}
    if len(completed)!=len(rows) or not completed<=states.keys() or any(r['error'] for r in rows):
        raise ValueError('Invalid resumed episode rows')
    run_id=json.loads((out/'wandb.json').read_text())['id'] if (out/'wandb.json').exists() else None
    run=wandb.init(project=args.wandb_project,entity=args.wandb_entity,name=args.wandb_name,id=run_id,
                   resume='allow',mode=os.environ.get('WANDB_MODE', 'disabled'),job_type='c1-deadline-evaluation',dir=str(out),config=manifest)
    write_json(out/'wandb.json',dict(id=run.id,url=run.url))
    engine=None
    try:
        engine=CompiledC1Engine(config)
        warmed=False
        for (t,i),initial_state in states.items():
            if (t,i) in completed: continue
            wait_for_space(out,512*1024**2,event_path=out/'storage_events.jsonl')
            random.seed(args.seed); np.random.seed(args.seed)
            rng=np.random.default_rng(np.random.SeedSequence([args.seed,0,t,i]))
            task=tasks[str(t)]
            env=OffScreenRenderEnv(bddl_file_name=task['bddl'],camera_heights=256,camera_widths=256,control_freq=hz,
                                   horizon=max(1000, max_steps+settling_steps+1))
            try:
                if env.env.control_freq!=hz or env.language_instruction.split()!=task['prompt'].split():
                    raise ValueError('Loaded environment differs from registered task/clock')
                env.seed(args.seed); env.reset(); obs=env.set_init_state(initial_state)
                for _ in range(settling_steps):
                    obs,_,_,_=env.step([0.]*6+[-1.])
                if isolated:
                    from openpi.dual.c1_render_binding import verify_render_device
                    binding = verify_render_device(env)
                    if not warmed:
                        write_json(out/'gpu_binding.json',binding)
                if not warmed:
                    engine.warmup(observation_element(obs,task['prompt']))
                    warmed=True
                initial_sim_time = float(env.env.sim.data.time)
                row,trace=run_episode(env,engine,obs,task['prompt'],rng,max_steps,observation_element,period_ns=period_ns)
                row['simulation_elapsed_s'] = float(env.env.sim.data.time) - initial_sim_time
                row['model_physics_dt_s'] = float(env.env.model_timestep)
                row['wall_from_first_observation_s'] = (trace['ticks'][-1]['observed_ns'] - trace['plans'][0]['observation_ns']) / 1e9
                row['command_hz'] = hz
            finally:
                env.close()
            path=out/'traces'/f't{t}_i{i}_s{args.seed}.json.gz'
            with gzip.open(path.with_suffix('.tmp'),'wt',compresslevel=3) as stream:
                json.dump(trace,stream,allow_nan=False,separators=(',',':'))
            path.with_suffix('.tmp').replace(path)
            row.update(task_id=t,init_index=i,seed=args.seed,trace=str(path.relative_to(out)),
                       gpu_uuid=str(gpu.uuid),gpu_name=gpu.name,hostname=socket.gethostname(),
                       job_id=os.environ.get('SLURM_JOB_ID'),created_utc=datetime.now(timezone.utc).isoformat())
            if isolated:
                row.update(render_gpu_uuid=binding['render_gpu_uuid'], render_gpu_name=binding['render_gpu_name'],
                           gpu_binding=binding)
            with (out/'episodes.jsonl').open('a') as stream:
                stream.write(json.dumps(row,allow_nan=False)+'\n')
                stream.flush(); os.fsync(stream.fileno())
            rows.append(row)
            summary=summarize(rows)
            run.log({f'eval/{k}':v for k,v in summary.items() if v is not None} | {
                'episode/on_time_slots': row['fresh_on_time_ticks'],
                'episode/simulation_elapsed_s': row['simulation_elapsed_s'],
                'episode/wall_elapsed_s': row['wall_from_first_observation_s'],
                'episode/realized_command_hz': row['command_execution_hz'],
                'episode/target_command_hz': hz})
            write_json(out/'status.json',dict(status='running',completed=len(rows),expected=len(states)))
            print(json.dumps(dict(task=t,init=i,**summary)),flush=True)
        summary=summarize(rows)
        run.summary.update(summary)
        run.finish()
        write_json(out/'summary.json',summary)
        write_json(out/'status.json',dict(status='completed',completed=len(rows)))
    except BaseException as error:
        write_json(out/'status.json',dict(status='failed',completed=len(rows),error=repr(error)))
        run.finish(exit_code=1)
        raise
    finally:
        if engine is not None:
            engine.close()


if __name__=='__main__':
    main()
