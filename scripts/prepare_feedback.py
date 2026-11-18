"""Read-only filtered LeRobot source -> timestamped updater-only sidecars."""
from pathlib import Path
import json, numpy as np, pyarrow.parquet as pq
from scipy.spatial.transform import Rotation,Slerp
import argparse

def build(state,actions,times,events):
    n=len(state); states=[]; command=[]; stamps=[]; indices=[]; observed=[]; segments=[]; seg=0
    for i in range(n):
        states.append(state[i]);command.append(actions[i]);stamps.append(times[i]);indices.append(i);observed.append(True);segments.append(seg)
        if i==n-1:break
        dt=times[i+1]-times[i]
        blocked=dt>.15 or state[i,6]!=state[i+1,6] or any(times[i]<=a<=times[i+1] or times[i]<=b<=times[i+1] or a<=times[i]<=b for a,b in events)
        if blocked:seg+=1;continue
        # Offline synthetic states only. No invented RGB or reduced action targets.
        ts=np.arange(times[i]+.01,times[i+1]-.001,.01);alpha=(ts-times[i])/dt
        if len(ts):
            rot=Slerp([0,1],Rotation.from_euler('xyz',state[i:i+2,3:6]))(alpha).as_euler('xyz');rot[:,0]%=2*np.pi
            for j,t in enumerate(ts):
                s=state[i].copy();s[:3]=state[i,:3]+alpha[j]*(state[i+1,:3]-state[i,:3]);s[3:6]=rot[j]
                states.append(s);command.append(actions[i]);stamps.append(t);indices.append(i);observed.append(False);segments.append(seg)
    ts=np.asarray(stamps);return dict(state=np.asarray(states,np.float32),actions=np.asarray(command,np.float32),time_s=ts-ts[0],source_index=np.asarray(indices,np.int64),observed=np.asarray(observed,bool),segment=np.asarray(segments,np.int64))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--timeline', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    SOURCE, TIMELINE, OUT = args.source, args.timeline, args.out
    OUT.mkdir(parents=True, exist_ok=False)
    split = json.loads(args.split.read_text())
    ids=sorted(split['train_episode_ids']+split['val_episode_ids']); timeline={x['episode_index']:x for x in map(json.loads,TIMELINE.open())}; records=[]
    for eid in ids:
        p=SOURCE/'data/chunk-000'/f'episode_{eid:06d}.parquet';before=p.stat();raw=pq.read_table(p,columns=['observation.state','action']).to_pydict();state=np.asarray(raw['observation.state'],np.float32);actions=np.asarray(raw['action'],np.float32)
        m=timeline[eid];t=np.asarray(m['raw_timestamp_sec'],float)[m['retained_raw_row_index']];assert len(t)==len(state) and np.all(np.diff(t)>0)
        events=[];begin=None
        for e in m['gripper_events']:
            if e['event']=='gripper_command':begin=float(e['timestamp_sec'])
            elif e['event']=='gripper_result' and begin is not None:events.append((begin,float(e['timestamp_sec'])));begin=None
        if begin is not None:events.append((begin,float(t[-1])))
        out=build(state,actions,t,events);obs=out['observed'];assert np.array_equal(out['state'][obs],state) and np.array_equal(out['actions'],actions[out['source_index']]);assert np.array_equal(out['source_index'][obs],np.arange(len(state)))
        assert (p.stat().st_size,p.stat().st_mtime_ns)==(before.st_size,before.st_mtime_ns)
        np.savez_compressed(OUT/f'episode_{eid:06d}.npz',**out)
        records.append(dict(episode=eid,source_rows=len(state),rows=len(obs),synthetic_rows=int((~obs).sum()),segments=int(out['segment'].max()+1),source_size=before.st_size,source_mtime_ns=before.st_mtime_ns))
    (OUT/'split.json').write_text(json.dumps(split,indent=2))
    manifest=dict(source=str(SOURCE),timeline=str(TIMELINE),source_filter_unchanged=True,expert_finetune=False,action='original next-retained-state proxy target held, never interpolated',state='synthetic linear XYZ/SO3 interpolation only inside <=150ms intervals without gripper events',rgb='causal source frame held; real-frame targets separately identified',time='original saved timestamps of retained rows, not frame_index/10',target_feedback_hz=100,train_episodes=len(split['train_episode_ids']),val_episodes=len(split['val_episode_ids']),episodes=records)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({k:v for k,v in manifest.items() if k!='episodes'}));print('rows',sum(x['rows'] for x in records),'source_rows',sum(x['source_rows'] for x in records))
if __name__=='__main__':main()
