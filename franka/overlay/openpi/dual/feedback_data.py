"""Updater history sidecars; original filtered dataset remains read-only."""
from pathlib import Path
import numpy as np
from openpi.dual.data import RawEpisodeDataset,PairBatch

class FeedbackDataset(RawEpisodeDataset):
    def __init__(self,*args,feedback_data,**kwargs):
        super().__init__(*args,**kwargs);self.side=Path(feedback_data);self.dense={};self.pairs={}
        for eid in self.ids:
            with np.load(self.side/f'episode_{eid:06d}.npz') as z:e={k:z[k] for k in z.files}
            self.dense[eid]=e
            # Eligible anchors and targets: causal, no gap or gripper barrier crossing.
            per={}
            for age in range(self.age_lo,self.age_hi+1):
                t=np.arange(age,len(e['state']));valid=e['segment'][t]==e['segment'][t-age];valid&=(e['time_s'][t]-e['time_s'][t-age])<=.25
                per[age]=t[valid]
            self.pairs[eid]=per
        self.split_manifest.update(feedback_sidecar=str(self.side),history_time='timestamped 100Hz synthetic history, original targets preserved',synthetic=True)
        self.sampling_manifest.update(strategy='task/episode/age balanced; half real targets when eligible; 60/30/10 percent 1/2/3 tick history thinning only under identical held commands',physical_dt='recorded timestamps, not relabelled fps')
    def dense_observation(self,eid,t):
        e=self.dense[eid];o=super().observation(eid,int(e['source_index'][t]));o['observation/state']=e['state'][t].copy();return o
    def sample_pairs(self,rng,batch_snapshots=2,targets_per_snapshot=4):
        # Each target gets its own real/held visual anchor; no reuse of predicted KV.
        p=batch_snapshots*targets_per_snapshot;sn=[];now=[];old=[];s0=[];s1=[];states=[];actions=[];ages=[];ids=[];aa=[];bb=[];intervals=[];real=[];cache_valid=[]
        task=getattr(self,'validation_task_id',None);tasks=sorted({self.episodes[i]['task_id'] for i in self.ids})
        for _ in range(p):
            task_i=int(rng.choice(tasks)) if task is None else task
            pool=[i for i in self.ids if self.episodes[i]['task_id']==task_i]
            while True:
                eid=int(rng.choice(pool));age=int(rng.choice(self._age_values,p=self._age_probabilities));ts=self.pairs[eid][age]
                if len(ts):break
            e=self.dense[eid];rs=ts[e['observed'][ts]]
            if len(rs) and rng.random()<.5:ts=rs
            t1=int(rng.choice(ts));t0=t1-age
            sn.append(self.dense_observation(eid,t0));now.append(self.dense_observation(eid,t1));s0.append(e['state'][t0]);s1.append(e['state'][t1])
            st=np.zeros((self.history,7),np.float32);ac=st.copy();dt=np.zeros(self.history,np.float32)
            points=np.arange(t0,t1+1)
            stride=int(rng.choice([1,2,3],p=[.6,.3,.1]))
            proposed=np.unique(np.r_[np.arange(t0,t1,stride),t1])
            # Combine samples only while the identical original target was held.
            if all(np.all(e['actions'][a:b]==e['actions'][a]) for a,b in zip(proposed,proposed[1:])):points=proposed
            age=len(points)-1
            st[:age]=e['state'][points[1:]];ac[:age]=e['actions'][points[:-1]];dt[:age]=np.diff(e['time_s'][points]);states.append(st);actions.append(ac);intervals.append(dt)
            ages.append(age);ids.append(eid);aa.append(t0);bb.append(t1);real.append(bool(e['observed'][t1]))
            shift=getattr(self,'cache_replan_steps',0);prev=int(points[-1-shift]) if len(points)>shift else max(0,t1-shift*stride);cache_valid.append(prev<t1 and e['segment'][prev]==e['segment'][t1]);old.append(self.dense_observation(eid,prev))
        batch=PairBatch(sn,now,np.arange(p),np.asarray(s0),np.asarray(s1),np.asarray(states),np.asarray(actions),np.asarray(ages),np.asarray(ids),np.asarray(aa),np.asarray(bb),old)
        batch.cache_valid=np.asarray(cache_valid);batch.interval_s=np.asarray(intervals);batch.target_observed=np.asarray(real);return batch
