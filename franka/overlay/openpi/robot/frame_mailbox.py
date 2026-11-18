"""Local-only latest observation mailbox; atomic file replacement, no command path."""
from pathlib import Path
import threading,time
import numpy as np

class FrameMailbox:
    def __init__(self,path,worker,skip_current=False,initial_frame_key=None):
        self.frame_key=initial_frame_key;self.path=Path(path);self.initial_seen=self.path.stat().st_mtime_ns if skip_current and self.path.exists() else None;self.worker=worker;self.stop=threading.Event();self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start()
    def run(self):
        seen=self.initial_seen
        while not self.stop.wait(.002):
            try:
                stamp=self.path.stat().st_mtime_ns
                if stamp==seen:continue
                with np.load(self.path,allow_pickle=False) as f:
                    obs={'observation/state':f['state'],'observation/image':f['front'],'observation/wrist_image':f['wrist'],'prompt':str(f['prompt'])}
                    seq=int(f['seq']);frame_key=str(f['frame_key'])
                    if 'feedback_observed_ns' in f:obs['feedback_observed_ns']=int(f['feedback_observed_ns'])
                seen=stamp
                if frame_key==getattr(self,'frame_key',None):continue
                self.frame_key=frame_key;self.worker.observe(obs,seq)
            except FileNotFoundError:continue
            except Exception as exc:
                with self.worker.condition:self.worker.error=exc
                return
    def close(self):self.stop.set();self.thread.join()

def publish(path,obs,seq):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    meta=obs.get('camera_metadata',{})
    key=str([(k,v.get('seq')) for k,v in sorted(meta.items())]) if meta else str(seq)
    temp=p.with_suffix('.tmp')
    with temp.open('wb') as f:np.savez(f,state=obs['observation/state'],front=obs['observation/image'],wrist=obs['observation/wrist_image'],prompt=obs['prompt'],seq=seq,frame_key=key,feedback_observed_ns=obs.get('feedback_observed_ns',0))
    temp.replace(p)
