from pathlib import Path
from dataclasses import asdict
import json,torch
from openpi.dual.config import UpdaterConfig
from openpi.dual.model import SelfUpdater,UpdaterNorm
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Expand the 6k updater's time inputs for FR3 fine-tuning.")
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    R, p = args.out, args.checkpoint
    R.mkdir(parents=True, exist_ok=False)
    with torch.serialization.safe_globals([np.core.multiarray._reconstruct, np.ndarray, np.dtype,
                                           type(np.dtype('uint32')), torch.torch_version.TorchVersion]):
        old = torch.load(p, map_location='cpu', weights_only=True)
    c=UpdaterConfig(**{**old['updater_config'],'time_conditioned':True});m=SelfUpdater(c,UpdaterNorm.from_dict(old['norm']))
    s=dict(old['updater']);key='condition_encoder.0.weight';x=m.state_dict()[key].clone();x.zero_();x[:,:s[key].shape[1]]=s[key];s[key]=x;m.load_state_dict(s,strict=True)
    assert all(torch.equal(a,b) for a,b in zip(m.norm.to_dict().values(),old['norm'].values()))
    old['updater_config']=asdict(c);old['updater']=m.state_dict();old['optimizer']=torch.optim.AdamW(m.parameters(),lr=3e-5).state_dict();old['selection']=None
    old['provenance']['lineage_id']='fr3-feedback100-20261114'
    old['feedback_adaptation']=dict(initial_updater=str(p),old_global_step=old['global_step'],extra_time_columns_zero_initialized=True,optimizer_reset=True,original_expert_frozen=True)
    torch.save(old,R/'init.pt')
    (R/'initialization.json').write_text(json.dumps(dict(config=asdict(c),**old['feedback_adaptation']),indent=2))
    print(json.dumps(old['feedback_adaptation']))

if __name__ == '__main__':
    main()
