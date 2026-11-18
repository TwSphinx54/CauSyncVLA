"""FR3 state-on Pi0.5 contract supplied with the step49999 checkpoint.

Absolute xyz/Euler/commanded-gripper, 1=open. Pad before quantile normalization;
rebase rx by a pure translation, wrap only the action delta.
"""
from dataclasses import dataclass, replace
import numpy as np
from openpi import transforms
from openpi.policies.libero_policy import LiberoInputs, LiberoOutputs
from openpi.models.model import ModelType

@dataclass(frozen=True)
class RebaseEuler:
    offset: float
    keys: tuple[str, ...] = ("state", "actions")
    def __call__(self, data):
        for key in self.keys:
            if key in data:
                value = np.asarray(data[key]).copy()
                value[..., 3] += self.offset
                data[key] = value
        return data

@dataclass(frozen=True)
class WrapActionAngles:
    def __call__(self, data):
        if "actions" in data:
            value = np.asarray(data["actions"]).copy()
            value[..., 3:6] = -((-value[..., 3:6] + np.pi) % (2 * np.pi) - np.pi)
            data["actions"] = value
        return data

@dataclass(frozen=True)
class FR3Inputs:
    def __call__(self, data):
        if np.asarray(data['observation/state']).shape[-1] != 7:
            raise ValueError('FR3 raw state is [x,y,z,rx,ry,rz,commanded_gripper]')
        out = LiberoInputs(ModelType.PI05)(data)
        return transforms.PadStatesAndActions(32)(out)

def data_transforms(_=None):
    mask = transforms.make_bool_mask(6, -1)
    return transforms.Group(
        inputs=[FR3Inputs(), RebaseEuler(-np.pi), transforms.DeltaActions(mask), WrapActionAngles()],
        outputs=[transforms.AbsoluteActions(mask), RebaseEuler(np.pi, ("actions",)), LiberoOutputs()],
    )

def fr3_config():
    from openpi.training.config import get_config, SimpleDataConfig, DataConfig, AssetsConfig
    cfg = get_config('pi05_libero')
    return replace(cfg, name='pi05_fr3_robotiq_state',
        model=replace(cfg.model, discrete_state_input=True, action_horizon=10, max_token_len=200),
        data=SimpleDataConfig(repo_id='fr3', assets=AssetsConfig(asset_id='fr3'),
            base_config=DataConfig(prompt_from_task=True), data_transforms=data_transforms))

def rebase_endpoint(endpoint, previous_state, current_state, norm):
    """Keep cached absolute targets fixed when the delta-action origin moves."""
    import torch
    def tensor(x): return torch.as_tensor(x, dtype=torch.float32, device=endpoint.device)
    slo, shi = tensor(norm['state'].q01), tensor(norm['state'].q99)
    alo, ahi = tensor(norm['actions'].q01), tensor(norm['actions'].q99)
    old = (previous_state.float()+1.)*.5*(shi-slo+1e-6)+slo
    now = (current_state.float()+1.)*.5*(shi-slo+1e-6)+slo
    physical = (endpoint.float()+1.)*.5*(ahi-alo+1e-6)+alo
    xyz = physical[..., :3] + (old[:, :3]-now[:, :3])[:, None]
    angles = physical[..., 3:6] + (old[:, 3:6]-now[:, 3:6])[:, None]
    angles = -torch.remainder(-angles+torch.pi, 2*torch.pi)+torch.pi
    delta = torch.cat((xyz, angles), dim=-1)
    rebased = (delta-alo[:6])/(ahi[:6]-alo[:6]+1e-6)*2.-1.
    return torch.cat((rebased, endpoint[..., 6:]), dim=-1)


def normalized_state(raw_state, norm):
    """Same state preprocessing as the full policy, without touching cameras."""
    state = transforms.pad_to_dim(np.asarray(raw_state, dtype=np.float32), 32, axis=-1).copy()
    state[3] -= np.pi
    return transforms.Normalize(norm, use_quantiles=True)({'state': state})['state']
