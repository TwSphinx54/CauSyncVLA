"""Portable inference checkpoints for the frozen action expert and trained KV updater."""
import hashlib
import json
from pathlib import Path

from safetensors import safe_open
import torch

from openpi.dual.config import UpdaterConfig
from openpi.dual.model import SelfUpdater, UpdaterNorm


def norm_signature(path):
    """Identify the numerical normalization asset independently of JSON formatting."""
    values = json.loads(Path(path).read_text())
    canonical = json.dumps(values, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def load_adapter(path, config, device):
    path = Path(path)
    if path.suffix == '.safetensors':
        with safe_open(path, framework='pt', device='cpu') as f:
            metadata = f.metadata() or {}
            if metadata.get('format') != 'causyncvla-updater-v1':
                raise ValueError('Unknown updater format')
            cfg = UpdaterConfig(**json.loads(metadata['architecture']))
            base = Path(config.base_checkpoint)
            profile = getattr(config, 'robot_profile', 'libero')
            if metadata['profile'] != profile:
                raise ValueError('The updater belongs to a different robot profile')
            expected_asset = {'libero': 'assets/physical-intelligence/libero/norm_stats.json',
                              'fr3': 'assets/fr3/norm_stats.json'}[profile]
            if norm_signature(base / expected_asset) != metadata['action_normalization']:
                raise ValueError('The action normalization is incompatible with this updater')
            state = {key: f.get_tensor(key) for key in f.keys()}
            norm = {key: state[key] for key in UpdaterNorm.__dataclass_fields__}
            provenance = {'profile': profile, 'global_step': int(metadata['global_step'])}
    else:
        # Full checkpoints from this trainer also contain per-rank NumPy RNG state.
        import numpy as np
        with torch.serialization.safe_globals([np.core.multiarray._reconstruct, np.ndarray,
                                               np.dtype, type(np.dtype('uint32')),
                                               torch.torch_version.TorchVersion]):
            saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved.get('schema_version') != 1:
            raise ValueError('Unknown training checkpoint schema')
        cfg, norm, state = UpdaterConfig(**saved['updater_config']), saved['norm'], saved['updater']
        provenance = saved['provenance']
        if Path(provenance['base_checkpoint']['path']).resolve() != Path(config.base_checkpoint).resolve():
            raise ValueError('Training checkpoint specifies a different base model directory')
    if cfg != config.updater:
        raise ValueError('Updater architecture differs from the runtime configuration')
    for name, tensor in norm.items():
        if not torch.equal(tensor, state[name]):
            raise ValueError(f'Normalization buffer differs from checkpoint: {name}')
    updater = SelfUpdater(cfg, UpdaterNorm.from_dict(norm))
    updater.load_state_dict(state, strict=True)
    return updater.to(device).eval(), provenance
