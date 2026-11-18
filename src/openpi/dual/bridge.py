"""Frozen ROSE stock Pi0.5, compact post-RoPE prefix and differentiable hand.

No local legacy Pi-FS model is imported. The only model operations used here are
the stock remote embed_prefix/denoise_step APIs. Gradients to predicted KV must
remain enabled in hand(), even though all foundation-model weights are frozen.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import os
from pathlib import Path

import jax
import numpy as np
import torch
from transformers.cache_utils import DynamicCache

from openpi.models.model import Observation
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
from openpi.policies.policy_config import create_trained_policy
from openpi.training.config import get_config


@dataclass
class PrefixKV:
    k: torch.Tensor
    v: torch.Tensor
    mask: torch.Tensor
    state: torch.Tensor


def file_sha256(path):
    if os.environ.get("PI_FS_SKIP_HASH_VERIFICATION") == "1":
        return None
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class FrozenPi05:
    def __init__(self, checkpoint, device='cuda', text_tokens=32):
        if not 1 <= text_tokens <= 200:
            raise ValueError('text_tokens must be in [1,200]')
        self.checkpoint = str(Path(checkpoint).resolve())
        self.device = torch.device(device)
        self.text_tokens = text_tokens
        config = get_config('pi05_libero')
        config = replace(config, model=replace(config.model, pytorch_compile_mode=None))
        self.policy = create_trained_policy(config, self.checkpoint, pytorch_device=str(self.device))
        self.model = self.policy._model
        self.model.requires_grad_(False).eval()
        if (not self.model.pi05 or self.model.config.discrete_state_input or
                self.model.config.action_horizon != 10 or self.model.config.action_dim != 32):
            raise ValueError('Requires original state-off pi05_libero H10 D32')

    def transform(self, observations):
        if not observations:
            raise ValueError('Empty observation batch')
        rows = [self.policy._input_transform(jax.tree.map(lambda x: x, obs)) for obs in observations]
        batch = jax.tree.map(lambda *values: torch.as_tensor(np.stack(values), device=self.device), *rows)
        return Observation.from_dict(batch)

    @torch.no_grad()
    def prefix(self, observations):
        return self.prefix_observation(self.transform(observations))

    @torch.no_grad()
    def prefix_observation(self, observation):
        images, image_masks, tokens, text_mask, state = self.model._preprocess_observation(observation, train=False)
        if bool(text_mask[:, self.text_tokens:].any()):
            raise ValueError('Active text exceeds compact capacity; increase updater.text_tokens and retrain')
        if len(images) != 3 or not bool(image_masks[0].all()) or not bool(image_masks[1].all()) or bool(image_masks[2].any()):
            raise ValueError('Compact layout requires valid base/left-wrist and masked right-wrist images')
        return self.prefix_tensors(images[:2], image_masks[:2],
                                   tokens[:, :self.text_tokens], text_mask[:, :self.text_tokens], state)

    def prefix_tensors(self, images, image_masks, tokens, text_mask, state):
        """Compile boundary; caller must validate compact layout and text length."""
        embs, mask, ar_mask = self.model.embed_prefix(images, image_masks, tokens, text_mask)
        if embs.shape[1] != 512 + self.text_tokens:
            raise ValueError('Expected 256 tokens per image')
        positions = mask.long().cumsum(1) - 1
        attention = self.model._prepare_attention_masks_4d(make_att_2d_masks(mask, ar_mask))
        self.model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = 'eager'
        _, cache = self.model.paligemma_with_expert(
            attention_mask=attention, position_ids=positions, past_key_values=None,
            inputs_embeds=[embs, None], use_cache=True)
        keys = torch.stack([cache[i][0][:, 0] for i in range(len(cache))], dim=1)
        values = torch.stack([cache[i][1][:, 0] for i in range(len(cache))], dim=1)
        if keys.shape[1] != 18 or keys.shape[-1] != 256 or cache[0][0].shape[1] != 1:
            raise ValueError('Unexpected stock Pi0.5 KV layout')
        return PrefixKV(keys, values, mask, state)

    def hand(self, k, v, mask, state, noise, num_steps=1, start_time=1.):
        """Euler integration with a fresh cache object per call; differentiable KV.

        Cache keys are already RoPE transformed. The stock expert appends only
        its own rotated suffix and uses sum(mask) for suffix position offsets.
        """
        if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps < 1:
            raise ValueError('num_steps must be a positive integer')
        if not 0 < start_time <= 1:
            raise ValueError('start_time must be in (0,1]')
        if k.shape != v.shape or k.ndim != 4 or mask.shape != (k.shape[0], k.shape[2]):
            raise ValueError('Incompatible KV/mask shape')
        if noise.shape != (k.shape[0], 10, 32):
            raise ValueError('Expected noise [B,10,32]')
        # Casting retains the autograd path through the frozen expert.
        dtype = self.model.paligemma_with_expert.gemma_expert.model.layers[0].self_attn.k_proj.weight.dtype
        cache = DynamicCache.from_legacy_cache(tuple(
            (k[:, layer, None].to(dtype), v[:, layer, None].to(dtype)) for layer in range(k.shape[1])))
        x = noise.float()
        dt = -float(start_time) / num_steps
        for step in range(num_steps):
            t = torch.full((x.shape[0],), start_time + step * dt, device=x.device, dtype=torch.float32)
            x = x + dt * self.model.denoise_step(state, mask, cache, x, t)
        return x

    def output_actions(self, normalized_actions, state):
        result = self.policy._output_transform({
            'actions': normalized_actions[0].detach().float().cpu().numpy(),
            'state': state[0].detach().float().cpu().numpy()})
        actions = np.asarray(result['actions'], dtype=np.float32)
        if actions.shape != (10, 7) or not np.isfinite(actions).all():
            raise ValueError('Invalid post-transformed actions')
        return actions
