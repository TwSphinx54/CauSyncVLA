"""Direct snapshot-to-current KV predictor from proprioception/action history.

The frozen brain supplies latent targets; this model never reads new images.
Keys stay in their cached, post-RoPE coordinates. Predictions are always anchored
at a real prefix, never fed recursively back as the next real snapshot.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn

from openpi.dual.config import UpdaterConfig


@dataclass(frozen=True)
class UpdaterNorm:
    kv_sd: Tensor
    dkv_sd: Tensor
    state_mu: Tensor
    state_sd: Tensor

    def validate(self, layers: int, state_dim: int = 8) -> None:
        for name, shape in (("kv_sd", (2, layers)), ("dkv_sd", (2, layers)), ("state_mu", (state_dim,)), ("state_sd", (state_dim,))):
            value = getattr(self, name)
            if not isinstance(value, Tensor) or tuple(value.shape) != shape:
                raise ValueError(f"{name} must be a tensor with shape {shape}")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} contains nonfinite normalization statistics")
            if name != "state_mu" and not bool((value > 0).all()):
                raise ValueError(f"{name} must be strictly positive")

    def to(self, device: torch.device | str) -> "UpdaterNorm":
        return UpdaterNorm(**{name: getattr(self, name).to(device=device, dtype=torch.float32) for name in self.__dataclass_fields__})

    def to_dict(self) -> dict[str, Tensor]:
        return {name: getattr(self, name).detach().cpu().clone() for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, values: Mapping[str, Tensor]) -> "UpdaterNorm":
        if set(values) != set(cls.__dataclass_fields__):
            raise ValueError("normalization checkpoint must contain exactly kv_sd, dkv_sd, state_mu, state_sd")
        return cls(**values)


class AdaLNBlock(nn.Module):
    """DiT-style residual attention/MLP block, an identity at initialization."""

    def __init__(self, hidden: int, heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden, elementwise_affine=False)
        self.attention = nn.MultiheadAttention(hidden, heads, dropout=0.0, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(hidden, 4 * hidden), nn.GELU(), nn.Linear(4 * hidden, hidden))
        self.modulation = nn.Linear(hidden, 6 * hidden)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x: Tensor, condition: Tensor, mask: Tensor) -> Tensor:
        scale1, shift1, gate1, scale2, shift2, gate2 = self.modulation(condition).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + scale1[:, None]) + shift1[:, None]
        attended, _ = self.attention(h, h, h, key_padding_mask=~mask, need_weights=False)
        x = x + gate1[:, None] * attended
        h = self.norm2(x) * (1 + scale2[:, None]) + shift2[:, None]
        return x + gate2[:, None] * self.mlp(h)


class SelfUpdater(nn.Module):
    """Report architecture: 18-layer K/V -> 384-wide six-block latent model.

    All learned computations use token-aligned K/V and a 745D causal execution
    record. The 512 visual and configured text slots retain fixed positions;
    inactive slots are masked in attention and receive no residual delta.
    """

    def __init__(self, config: UpdaterConfig, norm: UpdaterNorm):
        super().__init__()
        norm.validate(config.layers, config.state_dim)
        self.config = config
        for name in norm.__dataclass_fields__:
            self.register_buffer(name, getattr(norm, name).detach().float().clone())
        hidden = config.hidden_dim
        self.input_projection = nn.Linear(config.kv_features, hidden)
        self.position_embedding = nn.Parameter(torch.zeros(1, config.tokens, hidden))
        nn.init.normal_(self.position_embedding, std=0.02)
        self.condition_encoder = nn.Sequential(
            nn.Linear(config.condition_dim, 2 * hidden), nn.GELU(), nn.Linear(2 * hidden, hidden)
        )
        self.condition_projection = nn.Linear(hidden, hidden)
        self.blocks = nn.ModuleList(AdaLNBlock(hidden, config.heads) for _ in range(config.depth))
        self.output_norm = nn.LayerNorm(hidden)
        self.output_projection = nn.Linear(hidden, config.kv_features)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    @property
    def norm(self) -> UpdaterNorm:
        return UpdaterNorm(self.kv_sd, self.dkv_sd, self.state_mu, self.state_sd)

    def forward(self, k: Tensor, v: Tensor, mask: Tensor, condition: Tensor, target_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        cfg = self.config
        if k.ndim != 4 or v.shape != k.shape or tuple(k.shape[1:]) != (cfg.layers, cfg.tokens, cfg.head_dim):
            raise ValueError(f"K/V must have shape [B,{cfg.layers},{cfg.tokens},{cfg.head_dim}]")
        batch = k.shape[0]
        if mask.shape != (batch, cfg.tokens) or mask.dtype != torch.bool:
            raise ValueError("prefix mask must be bool [B,tokens]")
        if condition.shape != (batch, cfg.condition_dim):
            raise ValueError(f"condition must have shape [B,{cfg.condition_dim}]")
        if not bool(mask.any(dim=1).all()):
            raise ValueError("every sample must contain at least one active prefix token")
        if k.device != v.device or mask.device != k.device or condition.device != k.device:
            raise ValueError("K, V, mask and condition must be on one device")
        if not bool(torch.isfinite(condition).all()) or bool(((condition[:, 0] < 0) | (condition[:, 0] > 1)).any()):
            raise ValueError("condition must be finite with normalized age in [0,1]")
        if target_mask is not None and (target_mask.shape != mask.shape or target_mask.dtype != torch.bool):
            raise ValueError("target_mask must match the source mask layout")
        return self.forward_tensor(k, v, mask, condition, target_mask)

    def forward_tensor(self, k: Tensor, v: Tensor, mask: Tensor, condition: Tensor, target_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Pure tensor core for compilation after the caller validates inputs.

        Training and ordinary inference keep forward() checks. Compile adapters
        must validate dynamic values before calling this core; no checks silently
        disappear from the public forward contract.
        """
        cfg = self.config
        batch = k.shape[0]
        if target_mask is not None:
            # Newly active state-text slots start at zero, never at padding KV.
            k = torch.where(mask[:, None, :, None], k, 0.)
            v = torch.where(mask[:, None, :, None], v, 0.)
        output_mask = mask if target_mask is None else target_mask
        # Flatten [K_0 ... K_L-1, V_0 ... V_L-1] per token, never interleave K/V.
        key = (k.float() / self.kv_sd[0][None, :, None, None]).permute(0, 2, 1, 3).flatten(2)
        value = (v.float() / self.kv_sd[1][None, :, None, None]).permute(0, 2, 1, 3).flatten(2)
        token_input = torch.cat((key, value), dim=-1)
        token_input = torch.where(mask[..., None], token_input, 0.0)
        dtype = self.input_projection.weight.dtype
        c = self.condition_encoder(condition.to(dtype))
        x = self.input_projection(token_input.to(dtype)) + self.position_embedding + self.condition_projection(c)[:, None]
        for block in self.blocks:
            x = block(x, c, output_mask)
        residual = self.output_projection(self.output_norm(x)).float()
        dk, dv = residual.chunk(2, dim=-1)
        dk = dk.reshape(batch, cfg.tokens, cfg.layers, cfg.head_dim).permute(0, 2, 1, 3)
        dv = dv.reshape(batch, cfg.tokens, cfg.layers, cfg.head_dim).permute(0, 2, 1, 3)
        # A zero-age true snapshot is already current, even after training.
        active = output_mask[:, None, :, None] & (condition[:, :1, None, None] > 0)
        dk = torch.where(active, dk * self.dkv_sd[0][None, :, None, None], 0.0)
        dv = torch.where(active, dv * self.dkv_sd[1][None, :, None, None], 0.0)
        # Keep latent regression in FP32. The frozen expert bridge casts only
        # at its attention boundary, preserving gradients through that cast.
        return k.float() + dk, v.float() + dv


def masked_kv_loss(
    pred_k: Tensor,
    pred_v: Tensor,
    target_k: Tensor,
    target_v: Tensor,
    mask: Tensor,
    norm: UpdaterNorm,
) -> Tensor:
    """Mean squared latent error, normalized per layer and valid prefix token."""
    if pred_k.ndim != 4 or not (pred_k.shape == pred_v.shape == target_k.shape == target_v.shape):
        raise ValueError("all K/V loss tensors must have the same [B,L,T,D] shape")
    if mask.shape != (pred_k.shape[0], pred_k.shape[2]) or mask.dtype != torch.bool:
        raise ValueError("loss mask must have shape [B,T] and bool dtype")
    if not bool(mask.any(dim=1).all()):
        raise ValueError("loss cannot average an empty prefix")
    scale = norm.dkv_sd.to(device=pred_k.device, dtype=torch.float32)
    if scale.shape != (2, pred_k.shape[1]) or not bool((scale > 0).all()):
        raise ValueError("invalid KV delta normalization")
    valid = mask[:, None, :, None]
    ek = torch.where(valid, (pred_k.float() - target_k.detach().float()) / scale[0][None, :, None, None], 0.0)
    ev = torch.where(valid, (pred_v.float() - target_v.detach().float()) / scale[1][None, :, None, None], 0.0)
    denominator = mask.sum() * pred_k.shape[1] * pred_k.shape[3]
    return (ek.square().sum() + ev.square().sum()) / (2 * denominator)
