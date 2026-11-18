"""Causal state/action features shared verbatim by training and deployment."""
from __future__ import annotations

import torch
from torch import Tensor

from openpi.dual.model import UpdaterNorm


def condition_features(
    snapshot_state: Tensor,
    current_state: Tensor,
    states_after: Tensor,
    executed_actions: Tensor,
    ages: Tensor,
    norm: UpdaterNorm,
    history: int = 48,
) -> Tensor:
    """Encode snapshot-to-now history; never consumes predicted future actions.

    For age k: states_after[:, :k] = s[t0+1:t1+1], executed_actions[:, :k]
    = a[t0:t1]. Both histories are chronological, left aligned, then zero
    padded. State displacements are relative to s[t0]; actions keep raw robot
    units. Ages greater than the trained capacity are rejected, not truncated.
    """
    if isinstance(history, bool) or not isinstance(history, int) or history < 1:
        raise ValueError("history must be a positive integer")
    if snapshot_state.ndim != 2 or snapshot_state.shape[1] != 8:
        raise ValueError("snapshot_state must have shape [B,8]")
    batch = snapshot_state.shape[0]
    if current_state.shape != snapshot_state.shape:
        raise ValueError("current_state and snapshot_state shapes differ")
    if states_after.shape != (batch, history, 8) or executed_actions.shape != (batch, history, 7):
        raise ValueError("history must have shape [B,history,8/7]")
    device = snapshot_state.device
    ages = torch.as_tensor(ages, device=device)
    if ages.shape != (batch,) or ages.dtype == torch.bool or ages.is_floating_point():
        raise ValueError("ages must be an integer tensor of shape [B]")
    if bool(((ages < 0) | (ages > history)).any()):
        raise ValueError("age outside [0,history]; refresh the real prefix instead of truncating")
    for value in (snapshot_state, current_state, states_after, executed_actions):
        if value.device != device:
            raise ValueError("all state/action history tensors must use the same device")
    valid = torch.arange(history, device=device)[None, :] < ages[:, None]
    s0, s1 = snapshot_state.float(), current_state.float()
    # Invalid padding may contain NaNs in upstream storage; select it away before
    # finite checks and normalization, so only observed history is consumed.
    history_states = torch.where(valid[..., None], states_after.float(), s0[:, None, :])
    history_actions = torch.where(valid[..., None], executed_actions.float(), 0.0)
    if not all(bool(torch.isfinite(value).all()) for value in (s0, s1, history_states, history_actions)):
        raise ValueError("observed state/action history contains nonfinite values")
    mean = norm.state_mu.to(device=device, dtype=torch.float32)
    std = norm.state_sd.to(device=device, dtype=torch.float32)
    if mean.shape != (8,) or std.shape != (8,) or not bool((std > 0).all()):
        raise ValueError("invalid state normalization statistics")
    relative_states = (history_states - s0[:, None, :]) / std
    return torch.cat(
        (
            ages.float()[:, None] / history,
            (s0 - mean) / std,
            (s1 - mean) / std,
            (s1 - s0) / std,
            relative_states.flatten(1),
            history_actions.flatten(1),
        ),
        dim=1,
    )
