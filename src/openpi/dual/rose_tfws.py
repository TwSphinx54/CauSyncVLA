"""Training-free TFWS-v1 for a stock ROSE PyTorch OpenPI policy.

This is a mathematical port of Pi-FS's endpoint alignment, partial-flow boundary
and Euler schedule, rather than a direct import of Pi-FS model code. In
particular, explicit ``noise`` is a fresh Gaussian innovation even on warm calls;
it does not bypass endpoint reuse as the original Pi-FS helper currently does.
No temporal embedding, learned cache adapter, or SCFC branch is introduced.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor


def integration_schedule(configured_steps: int, start_time: float) -> tuple[int, float]:
    """Return actual NFE and Euler dt for integration from start_time to zero."""
    if isinstance(configured_steps, bool) or not isinstance(configured_steps, int) or configured_steps < 1:
        raise ValueError("configured_steps must be a positive integer")
    start_time = float(start_time)
    if not math.isfinite(start_time) or not 0.0 < start_time <= 1.0:
        raise ValueError("start_time must be finite and in (0, 1]")
    effective_nfe = max(1, math.ceil(configured_steps * start_time))
    return effective_nfe, -start_time / effective_nfe


def shift_repeat_endpoint(endpoint: Tensor, consumed_steps: int) -> Tensor:
    """Left-shift a normalized [B,H,D] endpoint and repeat its last action."""
    if endpoint.ndim != 3 or endpoint.shape[1] < 1:
        raise ValueError("endpoint must have shape [B,H,D] with H >= 1")
    if isinstance(consumed_steps, bool) or not isinstance(consumed_steps, int) or consumed_steps < 0:
        raise ValueError("consumed_steps must be a nonnegative integer")
    horizon = endpoint.shape[1]
    shift = min(consumed_steps, horizon)
    aligned = endpoint.clone()
    if shift:
        if shift < horizon:
            aligned[:, : horizon - shift, :] = endpoint[:, shift:, :]
        aligned[:, horizon - shift :, :] = endpoint[:, -1:, :].expand(-1, shift, -1)
    return aligned


def resolve_initial_latent(
    innovation: Tensor,
    endpoint: Tensor | None,
    *,
    enabled: bool,
    restart_time: float = 0.5,
    replan_steps: int = 5,
) -> tuple[Tensor, float, bool]:
    """Resolve full-noise cold starts or an all-position re-noised warm boundary.

    Both overlapping positions and the repeated tail receive fresh innovation.
    Returned time labels therefore match the partial-flow integration interval.
    """
    integration_schedule(1, restart_time)
    if innovation.ndim != 3:
        raise ValueError("innovation must have shape [B,H,D]")
    if not enabled or endpoint is None:
        return innovation, 1.0, False
    if endpoint.shape != innovation.shape:
        raise ValueError("cached endpoint and fresh innovation shapes differ")
    aligned = shift_repeat_endpoint(endpoint, replan_steps)
    aligned = aligned.to(device=innovation.device, dtype=innovation.dtype)
    start_time = float(restart_time)
    latent = (1.0 - start_time) * aligned + start_time * innovation
    return latent, start_time, True


class TrainingFreeTFWS:
    """Wrap one stock Policy instance with an episode-local TFWS endpoint.

    The supplied policy must already be constructed from the original LIBERO
    checkpoint with state tokenization off and temporal mode disabled. Its input
    and output transforms remain responsible for normalization and robot action
    conversion. Only the normalized model sampling boundary is replaced.

    This wrapper is single-session and deliberately rejects concurrent inference.
    Call ``reset()`` before every episode, and pass executed_steps=0 for its first
    replan and executed_steps=replan_steps for every subsequent replan. Endpoint
    publication occurs only after the stock output transform completes.
    """

    def __init__(
        self,
        policy: Any,
        *,
        enabled: bool,
        configured_steps: int,
        restart_time: float = 0.5,
        replan_steps: int = 5,
        warm_nfe: int | None = None,
    ) -> None:
        integration_schedule(configured_steps, restart_time)
        if warm_nfe is not None and (
            isinstance(warm_nfe, bool) or not isinstance(warm_nfe, int) or warm_nfe < 1
        ):
            raise ValueError("warm_nfe must be a positive integer or None")
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be bool")
        if isinstance(replan_steps, bool) or not isinstance(replan_steps, int) or not 1 <= replan_steps <= 10:
            raise ValueError("replan_steps must be an integer in [1,10]")
        if not getattr(policy, "_is_pytorch_model", False):
            raise ValueError("TrainingFreeTFWS requires a PyTorch Policy")
        if getattr(policy, "_training_free_tfws_owner", None) is not None:
            raise ValueError("the policy already has a TrainingFreeTFWS wrapper")
        model = policy.model
        config = model.config
        if not getattr(config, "pi05", False):
            raise ValueError("TrainingFreeTFWS requires pi0.5")
        if config.action_horizon != 10 or config.action_dim != 32:
            raise ValueError("this LIBERO protocol requires action_horizon=10 and action_dim=32")
        if getattr(config, "discrete_state_input", None) is not False:
            raise ValueError("original pi05_libero requires discrete_state_input=False")
        if getattr(getattr(config, "temporal", None), "enabled", False):
            raise ValueError("temporal mode must be disabled")
        if getattr(model, "temporal_mode_embedding", None) is not None:
            raise ValueError("a temporal mode embedding is incompatible with this stock-policy wrapper")
        if getattr(config, "use_scfc", False) or getattr(model, "_use_scfc", False):
            raise ValueError("SCFC must be disabled")
        sample_kwargs = dict(getattr(policy, "_sample_kwargs", {}))
        unsupported = set(sample_kwargs) - {"num_steps"}
        if unsupported:
            raise ValueError(f"unsupported stock sampling kwargs: {sorted(unsupported)}")

        self._policy = policy
        self._model = model
        self.enabled = enabled
        self.configured_steps = configured_steps
        self.restart_time = float(restart_time)
        self.replan_steps = replan_steps
        # None preserves the local Pi-FS schedule ceil(configured_steps * t).
        # An explicit warm budget supports comparisons at the same actual NFE.
        self.warm_nfe = warm_nfe
        self._stock_sample_actions = policy._sample_actions
        self._stock_sample_kwargs = sample_kwargs
        self._endpoint: Tensor | None = None
        self._pending_endpoint: Tensor | None = None
        self._last_sampler_metadata: dict[str, Any] | None = None
        self._completed_replans = 0
        self._in_infer = False

        policy._sample_kwargs = {"num_steps": configured_steps}
        policy._sample_actions = self._sample_actions
        policy._training_free_tfws_owner = self

    @property
    def model(self) -> Any:
        return self._policy.model

    @property
    def metadata(self) -> dict[str, Any]:
        return self._policy.metadata

    def reset(self) -> None:
        """Discard this session's action endpoints; never modify model weights."""
        if self._in_infer:
            raise RuntimeError("cannot reset during inference")
        self._clear_session()

    def _clear_session(self) -> None:
        self._endpoint = None
        self._pending_endpoint = None
        self._last_sampler_metadata = None
        self._completed_replans = 0

    def _synchronize(self) -> None:
        device = torch.device(self._policy._pytorch_device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def infer(self, obs: dict, *, noise: np.ndarray | None = None, executed_steps: int) -> dict:
        """Infer with a fresh innovation and atomically publish a valid endpoint."""
        if self._in_infer:
            raise RuntimeError("TrainingFreeTFWS does not support concurrent inference")
        self._in_infer = True
        try:
            expected_steps = 0 if self._completed_replans == 0 else self.replan_steps
            if (
                isinstance(executed_steps, bool)
                or not isinstance(executed_steps, int)
                or executed_steps != expected_steps
            ):
                raise ValueError(
                    f"executed_steps must be {expected_steps} at replan {self._completed_replans}; "
                    f"received {executed_steps!r}. Reset before each new episode."
                )
            self._pending_endpoint = None
            self._last_sampler_metadata = None
            self._synchronize()
            started = time.perf_counter()
            result = self._policy.infer(obs, noise=noise)
            self._synchronize()
            infer_ms = (time.perf_counter() - started) * 1000.0

            if self._pending_endpoint is None or self._last_sampler_metadata is None:
                raise RuntimeError("the stock policy did not call the TFWS sampler")
            if not isinstance(result, dict) or "actions" not in result:
                raise ValueError("the stock output transform did not return actions")
            if not np.isfinite(np.asarray(result["actions"])).all():
                raise ValueError("post-transformed actions are not finite")
            timing = dict(result.get("policy_timing", {}))
            timing.update(
                effective_nfe=self._last_sampler_metadata["effective_nfe"],
                start_time=self._last_sampler_metadata["start_time"],
                warm_start=self._last_sampler_metadata["warm_start"],
                synchronized_infer_ms=infer_ms,
            )
            result["policy_timing"] = timing
            result["tfws"] = {
                "method": "training_free_tfws_v1",
                "enabled": self.enabled,
                "configured_steps": self.configured_steps,
                "restart_time": self.restart_time,
                "warm_nfe_override": self.warm_nfe,
                "integration_budget": "explicit_warm_nfe" if self.warm_nfe is not None else "ceil_configured_steps_times_start_time",
                "replan_steps": self.replan_steps,
                "executed_steps": executed_steps,
                "replan_index": self._completed_replans,
                "endpoint_shape": list(self._pending_endpoint.shape),
                "endpoint_space": "normalized_model_actions",
                "tail_padding": "repeat_last_action",
                "noise_semantics": "fresh_full_horizon_innovation",
                **self._last_sampler_metadata,
            }
            # Input transforms, sampling, device transfer and output transforms
            # all succeeded; only now may the next replan consume this endpoint.
            self._endpoint = self._pending_endpoint if self.enabled else None
            self._pending_endpoint = None
            self._completed_replans += 1
            return result
        except Exception:
            self._clear_session()
            raise
        finally:
            self._in_infer = False

    @torch.no_grad()
    def _sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """Stock prefix construction plus an explicit full/partial Euler loop."""
        if not self._in_infer:
            raise RuntimeError("call TrainingFreeTFWS.infer instead of the wrapped policy directly")
        if num_steps != self.configured_steps:
            raise ValueError("the configured full-path step budget changed unexpectedly")
        model = self._model
        batch_size = observation.state.shape[0]
        if batch_size != 1:
            raise ValueError("episode-local TFWS requires policy batch size 1")
        shape = (batch_size, model.config.action_horizon, model.config.action_dim)
        innovation = model.sample_noise(shape, device) if noise is None else noise
        if not torch.is_tensor(innovation) or tuple(innovation.shape) != shape:
            raise ValueError(f"fresh innovation must be a tensor with shape {shape}")
        innovation = innovation.to(device=device, dtype=torch.float32)
        if not bool(torch.isfinite(innovation).all().item()):
            raise ValueError("fresh innovation is not finite")
        x_t, start_time, warm_start = resolve_initial_latent(
            innovation,
            self._endpoint,
            enabled=self.enabled,
            restart_time=self.restart_time,
            replan_steps=self.replan_steps,
        )
        effective_nfe, dt_value = integration_schedule(self.configured_steps, start_time)
        if warm_start and self.warm_nfe is not None:
            effective_nfe = self.warm_nfe
            dt_value = -start_time / effective_nfe

        images, image_masks, language_tokens, language_masks, state = model._preprocess_observation(
            observation, train=False
        )
        prefix_embeddings, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images, image_masks, language_tokens, language_masks
        )
        cumulative_masks = torch.cumsum(prefix_att_masks, dim=1)
        prefix_attention = cumulative_masks[:, None, :] <= cumulative_masks[:, :, None]
        prefix_attention &= prefix_pad_masks[:, None, :] * prefix_pad_masks[:, :, None]
        prefix_positions = torch.cumsum(prefix_pad_masks, dim=1) - 1
        attention_4d = model._prepare_attention_masks_4d(prefix_attention)
        model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
        _, past_key_values = model.paligemma_with_expert.forward(
            attention_mask=attention_4d,
            position_ids=prefix_positions,
            past_key_values=None,
            inputs_embeds=[prefix_embeddings, None],
            use_cache=True,
        )

        dt = torch.tensor(dt_value, dtype=torch.float32, device=device)
        flow_time = torch.tensor(start_time, dtype=torch.float32, device=device)
        observed_nfe = 0
        for _ in range(effective_nfe):
            velocity = model.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                flow_time.expand(batch_size),
            )
            x_t = x_t + dt * velocity
            flow_time += dt
            observed_nfe += 1
        if not bool(torch.isfinite(x_t).all().item()):
            raise ValueError("sampled normalized endpoint is not finite")
        self._pending_endpoint = x_t.detach().clone()
        self._last_sampler_metadata = {
            "effective_nfe": observed_nfe,
            "start_time": start_time,
            "dt": dt_value,
            "warm_start": warm_start,
        }
        return x_t
