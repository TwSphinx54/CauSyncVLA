"""Explicit, independent switches for stock, endpoint cache and dual inference.

This module has no ML imports, so configuration can be inspected without loading
OpenPI or CUDA. Unknown JSON fields fail rather than silently disabling an arm.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import json
import math
from pathlib import Path
from typing import Any, TypeVar


def _integer(name: str, value: int, low: int, high: int | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < low or (high is not None and value > high):
        raise ValueError(f"{name} must be an integer in [{low}, {high or 'inf'}], got {value!r}")


def _positive(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _boolean(name: str, value: bool) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")


@dataclass(frozen=True)
class UpdaterConfig:
    history: int = 48
    layers: int = 18
    head_dim: int = 256
    hidden_dim: int = 384
    depth: int = 6
    heads: int = 4
    text_tokens: int = 32
    state_dim: int = 8
    action_dim: int = 7
    time_conditioned: bool = False

    def __post_init__(self) -> None:
        for name in ("history", "layers", "head_dim", "hidden_dim", "depth", "heads", "text_tokens", "state_dim", "action_dim"):
            _integer(name, getattr(self, name), 1)
        if self.hidden_dim % self.heads:
            raise ValueError("hidden_dim must be divisible by heads")
        if self.text_tokens > 200:
            raise ValueError("text_tokens exceeds stock pi05_libero's 200 token capacity")
        if (self.state_dim, self.action_dim) not in {(8, 7), (7, 7)}:
            raise ValueError("Supported robot histories are LIBERO 8/7 and FR3 7/7")

    @property
    def tokens(self) -> int:
        return 512 + self.text_tokens

    @property
    def condition_dim(self) -> int:
        return 1 + 3 * self.state_dim + self.history * (self.state_dim + self.action_dim) + (self.history + 1 if self.time_conditioned else 0)

    @property
    def kv_features(self) -> int:
        return 2 * self.layers * self.head_dim


@dataclass(frozen=True)
class CacheConfig:
    enabled: bool = False
    restart_time: float = 0.5
    warm_nfe: int | None = None

    def __post_init__(self) -> None:
        _boolean("cache.enabled", self.enabled)
        _positive("cache.restart_time", self.restart_time)
        if self.restart_time > 1:
            raise ValueError("cache.restart_time must be in (0, 1]")
        if self.warm_nfe is not None:
            _integer("cache.warm_nfe", self.warm_nfe, 1)


@dataclass(frozen=True)
class DualConfig:
    enabled: bool = False
    updater_checkpoint: str | None = None
    mode: str = "threaded"
    brain_ms: float = 73.5
    action_ms: float = 3.75
    updater_enabled: bool = True
    updater_min_age: int = 1
    max_age: int = 48

    def __post_init__(self) -> None:
        _boolean("dual.enabled", self.enabled)
        _boolean("dual.updater_enabled", self.updater_enabled)
        if self.updater_checkpoint is not None and (
            not isinstance(self.updater_checkpoint, str) or not self.updater_checkpoint.strip()
        ):
            raise ValueError("updater_checkpoint must be a nonempty path or null")
        if self.mode not in ("virtual", "threaded"):
            raise ValueError("dual.mode must be 'virtual' or 'threaded'")
        _positive("dual.brain_ms", self.brain_ms)
        _positive("dual.action_ms", self.action_ms)
        if self.mode == "virtual" and self.brain_ms < self.action_ms:
            raise ValueError("virtual clock requires brain_ms >= action_ms")
        _integer("dual.max_age", self.max_age, 1)
        _integer("dual.updater_min_age", self.updater_min_age, 1, self.max_age)


@dataclass(frozen=True)
class InferenceConfig:
    # Current defaults match the canonical native controller. Historical runs
    # retain their own frozen source and explicit manifests.
    backend: str = "stock"
    compile_mode: str | None = "max-autotune-no-cudagraphs"

    def __post_init__(self):
        if self.backend not in ("compact", "stock"):
            raise ValueError("inference.backend must be compact or stock")
        if self.compile_mode not in (None, "default", "max-autotune-no-cudagraphs"):
            raise ValueError("Unsupported compile mode; CUDA graph lifetimes are not validated")
        if self.backend == "compact" and self.compile_mode is not None:
            raise ValueError("Compilation requires the stock-preserving inference adapter")


@dataclass(frozen=True)
class SystemConfig:
    robot_profile: str = "libero"
    base_checkpoint: str = "models/pi05_libero_pytorch"
    num_steps: int = 1
    replan_steps: int = 5
    dual: DualConfig = field(default_factory=DualConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    updater: UpdaterConfig = field(default_factory=UpdaterConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.base_checkpoint, str) or not self.base_checkpoint.strip():
            raise ValueError("base_checkpoint must be a nonempty path")
        if self.robot_profile not in {"libero", "fr3"}:
            raise ValueError("robot_profile must be libero or fr3")
        expected_state = 8 if self.robot_profile == "libero" else 7
        if self.updater.state_dim != expected_state:
            raise ValueError("updater state dimension differs from robot profile")
        if self.robot_profile == "fr3" and self.updater.text_tokens != 200:
            raise ValueError("FR3 preserves all 200 state-conditioned text slots")
        _integer("num_steps", self.num_steps, 1)
        _integer("replan_steps", self.replan_steps, 1, 10)
        if not isinstance(self.dual, DualConfig) or not isinstance(self.cache, CacheConfig) or not isinstance(self.updater, UpdaterConfig):
            raise TypeError("dual, cache and updater must be their typed configurations")
        if not isinstance(self.inference, InferenceConfig):
            raise TypeError("inference must be InferenceConfig")
        if self.dual.max_age > self.updater.history:
            raise ValueError("dual.max_age cannot exceed updater.history")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_T = TypeVar("_T")


def _parse(cls: type[_T], raw: dict[str, Any]) -> _T:
    if not isinstance(raw, dict):
        raise TypeError(f"{cls.__name__} must be a JSON object")
    unknown = set(raw) - {item.name for item in fields(cls)}
    if unknown:
        raise ValueError(f"unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**raw)


def config_from_dict(raw: dict[str, Any]) -> SystemConfig:
    if not isinstance(raw, dict):
        raise TypeError("system configuration must be a JSON object")
    raw = dict(raw)
    for name, cls in (("dual", DualConfig), ("cache", CacheConfig), ("updater", UpdaterConfig), ("inference", InferenceConfig)):
        if name in raw:
            raw[name] = _parse(cls, raw[name])
    return _parse(SystemConfig, raw)


def load_config(path: str | Path) -> SystemConfig:
    def reject_constant(value: str) -> None:
        raise ValueError(f"nonfinite JSON constant {value}")

    return config_from_dict(json.loads(Path(path).read_text(), parse_constant=reject_constant))
