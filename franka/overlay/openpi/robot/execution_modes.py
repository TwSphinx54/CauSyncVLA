"""Nominal host-side rates. These do not change the 1 kHz Franka FCI loop."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionProfile:
    dispatch_hz: int
    feedback_hz: int
    latest: bool
    protocol: str

    @property
    def node_ms(self):
        return 1000 / self.feedback_hz

    @property
    def sample_period_ns(self):
        return 0 if self.latest else int(1e9 / self.feedback_hz)

    @property
    def reply_within_servo_budget(self):
        return self.latest and self.dispatch_hz >= 200

    @property
    def capability(self):
        return f"accelerated_feedback{self.feedback_hz}" if self.latest else "compatible_feedback10"


PROFILES = {
    "accelerated100": ExecutionProfile(100, 100, True, "latest_feedback100_gripper_v3"),
    "accelerated200": ExecutionProfile(200, 200, True, "latest_feedback200_gripper_v2"),
}
EXECUTION_MODES = tuple(PROFILES)


def execution_profile(mode):
    try:
        return PROFILES[mode]
    except (KeyError, TypeError):
        raise ValueError(f"Unknown execution mode: {mode!r}") from None
