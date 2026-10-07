"""Timing declarations and observation-bound action chunks.

Sensor timestamps belong to the controller's monotonic clock. Clients echo a
reference, never translate their own wall clock into a controller deadline.
"""

import math
from copy import deepcopy
from dataclasses import dataclass

from .infra.storage import digest


def finite_seconds(value, *, positive=True):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or (value <= 0 if positive else value < 0)):
        raise ValueError("Expected finite positive seconds" if positive else "Expected finite nonnegative seconds")
    return value


@dataclass(frozen=True)
class ControlTiming:
    period_seconds: float
    max_chunk_steps: int
    max_observation_age_seconds: float
    execution_slack_seconds: float = 0.25

    def __post_init__(self):
        finite_seconds(self.period_seconds)
        finite_seconds(self.max_observation_age_seconds)
        finite_seconds(self.execution_slack_seconds, positive=False)
        if type(self.max_chunk_steps) is not int or self.max_chunk_steps < 1:
            raise ValueError("max_chunk_steps must be a positive integer")
        finite_seconds(self.period_seconds * self.max_chunk_steps)


@dataclass(frozen=True)
class SensorSample:
    """A driver-owned source sequence and capture time, never a receipt time.

    captured_at is the earliest possible capture in the gateway clock, and
    capture_uncertainty_seconds bounds the remaining interval. Exact local
    samples have zero uncertainty. Drivers preserve conversion evidence and
    reject unsynchronized sources. Simulation time belongs in payload data.
    """
    payload: object
    sequence: int
    captured_at: float
    capture_uncertainty_seconds: float = 0
    timing_evidence: dict | None = None

    def __post_init__(self):
        if type(self.sequence) is not int or self.sequence < 0:
            raise ValueError("Sensor sequence must be a nonnegative integer")
        finite_seconds(self.captured_at, positive=False)
        finite_seconds(self.capture_uncertainty_seconds, positive=False)
        finite_seconds(self.captured_at + self.capture_uncertainty_seconds, positive=False)
        if self.timing_evidence is not None and not isinstance(self.timing_evidence, dict):
            raise ValueError("Sensor timing evidence must be an object")


def observation_reference(observation):
    if not isinstance(observation, dict) or observation.get("schema") != "physicalrsi.observation/v1":
        raise ValueError("Expected a controller observation packet")
    return {key: observation[key] for key in ("episode", "clock_domain", "sequence")} | {
        "sha256": digest(observation)}


def action_chunk(observation, actions, *, period_seconds):
    """Bind a policy's chunk to the exact input observation it actually used."""
    finite_seconds(period_seconds)
    if not isinstance(actions, (list, tuple)) or not actions:
        raise ValueError("An action chunk must contain at least one action")
    return dict(schema="physicalrsi.action-chunk/v1",
                observation=observation_reference(observation),
                period_seconds=period_seconds, actions=deepcopy(list(actions)))
