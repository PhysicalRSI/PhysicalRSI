"""Explicit sensor, controller and evaluator boundaries.

Judgment text is deliberately absent from every execution-facing type.
"""
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Protocol

CAMERAS = ("front", "left", "right")


def json_copy(value):
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def digest(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def vector(value, n):
    if (not isinstance(value, (list, tuple)) or len(value) != n
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in value)):
        raise ValueError(f"Expected {n} finite numbers")
    return list(value)


class Level(str, Enum):
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"


class Outcome(str, Enum):
    ONGOING = "ongoing"
    SUCCESS = "success"
    FAILURE = "failure"
    UNCERTAIN = "uncertain"


class Progress(str, Enum):
    IMPROVED = "improved"
    UNCHANGED = "unchanged"
    REGRESSED = "regressed"
    UNKNOWN = "unknown"


class CapabilityUnavailable(RuntimeError):
    """No action was attempted; another controller may try the task."""


class ActionRejected(CapabilityUnavailable):
    """A specific request was rejected before dispatch; reobserve and correct it."""


class ExecutionUncertain(RuntimeError):
    """An action might have happened; stop and reconcile, never auto-fallback."""


class ObservationTimingError(ValueError):
    """Real sensor timestamps do not form a fresh synchronized observation."""


class SensorUnavailable(RuntimeError):
    """Fresh RGB is missing; this is not a task judgment or an actuator failure.

    A port may attach the confirmed receipt of the SAME dispatched action. It
    must never attach a guessed result or resend an action to establish it.
    """
    def __init__(self, message, *, completed_action=None):
        super().__init__(message)
        self.completed_action = completed_action


class HumanDecisionPending(RuntimeError):
    """Human assessment is unavailable/uncertain; it is not a failed policy."""


class TaskFinished(BaseException):
    """Independent monitor ended the task; escape generated policy handlers."""


class TargetLost(BaseException):
    """Observed drop and absence in all views; end without an automatic retry."""


@dataclass(frozen=True)
class Frame:
    camera: str
    path: str
    captured_ns: int
    sha256: str

    def validate(self):
        p = Path(self.path)
        if self.camera not in CAMERAS or type(self.captured_ns) is not int or self.captured_ns <= 0:
            raise ValueError("Invalid camera identity or acquisition timestamp")
        if p.is_symlink() or not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != self.sha256:
            raise ValueError("Camera evidence is missing or changed")

    def public(self):
        self.validate()
        return dict(camera=self.camera, path=self.path, captured_ns=self.captured_ns, sha256=self.sha256)


@dataclass(frozen=True)
class Observation:
    id: str
    captured_ns: int
    frames: tuple[Frame, ...]
    qpos26: tuple[float, ...]
    # Measured robot-only FK, never a hidden object pose or task success signal.
    eef: dict

    def validate(self, *, live=False, max_age_s=2.0, max_skew_s=0.2):
        if not isinstance(self.id, str) or not self.id or type(self.captured_ns) is not int:
            raise ValueError("Observation identity and timestamp required")
        if len(self.frames) != 3 or {f.camera for f in self.frames} != set(CAMERAS):
            raise ValueError("Exactly three distinct camera views are required")
        vector(self.qpos26, 26)
        for i in (*range(7, 13), *range(20, 26)):
            if not 0 <= self.qpos26[i] <= 1:
                raise ValueError("Measured hand coordinates must be normalized per motor")
        if set(self.eef) != {"left", "right"}:
            raise ValueError("Both measured EEF poses are required")
        for pose in self.eef.values():
            if set(pose) != {"position", "quaternion_wxyz"}:
                raise ValueError("Only measured robot EEF geometry belongs in this field")
            vector(pose["position"], 3)
            q = vector(pose["quaternion_wxyz"], 4)
            if abs(sum(v*v for v in q) - 1) > 2e-4:
                raise ValueError("EEF quaternion must be unit wxyz")
        for f in self.frames:
            f.validate()
            if abs(f.captured_ns - self.captured_ns) > max_skew_s * 1e9:
                raise ObservationTimingError("Camera views are not synchronized with the observation")
        if live and not 0 <= time.time_ns() - self.captured_ns <= max_age_s * 1e9:
            raise ObservationTimingError("Observation is stale or from the future")

    def public(self):
        self.validate()
        return dict(id=self.id, captured_ns=self.captured_ns,
                    frames=[f.public() for f in self.frames],
                    qpos26=list(self.qpos26), eef=json_copy(self.eef))

    @classmethod
    def from_public(cls, value):
        if set(value) != {"id", "captured_ns", "frames", "qpos26", "eef"}:
            raise ValueError("Unexpected fields in public sensor observation")
        result = cls(value["id"], value["captured_ns"],
                     tuple(Frame(**f) for f in value["frames"]),
                     tuple(value["qpos26"]), json_copy(value["eef"]))
        result.validate()
        return result


@dataclass(frozen=True)
class Assessment:
    observation_id: str
    outcome: Outcome
    progress: Progress
    evidence: str
    source: str = "monitor"
    target_lost: bool = False

    def __post_init__(self):
        if (not self.observation_id or not isinstance(self.outcome, Outcome)
                or not isinstance(self.progress, Progress) or not self.evidence.strip()
                or self.source not in {"monitor", "human"} or type(self.target_lost) is not bool
                or self.target_lost and self.outcome == Outcome.SUCCESS):
            raise ValueError("Invalid independent assessment")

    def private(self):
        return dict(observation_id=self.observation_id, outcome=self.outcome.value,
                    progress=self.progress.value, evidence=self.evidence, source=self.source,
                    target_lost=self.target_lost)


@dataclass(frozen=True)
class Task:
    id: str
    instruction: str
    success_criteria: str

    def public(self):
        # The declared criterion is public. A judge's answer never is.
        if not all(isinstance(v, str) and v.strip() for v in (self.id, self.instruction, self.success_criteria)):
            raise ValueError("Task identity, instruction and declared success criteria required")
        return dict(id=self.id, instruction=self.instruction, success_criteria=self.success_criteria)


class RobotPort(Protocol):
    """Host-owned physical capabilities. Generated source never owns this object."""
    def observe(self) -> Observation: ...
    def scene(self, names: list[str]) -> dict: ...
    def approach(self, arm: str, position_m: list, quaternion_wxyz: list, *, checkpoint) -> dict: ...
    def gripper(self, arm: str, action: str, mode=None, *, checkpoint, observation=None) -> dict: ...
    def act_eef(self, targets: dict, steps: int, *, checkpoint, observation=None) -> dict: ...
    def hold(self) -> None: ...
    def release_for_human(self) -> None: ...
    def stop(self) -> None: ...


class Model(Protocol):
    def complete(self, history: list, tools: list): ...
    def tool_result(self, call_id: str, value: dict) -> dict: ...
