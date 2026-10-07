"""Read-only temporal trigger evaluation over verified development receipts.

Only explicitly selected observation fields enter the declarative critic.
No candidate Python, recovery actions, environment, or device calls run here.
Observation provenance still belongs to the reviewed adapter.
"""

import math
from dataclasses import asdict, dataclass
from pathlib import Path

from ..experiments import ExperimentRuntime
from ..infra.storage import digest, file_digest, read_json
from .artifacts import verify_harness
from .protocol import _sha


@dataclass(frozen=True)
class TemporalTrigger:
    feature_path: tuple[str, ...]
    threshold: float
    consecutive: int = 2
    operator: str = "ge"

    def __post_init__(self):
        object.__setattr__(self, "feature_path", tuple(self.feature_path))
        if not self.feature_path or any(not isinstance(k, str) or not k for k in self.feature_path):
            raise ValueError("Declare a reviewed observation feature path")
        if (type(self.threshold) not in (int, float) or not math.isfinite(self.threshold)
                or type(self.consecutive) is not int or self.consecutive < 1
                or self.operator not in {"ge", "le"}):
            raise ValueError("Invalid temporal trigger")

    def replay(self, observations):
        streak, first, missing = 0, None, []
        for index, observation in enumerate(observations):
            value = observation
            for key in self.feature_path:
                value = value.get(key) if isinstance(value, dict) else None
            if type(value) not in (int, float) or not math.isfinite(value):
                missing.append(index)
                streak = 0
                continue
            matches = value >= self.threshold if self.operator == "ge" else value <= self.threshold
            streak = streak + 1 if matches else 0
            if first is None and streak >= self.consecutive:
                first = index
        return dict(first_trigger_index=first, unavailable_indices=missing)


@dataclass(frozen=True)
class ReplayCase:
    receipt: Path
    divergence_index: int | None = None


def shadow_replay(*, parent_sha256, candidate_sha256, trigger, cases,
                  max_trace_bytes=16_000_000, max_total_frames=100_000):
    """Require timely failure detection and zero triggers on success controls.

    Indices address trajectory rows, with reset observation at index zero.
    Divergence annotations are reviewer assertions, preserved in the report.
    Missing features or missing failure annotations make replay inconclusive.
    A passing result admits only further evaluation, never policy promotion.
    """
    _sha(parent_sha256)
    _sha(candidate_sha256)
    if not isinstance(trigger, TemporalTrigger):
        raise ValueError("Use a declarative TemporalTrigger")
    if any(type(n) is not int or n < 1 for n in (max_trace_bytes, max_total_frames)):
        raise ValueError("Declare positive replay budgets")
    sources, rows, seen, frames = {}, [], set(), 0
    for case in cases:
        path = Path(case.receipt).resolve()
        if path.name != "receipt.json":
            raise ValueError("Expected an ExperimentRuntime receipt")
        trajectory = path.with_name("trajectory.json")
        if trajectory.stat().st_size > max_trace_bytes:
            raise ValueError("Shadow trajectory exceeds byte budget")
        runtime = ExperimentRuntime(path.parent.parent)
        receipt = runtime.read(path.parent.name)
        attribution = receipt.get("system2") or {}
        if (receipt.get("state") != "completed" or receipt.get("outcome") not in {"success", "failure"}
                or attribution.get("split") != "development"
                or attribution.get("candidate_sha256") != parent_sha256):
            raise ValueError("Shadow replay needs valid development trials of the frozen parent")
        specification = read_json(path.with_name("experiment.json"))
        identity = digest([specification["task"], specification["case"]])
        if identity in seen:
            raise ValueError("Repeated development case in replay corpus")
        seen.add(identity)
        trace = read_json(trajectory)
        frames += len(trace)
        if not trace or frames > max_total_frames:
            raise ValueError("Empty trace or total frame budget exceeded")
        divergence = case.divergence_index
        if divergence is not None and (type(divergence) is not int or not 0 <= divergence < len(trace)):
            raise ValueError("Divergence index outside the recorded trajectory")
        if receipt["outcome"] == "success" and divergence is not None:
            raise ValueError("Successful controls must not carry failure annotations")
        outcome = trigger.replay([event["observation"] for event in trace])
        first = outcome["first_trigger_index"]
        rows.append(dict(receipt=str(path), case_sha256=identity,
                         role="failure" if receipt["outcome"] == "failure" else "success_control",
                         divergence_index=divergence, **outcome,
                         timely=divergence is not None and first is not None and first <= divergence))
        # Bind every file verified by ExperimentRuntime, not just extracted features.
        sources[str(path)] = file_digest(path)
        for name, sha in receipt["evidence"].items():
            sources[str(path.parent / name)] = sha
    failures = [x for x in rows if x["role"] == "failure"]
    controls = [x for x in rows if x["role"] == "success_control"]
    conclusive = bool(failures and controls) and all(
        not x["unavailable_indices"] and (x["role"] != "failure" or x["divergence_index"] is not None)
        for x in rows)
    false_positives = sum(x["first_trigger_index"] is not None for x in controls)
    timely = sum(x["timely"] for x in failures)
    passed = conclusive and timely == len(failures) and false_positives == 0
    return dict(schema="physicalrsi.shadow-replay/v1", parent_sha256=parent_sha256,
                candidate_sha256=candidate_sha256, trigger=asdict(trigger),
                trigger_sha256=digest(asdict(trigger)),
                implementation_sha256=file_digest(Path(__file__)), sources=sources, rows=rows,
                limits=dict(max_trace_bytes=max_trace_bytes, max_total_frames=max_total_frames),
                failure_count=len(failures), success_control_count=len(controls),
                timely_detections=timely, false_positives=false_positives,
                conclusive=conclusive, passed=passed,
                scope="read-only trigger preflight; recovery effectiveness untested",
                qualification=None)


class ShadowGate:
    """Bind admission to externally frozen report digests and source evidence.

    The report map belongs to the trusted experiment configuration, never to
    the candidate. This checks local integrity, not producer authenticity.
    """

    def __init__(self, *, parent_sha256, reports):
        self.parent_sha256 = _sha(parent_sha256)
        self._reports = {str(Path(p).resolve()): _sha(sha) for p, sha in reports.items()}

    def identity(self):
        for name, sha in self._reports.items():
            if file_digest(Path(name)) != sha:
                raise ValueError("Frozen shadow report changed")
        return dict(kind="shadow-admission/v1", parent_sha256=self.parent_sha256,
                    reports=dict(self._reports), implementation=file_digest(Path(__file__)))

    def check(self, candidate):
        self.identity()
        sha = verify_harness(candidate)
        if sha == self.parent_sha256:
            return dict(accepted=True, evidence={"role": "unchanged_parent"})
        matched = []
        for name in self._reports:
            report = read_json(name)
            if report.get("candidate_sha256") == sha:
                matched.append((name, report))
        if len(matched) != 1:
            return dict(accepted=False, reason="Exactly one bound shadow report is required")
        name, report = matched[0]
        if (report.get("schema") != "physicalrsi.shadow-replay/v1"
                or report.get("parent_sha256") != self.parent_sha256
                or report.get("conclusive") is not True or report.get("passed") is not True):
            return dict(accepted=False, reason="Shadow replay is inconclusive or rejected")
        for source, expected in report["sources"].items():
            if file_digest(Path(source)) != expected:
                raise ValueError("Shadow source evidence changed")
        return dict(accepted=True, evidence=dict(shadow_report=name,
                    shadow_report_sha256=self._reports[name], qualification=None,
                    scope="Trigger preflight only; paired native evaluation still required"))
