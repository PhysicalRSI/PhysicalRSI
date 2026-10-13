"""Protect historical observed capability scores under a frozen task protocol.

Floors come from audited lineage, including retained-parent assessments,
never from a caller's claimed leaderboard.
They are empirical retention constraints, not population confidence bounds.
Changed task meanings/scales require a new declared research policy.
"""
from copy import deepcopy
from pathlib import Path

from ..infra.storage import digest, file_digest, read_json
from . import selection


class RetentionSelector:
    def __init__(self, *, state, parent_revision, maximum_regression):
        selection._sha(parent_revision)
        if (not isinstance(maximum_regression, dict) or not maximum_regression
                or any(not 0 <= selection._number(v) <= 1 for v in maximum_regression.values())):
            raise ValueError("Declare per-capability historical regression limits")
        self.state, self.parent_revision = state, parent_revision
        self.limits = deepcopy(maximum_regression)
        self.view = self._view()

    def policy_identity(self):
        return dict(kind="physicalrsi.retention-selector/v1", state_root=str(self.state.root),
            maximum_regression=deepcopy(self.limits), implementation=file_digest(Path(__file__)),
            scoring=file_digest(Path(selection.__file__)),
            lineage=file_digest(Path(__file__).parent.parent / "lineage/__init__.py"),
            artifacts=file_digest(Path(__file__).with_name("artifacts.py")))

    def _view(self):
        high_water = {task: 0. for task in self.limits}
        records, current = [], self.parent_revision
        # ResearchCampaign constructs a declaration-only template before it
        # bootstraps lineage. That template cannot perform selection.
        if current == "0" * 64:
            return dict(parent_revision=current, records=[], high_water=high_water)
        scope, policy = None, None
        for row in self.state.history(current):
            if scope is None:
                scope, policy = row["scope"], row["policy"]
            if row["scope"] != scope or row["policy"] != policy:
                raise ValueError("Retention history crossed its scoring policy or scope")
            if set(row["policy"]["profile"]["tasks"]) != set(self.limits):
                raise ValueError("Retention history changed its capability coverage")
            decision = row.get("decision")
            if decision is not None:
                for name in {decision["parent_id"], decision["survivor_id"]}:
                    scores = decision["metrics"][name]["per_task"]
                    if set(scores) != set(high_water):
                        raise ValueError("Historical capability coverage changed")
                    for task, score in scores.items():
                        if not 0 <= selection._number(score) <= 1:
                            raise ValueError("Invalid historical normalized score")
                        high_water[task] = max(high_water[task], score)
            records.append(row["revision"])
        return dict(parent_revision=self.parent_revision, records=records, high_water=high_water)

    def identity(self):
        if self._view() != self.view:
            raise ValueError("Frozen retention history changed")
        return dict(self.policy_identity(), view=deepcopy(self.view))

    def select(self, comparison, cohort, results, *, evidence_root):
        self.identity()
        if self.parent_revision == "0" * 64:
            raise ValueError("A declaration-only retention template cannot select")
        anchor = self.state.read(self.parent_revision)
        if (read_json(Path(evidence_root) / "parent.json") != anchor
                or comparison["parent_id"] != anchor["harness"]["id"]
                or comparison["candidates"].get(comparison["parent_id"]) != anchor["freeze_sha256"]
                or comparison["profile"] != anchor["policy"]["profile"]
                or set(comparison["profile"]["tasks"]) != set(self.limits)):
            raise ValueError("Retention comparison differs from its frozen parent or capability protocol")
        result = selection.select_survivor(comparison, cohort, results, evidence_root=evidence_root)
        parent, metrics = comparison["parent_id"], result["metrics"]
        high_water = {t: max(self.view["high_water"][t], metrics[parent]["per_task"][t]) for t in self.limits}
        tolerance = comparison["profile"]["tie_tolerance"]
        violations = {}
        for name, metric in metrics.items():
            if name == parent:
                continue
            lost = [t for t in sorted(self.limits)
                    if metric["per_task"][t] + self.limits[t] + tolerance < high_water[t]]
            if lost:
                violations[name] = lost
                result["rejected"].setdefault(name, dict(reason="historical_capability_regression", tasks=lost))
        eligible = [name for name in metrics if name != parent and name not in result["rejected"]]
        winner = parent
        if eligible:
            best = max(metrics[name]["normalized_score"] for name in eligible)
            winner = min(name for name in eligible if best - metrics[name]["normalized_score"] <= tolerance)
        result.update(survivor_id=winner, survivor_freeze_sha256=comparison["candidates"][winner],
                      decision="retain_parent" if winner == parent else "inherit_child",
                      retention=dict(view_sha256=digest(self.view), parent_revision=self.parent_revision,
                                     high_water=high_water, maximum_regression=self.limits, violations=violations))
        return deepcopy(result)
