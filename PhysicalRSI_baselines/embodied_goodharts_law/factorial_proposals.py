"""Parent-bound component ablations for reviewed EGL hypotheses.

This supplies search alternatives to StructuredProposer, not an autonomous
repair model. Missing component dependencies may make an ablation invalid;
normal candidate admission and evaluation must retain that outcome.
"""
from copy import deepcopy
from itertools import combinations
from pathlib import Path

from PhysicalRSI_core.infra.storage import canonical, file_digest
from .candidate_program import PRIMITIVES, MEMORY, COMBINATIONS, edit_contract


class FactorialEdits:
    """Compare all nonempty subsets of one evidence-backed joint hypothesis.

    A hypothesis is anchored to exact input revisions. After inheritance the
    caller must formulate a new hypothesis rather than replay stale edits.
    Evidence hashes identify observations, not proof of improvement.
    """

    def __init__(self, *, inputs, replacements, evidence, rationale):
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError("A hypothesis rationale is required")
        if not evidence or not isinstance(evidence, dict):
            raise ValueError("Declare evidence paths and SHA256 digests")
        self.evidence = deepcopy(evidence)
        self._verify_evidence()
        if set(inputs) != {PRIMITIVES, MEMORY, COMBINATIONS}:
            raise ValueError("All three parent component inputs are required")
        proposal = dict(rationale=rationale, edits=[
            dict(path=name, before_sha256=inputs[name]['sha256'], value=value)
            for name, value in replacements.items() if name in inputs])
        if not replacements or not set(replacements) <= set(inputs):
            raise ValueError("Declare replacements within the EGL edit domain")
        changed = edit_contract().replacements(proposal, inputs)
        if not changed:
            raise ValueError("Hypothesis does not change the parent")
        self.inputs = deepcopy(inputs)
        self.replacements = {name: deepcopy(replacements[name]) for name in changed}
        self.rationale = rationale

    def _verify_evidence(self):
        for path, expected in self.evidence.items():
            if file_digest(Path(path)) != expected:
                raise ValueError("Hypothesis evidence changed")

    def identity(self):
        return dict(kind="egl-factorial-edits/v1", inputs=deepcopy(self.inputs),
                    replacements=deepcopy(self.replacements), evidence=deepcopy(self.evidence),
                    rationale=self.rationale, implementation=file_digest(Path(__file__)))

    def generate(self, request, limits):
        self._verify_evidence()
        if request['editable'] != self.inputs:
            raise ValueError("Hypothesis belongs to a different parent; propose from new evidence")
        names = [name for name in (PRIMITIVES, MEMORY, COMBINATIONS)
                 if name in self.replacements]
        count = 2 ** len(names) - 1
        if count > limits.max_candidates:
            raise ValueError("Candidate allowance must cover every component ablation")
        candidates = []
        for size in range(1, len(names) + 1):
            for subset in combinations(names, size):
                candidates.append(dict(
                    rationale=self.rationale + "; changed components: " + ", ".join(subset)
                    + "; evidence SHA256: " + ", ".join(sorted(self.evidence.values())),
                    edits=[dict(path=name, before_sha256=self.inputs[name]['sha256'],
                                value=deepcopy(self.replacements[name])) for name in subset]))
        return dict(text=canonical(dict(parent_sha256=request['parent_sha256'],
                                        candidates=candidates)).decode(),
                    usage=None, provider=None, tool_calls=[])
