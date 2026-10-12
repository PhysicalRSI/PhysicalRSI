# Recursive research in core

`ResearchCampaign` adds an opt-in research lifecycle around the existing
Self-Harness. Both the selected execution harness and the research memory have
explicit revisions. A retained harness can still produce a new memory revision;
the next proposer receives that revision and its scoped lessons.

This implements feedback reuse. It does not demonstrate that a model improves
its own research ability, modify foundation weights, or qualify physical tasks.

## Fixed rules and changing round bindings

`SelfHarness(..., round_context=...)` uses `physicalrsi.self-harness/v2` and
derives a `physicalrsi.campaign-policy/v2` from the scoring profile, protocol
semantics, scope, candidate budget and implementation identities. Providers may
implement `policy_identity()` to distinguish stable rules from per-round data.
Without it, the provider's entire identity stays fixed.

The built-in evaluator and suite wrappers distinguish immutable case plans,
memory revisions, selected parents and admission reports from implementation
rules. Full provider identities are still frozen and checked within each round.
Lineage binds the complete round configuration and rechecks selection evidence.
Changing a verifier implementation or scoring rule requires a new campaign.

`CaseProtocol` plans remain immutable. Each round gets a new validation plan;
the state store registers semantic reset identities before development starts.
Development may reuse development cases and test may reuse test cases. Validation
cannot reuse any case from another round, and a case cannot move between splits.
The final test manifest is part of the stable campaign policy and cannot change
between rounds. A `MemoryBoundSuite` outside the preregistered suite also checks
that its memory revision, parent and scope match the context given to the proposer.
Reset identity keys must identify actual reset conditions; providers still own
sampling validity and the completeness of those declarations.

## Provider integration

Use `ResearchCampaign` from `PhysicalRSI_core.self_harness.research_campaign`.
Its `build_loop(directory, context)` factory returns a v2 `SelfHarness` with the
supplied context. Factories may create immutable plan files, but must not call
models, execute trials, or cause device effects. A declaration-only template is
constructed before the first round to establish the stable policy.

The proposer implements:

```python
def propose_with_context(parent, feedback, research_context, output):
    # Return materialized candidate manifests with normal provenance plus:
    # used_lessons, hypothesis, expected_effect, required_checks.
    ...
```

The context contains the selected parent revision, memory revision, scoped
lessons, round index, preceding round revision and remaining round budget. Used
lesson IDs must come from the pinned retrieval. Existing development and
explicit recovery methods retain their roles. The optional proposal recovery
method is `resume_proposal_with_context`, with the same arguments.

The reflector implements `identity()` and
`reflect(research_context, development, output)`. It returns:

```json
{
  "schema": "physicalrsi.memory-delta/v1",
  "operations": []
}
```

An empty delta retains memory. Supported operations are `add`, `counterexample`
and `refine`, each naming a `lesson_id`. Add carries a complete `lesson` with
conditions, hypothesis, required checks, counterexample checks and evidence.
Counterexample carries `check`, `explanation` and `evidence`. Refine carries
`conditions`, `hypothesis` and `evidence`.

All new evidence must be in the campaign's verified development allowlist.
Existing lessons cannot be overwritten, counterexamples cannot be erased and
refinement cannot broaden scope. Reflection receives no validation outcomes or
test results. These are trusted Python ports, not sandboxed adversaries; an
external provider must enforce any stronger filesystem or process isolation.

## Durable rounds and recovery

Research stages live outside the immutable execution directory. A round receipt
binds parent/result harness revisions, parent/result memory revisions, context,
reflection, outcome and a complete file manifest. A content-addressed research
cursor advances only after the complete receipt exists.

Harness CAS may complete before reflection. After interruption, only that exact
pending round can explain an advanced harness. Completed trials and reflection
receipts are reused. An incomplete reflection requires an explicit
`resume_reflection` method or external reconciliation; it never silently repeats
an unknown model call. Unknown physical effects retain the existing runtime's
reconciliation behavior. Mutable memory, quota and device stores belong outside
round evidence.

Normal rejection, no candidate, retained parent and inherited child all permit
development reflection. Budget exhaustion leaves the pending round inspectable
and reports `budget_exhausted`; it does not invent a completed round.

## Selection constraints and uncertainty

Existing profiles keep their deterministic behavior. Tasks may additionally
declare `minimum_success_rate`, `maximum_success_regression`, and `cost_limits`
(measurement name to maximum mean cost). Costs must be nonnegative measured
values supplied by the verifier. Missing measurements invalidate evaluation.
These are selection constraints, not token or GPU resource reservations.

For bounded statistical selection, add:

```json
"statistics": {
  "method": "paired-hoeffding",
  "alpha": 0.05,
  "max_comparisons": 30
}
```

The selector uses bounded normalized paired differences and a weighted Hoeffding
lower bound, allocating `alpha / max_comparisons` to each comparison. Campaigns
require `max_rounds * max_candidates <= max_comparisons`. This is conservative
and assumes independent sampled cases, including across tasks, with fresh
validation samples for adaptive rounds. Identity hashes do not establish those
statistical assumptions. The reported lower bound concerns the declared weighted
score, not physical qualification or confidence bounds for every constraint.
Insufficient evidence retains the parent with an explicit reason. Standalone
selector users own cross-comparison budget accounting.

## CPU validation and a frozen-memory control

```bash
PYTHONDONTWRITEBYTECODE=1 python - <<'PY'
from tempfile import TemporaryDirectory
from PhysicalRSI_demos.recursive_research import run
with TemporaryDirectory(prefix='physicalrsi-recursive-') as workspace:
    result = run(workspace)
    print([r['outcome'] for r in result['rounds']])
    print(result['final_test']['metrics'])
    print(result['quota'])
PY

PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  tests/test_recursive_research.py tests/test_statistical_selection.py
```

The two task adapters use numeric increment actions and structured text append
actions. Under equal 30-trial research budgets, the scripted updating-memory arm
retains, inherits, then retains; its frozen-memory control retains three times.
The control still computes and validates reflection deltas but does not adopt
them. This demonstrates the memory-to-proposal dependency, not a general research
algorithm advantage. A scientific comparison needs suitable task distributions,
multiple runs, costs and uncertainty.

`campaign.report_test(initial)` completes/audits the campaign and evaluates its
final candidate on the preregistered test cases. The final claim and report live
outside archived rounds, consume trial quota and resume verified receipts. They
never update the selected harness or research memory. Invalid and uncertain
outcomes remain visible and all planned cases remain in the denominator. The
demo reserves two extra trials for this report, for 32 trials in total.

## Compatibility

Omitting `round_context` retains the original Self-Harness configuration shape.
`ImprovementCampaign` remains available for existing factories and fixed-port
workflows. Historical lineage formats are read without reinterpretation. New
campaigns use distinct schemas and workspaces; immutable historical experiments
are not migrated or rewritten. As before, executable identity changes require
a fresh experiment workspace rather than resuming under changed code.
