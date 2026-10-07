# Research protocol and read-only trigger preflight

This opt-in core integration adapts useful **design ideas** from
[Zetta-Embodiment](https://github.com/air-embodied-brain/Zetta-Embodiment/tree/21d7a3afd6ca5a34de9d0350f0158252c5c55467),
reviewed at commit `21d7a3afd6ca5a34de9d0350f0158252c5c55467`.
The implementation is original physicalRSI code. The inspected upstream
`pyproject.toml` references `LICENSE`, but that file is absent in this checkout;
no Zetta implementation is vendored or installed.

## What transfers to core

| Upstream design | physicalRSI decision |
| --- | --- |
| Preregister development, paired evaluation and held-out cases | Add `CaseProtocol` and `PreregisteredSuite`; freeze case identities before development and keep test access bound to one final selection |
| Shadow replay on failures and successful controls | Add a read-only temporal trigger preflight over verified development receipts; reject late triggers and false positives; mark missing observations inconclusive |
| Separate diagnosis, critic, authority and recovery | Preserve System 2 proposal and System 1 execution boundaries; replay has no device callbacks or arbitrary candidate code |
| Frozen base policy and versioned recovery bundles | Reuse existing candidate closure, admission, immutable memory and lineage; do not build a competing artifact store |
| Separate environment and model workers, bounded queues | Existing controller/service boundaries already cover these roles; do not add Ray or migrate active DSW jobs without measured throughput/memory evidence |
| Privileged simulator features for some critics | Do not import these features into policy/critic inputs; continue using reviewed public observations, calibration and proprioception |

Source references:
[evolution protocol](https://github.com/air-embodied-brain/Zetta-Embodiment/blob/21d7a3afd6ca5a34de9d0350f0158252c5c55467/zetta/evolution/protocol.py),
[shadow replay](https://github.com/air-embodied-brain/Zetta-Embodiment/blob/21d7a3afd6ca5a34de9d0350f0158252c5c55467/zetta/evolution/shadow_replay.py),
[runtime configuration](https://github.com/air-embodied-brain/Zetta-Embodiment/blob/21d7a3afd6ca5a34de9d0350f0158252c5c55467/rollout_runtime/config/schema.py),
[privileged LIBERO backend](https://github.com/air-embodied-brain/Zetta-Embodiment/blob/21d7a3afd6ca5a34de9d0350f0158252c5c55467/rollout_runtime/backends/libero_privileged.py).

## Frozen cases in the existing evaluator

```python
from PhysicalRSI_core.self_harness.protocol import CaseProtocol, PreregisteredSuite

plan = CaseProtocol('/tmp/physicalrsi-protocol', splits={
    'development': {'pick': [{'seed': 101}, {'seed': 102}]},
    'validation': {'pick': [{'seed': 201}, {'seed': 202}]},
    'test': {'pick': [{'seed': 301}, {'seed': 302}]},
})
# provider_suite implements the existing ExperimentSuite ports.
suite = PreregisteredSuite(provider_suite, plan)
# Construct a NEW ExperimentEvaluator(suite=suite, ...), then a new
# SelfHarness workspace; never replace ports after freezing a live round.
```

All tasks must occur in all three splits. Duplicate task-local reset identities
are rejected, even if additional case metadata differs. Set `identity_keys`
explicitly when reset identity is a layout ID or multiple fields instead of a
seed. This checks declared identity, not physical layout novelty. It cannot
detect a provider assigning different identities to the same scene.
The suite binds validation to exactly one frozen comparison, allowing identical
resume. A new round must preregister a new plan with fresh layouts; repeatedly
tuning against this validation cohort is rejected. Cross-workspace novelty
still requires an external layout inventory.

`plan.claim_test(candidate_sha256=..., selection_sha256=...)` durably binds the
test cases to one final selection; an identical resume succeeds and a different
selection fails. The caller must verify that selection, run the test trials,
and report them separately. The adapter never feeds the test split to
development or paired selection. A JSON manifest is not a secret test set;
this API does not prove independence from a privileged operator or historical
experiments in other workspaces.

## Read-only temporal trigger gate

`shadow_replay` consumes `ReplayCase(receipt_path, divergence_index)` records.
Every receipt is verified by `ExperimentRuntime.read`. Trials must be completed
development runs of the declared parent, with a definite success/failure
outcome. Validation/test traces, repeated cases, other parents and changed
evidence are rejected. Failed resets never count as negative policy examples.

`TemporalTrigger(('progress', 'distance_m'), threshold=0.1, consecutive=3)`
reads only that path beneath each trajectory observation. The feature must be
reviewed for public provenance by the adapter owner: an allowlisted field name
does not prove its sensor provenance. Replay never receives evaluator-only
measurements as critic inputs, executes recovery actions, or imports generated
critic source. A rule can use `ge` or `le`; history resets for each trajectory.

The report records the candidate, parent, rule, implementation and source file
digests. The reviewer supplies divergence annotations; row zero is the reset
observation. Passing requires at least one failure and one successful control,
no missing features, detection by every annotated divergence, and no control
triggers. Unknown divergence or missing data is **inconclusive**, not evidence
of an absent trigger. Default replay limits are 16 MB per trajectory and 100,000
total frames. These are CPU preflight limits, not simulator settings.

Freeze the report with `atomic_json`, then configure a trusted
`ShadowGate(parent_sha256=..., reports={path: file_digest(path)})`. Pass it to
`PreregisteredSuite(..., shadow_gate=gate)`. The gate rechecks the report digest,
candidate binding and source evidence during admission. A rejected candidate
does not reach the provider's admission callback. Parent evaluation remains
available. Report digests belong to the trusted experiment configuration;
this is integrity checking, not signing or producer authentication.

A passing trigger gate authorizes **further paired evaluation only**. It says
nothing about recovery effectiveness, semantic identity, grasp attachment,
benchmark-wide success, or physical qualification. The rule must also be bound
to the candidate's actual execution artifact by its provider. No existing
native benchmark campaign opts into these modules automatically.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  tests/test_research_protocol.py tests/test_research_heartbeat.py
```

Tests run real CPU `ExperimentRuntime` receipts through replay and admission,
exercise frozen split enforcement, missing/late/false triggers, tampering,
replay budgets, and a complete existing Self-Harness selection with the new
suite adapter. They do not establish simulator performance.
