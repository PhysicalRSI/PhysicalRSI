# PhysicalRSI-core

Core makes a proposed improvement inspectable: the experiment records its conditions, executes a frozen candidate, asks an independent verifier for an outcome, and preserves the evidence used to select a revision.

## Experiment runtime

`experiments.py` provides `ExperimentRuntime`, `Budget`, and the `Environment`, `Policy`, and `Verifier` ports. An environment verifies reset readiness and reports observations/termination; a policy returns actions; a verifier compares the observations with the declared task requirement.

```python
from PhysicalRSI_demos.experiment import run
receipt = run('/tmp/physicalrsi-experiments', 'trial-1')
print(receipt['outcome'], receipt['evidence'])
```

An experiment directory contains `experiment.json`, `reset.json`, `trajectory.json`, `verdict.json` and `receipt.json`. Adapter identities, budget, case and evidence hashes are bound to the receipt. `success`, `failure`, `uncertain` and `invalid` remain separate. A reset that is not ready cannot become a failed policy trial. An external effect interrupted before completion requires reconciliation; rerunning the same ID never silently repeats it.

New receipts use `physicalrsi.experiment-receipt/v2`; historical v1 receipts remain readable. `embodiment.py` binds declared environment and System 1 observation/action contracts, frozen dependencies and optional System 2 trial attribution. Changing runtime or adapter identity requires a new trial ID. Policies receive the executing revision through `Context.harness_revision` and can implement explicit episode initialization and teardown.

Deadlines are cooperative. Adapters must bound device calls and honor `Context.check()`. Physical declarations require named resources and an explicitly shared `infra.devices.DeviceRegistry`. Its leases coordinate cooperating processes on one local controller host, retain uncertain ownership across interruption, and require confirmed quiescence before release. They do not stop hardware or fence another controller host. The runtime verifies local evidence and identity consistency; it cannot guarantee sensor accuracy, verifier accuracy, physical reproducibility or process isolation.

## Self-Harness

The original preview release's `self_harness/` is retained. Its injected proposal, evaluation and selection ports perform:

1. Development feedback, including failures.
2. Candidate materialization with parent and dependency identities.
3. Admission and a frozen comparison pool/profile.
4. Fresh validation cases and paired parent/candidate evaluation.
5. Selection with regression limits and evidence checks.
6. Compare-and-swap inheritance through `lineage.HarnessState`.

`/evolve` runs the CPU service example end to end. Completed stages resume from receipts. Unknown external effects require reconciliation. The current implementation fixes the foundation component during an iteration; it does not automatically improve foundation-model weights.

The experiment runtime is an adapter building block. Task-specific Self-Harness evaluators decide how to turn measured trials into comparison results. `self_harness/selection.py` checks paired cases, evaluator identity, native outcomes, evidence digests and per-task regression limits. It does not infer physical qualification from passing software checks.

`self_harness.evaluation.ExperimentEvaluator` connects declared suites to development and paired validation trials. The experiment-result bridge checks receipt attribution and matching execution/scoring protocols. `infra.trial_quota.TrialQuota` reserves complete plans before execution, and `self_harness.campaign.ImprovementCampaign` bounds sequential rounds while auditing retained and inherited revisions. Reservations count planned trials, including interrupted attempts; they are not GPU or token accounting. Recovery uses verified receipts and does not automatically repeat unknown external effects.

Opt-in `CaseProtocol` and `PreregisteredSuite` freeze development, validation and report-only test cases. `ShadowGate` checks temporal triggers against verified failure and successful-control trajectories before further evaluation. See the [research protocol guide](../docs/research-protocol.md) for integration and public-observation boundaries.

`self_harness.research_memory.ResearchMemory` stores immutable, scoped research lessons with evidence and counterexamples. `ResearchMemoryGate` and `MemoryBoundSuite` bind required checks to a candidate and its selected parent; passing admission does not select a survivor. The [memory guide](../docs/research-memory-evolution.md) describes suspension and refinement. [Autoresearch records](../docs/autoresearch-records.md) connect these contracts to an external agent's hypothesis, experiment, result and explicit memory disposition. They do not launch jobs or generate research proposals.

The complete CPU campaign and research-record lifecycle can be checked with:

```bash
python -m pytest -q tests/test_improvement_campaign.py tests/test_autoresearch_records.py
```

These checks exercise software integration, restart and evidence integrity. They do not establish native benchmark gains or physical qualification.

`self_harness.research_campaign.ResearchCampaign` adds an opt-in recursive
research workflow: each round consumes a pinned research memory, proposes and
evaluates candidates, then creates a development-evidence-bound memory delta.
A durable round receipt advances the harness and memory references together for
the next proposal, including rounds that retain the parent. Stable campaign
rules are separated from fresh case plans and memory bindings. Profiles can
declare success/cost constraints and conservative paired statistical selection.
Final held-out tests are report-only. See [recursive research](../docs/recursive-research.md)
for provider contracts, recovery boundaries, a two-task CPU example and its
frozen-memory control.

## Other modules

- `contracts.py`: typed, versioned operations and cooperative execution context.
- `infra/`: atomic storage, events, resource pools, process/service lifecycle, RPC and trajectory evidence.
- `lineage/`: current revision, historical evidence and explicit rollback records.

Core imports no CLI, baseline or concrete task implementation. Application adapters provide those dependencies. See [architecture](../docs/architecture.md) for System 1 and System 2.
