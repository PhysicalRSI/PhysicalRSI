# PhysicalRSI-core

Core makes a proposed improvement inspectable: the experiment records its conditions, executes a frozen candidate, asks an independent verifier for an outcome, and preserves the evidence used to select a revision.

The [core development guide](../docs/core-development.md) records P0–P2 priorities,
acceptance criteria and remaining research boundaries. New lineage records own
content-addressed file snapshots; `EvolutionSpec` freezes component mutation
boundaries. `trials.Trial` and `trials.Evaluation` share the runtime's verified
rollout contract, and `RolloutProposer` supplies complete development trajectories
to a replaceable evolver. The guide includes a two-task CPU calibration example,
the optional Isaac native research path, and recovery checks for damaged active
versions. Environments can export declared recordings into each trial's verified
artifact closure; core retains the files without depending on a simulator.
`HarnessState.restore_snapshot` explicitly recovers an owned snapshot from an
independent replica with the original hashes. It journals the operation and
quarantines damaged bytes without changing the lineage head, so later retention
checks can audit the repaired ancestor. Ordinary capture still refuses corruption.
`HarnessState.history(revision)` audits the full ancestry in one read. Repeated
snapshot and evidence checks are shared within that call; subsequent reads
verify bytes again. Candidate archive reads use the same bounded reuse and parse
historical search views from the exact bytes checked against their stored hashes.

## Experiment runtime

`experiments.py` provides `ExperimentRuntime`, `Budget`, and the `Environment`, `Policy`, and `Verifier` ports. An environment verifies reset readiness and reports observations/termination; a policy returns actions; a verifier compares the observations with the declared task requirement.

Stateful suites can implement `trial(candidate, task)` and return `TrialAdapters`
containing one environment, policy and verifier bound together. Each concurrent
call must construct independent adapters without starting workers or resetting
devices. `ExperimentEvaluator` and the preregistration/memory wrappers preserve
this factory; existing separate factories remain supported. The evaluator checks
the candidate and factory identity again before execution. This avoids a shared
"current environment" variable binding one trial's policy to another trial.

`runtime_implementation()` declares the owned execution sources once for both
Trial receipts and `ExperimentEvaluator` identities, including evidence readers,
storage, devices and contracts. Changing one of those files invalidates the
frozen evaluator before new execution. Raw historical evidence remains readable;
resuming a campaign requires its original implementation or a new campaign.

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

Every completed paired comparison advances lineage. A `retain_parent` decision
creates an `assessment` record with the same harness and its verified comparison
evidence; an `inherit_child` decision creates a `selection` record. This preserves
historical capability measurements even when no candidate is promoted. Compare
`freeze_sha256` for artifact changes rather than assuming every new lineage
revision is a new controller. Default rollback skips unchanged assessments when
stepping back a selected generation. Rounds without a complete paired comparison
continue to keep their evidence in campaign/archive receipts.

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

`self_harness.archive.CandidateArchive` retains owned snapshots and verified
development/validation evidence for selected, rejected and losing candidates.
An optional archive freezes each round's search view; a retrieved branch is a
new proposal against the current parent and needs fresh paired validation.
Archive publication resumes after interruption without repeating trials.
See [candidate branches](../docs/core-development.md#candidate-branches-and-search-history)
for the two-task branch-reuse example and storage/recovery boundaries.
`ScheduledEvaluator` runs the paired candidate pool through a durable local
`JobScheduler`. `MultiFidelityEvaluator` screens candidates on development
rollouts before independently scoring the surviving full-validation pool.
By default only children receive screening rollouts. The incumbent is retained
with `measured: false` and `per_task: null`, then measured on the same full
validation cases as surviving children. Set `screen_incumbent=True` for an
additional diagnostic measurement; this choice is frozen in evaluator identity.
Dead worker reservations persist until verified completion or recorded
reconciliation; screening scores never count as promotion scores.
Requests and callback results are owned JSON snapshots. The queue checks handler
identity before and after execution and collection, including completed-job
reads and recovery. Result comparisons use their canonical JSON hashes, so a
shared mutable return object cannot erase disagreement between execution and
verified evidence. Unknown work retains capacity until verified recovery or
recorded quiescence; result collection never authorizes reexecution.
See [evaluation stages](../docs/core-development.md#asynchronous-evaluation-and-fidelity-stages)
for resource boundaries and the CPU screening comparison.

`self_harness.skill_compiler.ExperienceCompiler` turns independently retained
development trials into checked executable skill bundles. `skills.SkillLibrary`
binds them to explicit memory revisions and applicability conditions, and copies
their executable files into candidate harnesses. Definite development failures
can suspend the corresponding lesson; a new skill revision must pass checks and
fresh selection before deployment. See [compiled skills](../docs/core-development.md#compiled-skills-and-versioned-memory)
for the scalar-to-vector transfer, counterexample and rollback example.

`harness_tree.HarnessTree` packages scoped branches as one ordinary candidate,
preserving their component boundaries and logical file names. `RoutedPolicy`
chooses a branch from public observations and records the actual route in Trial
evidence. `self_harness.retention.RetentionSelector` binds historical capability
floors to fresh paired selection, so aggregate improvement cannot override the
declared retention limits. See [Harness Tree](../docs/core-development.md#harness-tree-and-capability-retention)
for the four-capability CPU study, snapshot/rollback behavior and current limits.

`checkpoints.CheckpointStore` owns complete inference bundles and development
training provenance. The optional `self_harness.training.CheckpointProposer`
uses durable provider jobs; `CheckpointEvolutionSpec` separately permits weights
updates while fixing architecture, contracts and other harness components.
Unknown submissions require lookup and reconciliation; completed training metrics
cannot select a survivor. See [optional training](../docs/core-development.md#optional-training-and-checkpoint-evolution)
for the provider contract, two real CPU policy-gradient updates and rollback.
Core has no mandatory training-framework dependency.

`ExperimentRuntime.read_evidence()` performs a fresh audit and returns the
receipt, checked JSON documents and complete evidence hashes together.
`Evaluation` uses those exact parsed bytes; binary artifacts are streamed once
per Trial audit. Development experience and checkpoint ancestry reuse results
within one read instead of reopening the same source for metadata. No digest
cache survives an audit or an adapter call. See [evidence reads](../docs/core-development.md#evidence-read-cost-and-integrity)
for the integrity boundary and corruption checks.

## Other modules

- `contracts.py`: typed, versioned operations and cooperative execution context.
- `infra/`: atomic storage, events, resource pools, process/service lifecycle, RPC and trajectory evidence.
- `lineage/`: current revision, historical evidence and explicit rollback records.

Core imports no CLI, baseline or concrete task implementation. Application adapters provide those dependencies. See [architecture](../docs/architecture.md) for System 1 and System 2.

`self_harness.parameter_search.ScalarRefinementEvolver` refines one declared
numeric JSON parameter using verified development failures, including rejected
probes returned by `CandidateArchive.development()`. It conditions history on
the task, case, scope and fixed controller inputs, records the empirical bracket,
and abstains on contradictory observations. Adapters supply the feedback
classification and monotonicity assumption; fresh paired selection still governs
inheritance. See [parameter search](../docs/core-development.md#development-feedback-parameter-search)
for the CPU acceptance and optional Isaac integration.

`self_harness.adaptive_search.AdaptiveParameterEvolver` searches a declared
`ParameterSpace` of numeric JSON fields. It ranks only complete comparable
development panels, mixes global exploration with sampling near elite observed
vectors, and keeps rejected probes available for later rounds. It does not
assume a monotone response. Environment/verifier identities, execution budget,
contracts and fixed controller inputs condition usable observations. The
objective is an explicit adapter; independent paired selection still controls
inheritance. This is opt-in: the initial three-seed CPU comparison did not beat
uniform search. See [adaptive parameter search](../docs/core-development.md#adaptive-parameter-search).
