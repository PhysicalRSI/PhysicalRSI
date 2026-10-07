# PhysicalRSI-core

Core makes a proposed improvement inspectable: the experiment records its conditions, executes a frozen candidate, asks an independent verifier for an outcome, and preserves the evidence used to select a revision.

## Experiment runtime

`experiments.py` provides `ExperimentRuntime`, `Budget`, and the `Environment`, `Policy`, and `Verifier` ports. An environment verifies reset readiness and reports observations/termination; a policy returns actions; a verifier compares the observations with the declared task requirement.

```python
from PhysicalRSI_demos.experiment import run
receipt = run('/tmp/physicalrsi-experiments', 'trial-1')
print(receipt['outcome'], receipt['evidence'])
```

An experiment directory contains `experiment.json`, `reset.json`, `trajectory.json`, `verdict.json` and `receipt.json`. Adapter identities, budget, case and evidence hashes are bound to the receipt. `success`, `failure`, `uncertain` and `invalid` remain separate. A reset that is not ready cannot become a failed policy trial. An external effect interrupted before completion requires reconciliation; rerunning the same ID never silently repeats it. New receipts use `physicalrsi.experiment-receipt/v2`; `read()` also verifies historical v1 receipts. Changed runtime or adapter identities require a new run ID, not reinterpretation of old evidence.

`embodiment.py` declares an `Embodiment`, a `System1` configuration and optional `System2Trial` attribution. The environment and policy expose `describe()` together. Their observation and action `Contract`s must match exactly, including units, frames and embodiment. A declared policy implements `begin_episode()` to reset history and action chunks using the frozen task inputs. System 1 receives its executing revision through `Context.harness_revision`. The runtime checks adapter identity and dependencies before dispatch; a candidate cannot change its memory revision in the middle of a trial.

Physical declarations require named resources and an explicitly shared `infra.devices.DeviceRegistry`. Leases cover reset, execution, verification and confirmed quiescence. Acquisition is atomic across the whole resource set and coordinates cooperating local processes across experiment workspaces. A killed owner leaves a durable claim; elapsed time cannot release it. A new experiment ID cannot bypass that claim. `lease.json` and `quiescence.json` join the trial's hashed evidence.

Deadlines are cooperative. Adapters must bound device calls and honor `Context.check()`. Device registries require local SQLite storage and POSIX locking on one controller host. They do not fence another controller host or stop a physical device. The runtime verifies local evidence and identity consistency; it cannot guarantee sensor accuracy, verifier accuracy, physical reproducibility or process isolation. Legacy adapters remain available without declared embodiment or resource guarantees. See the [embodied runtime guide](../docs/embodied-runtime.md) for integration, recovery and the remaining infrastructure work.

`timing.ControlTiming` and `infra.controller.ControllerGateway` add controller-side observation references, capture-time/sequence checks, action periods and bounded chunks. Expired commands produce definite rejection receipts; dispatched requests with unknown outcomes remain blocked. `ControllerHost`/`ControllerClient` reuse the RPC infrastructure with receiver-side queue deadlines and observable request status. The [controller gateway guide](../docs/controller-gateway.md) includes an optional MuJoCo demonstration over two distinct actuator contracts.

`infra.control_authority.ControlAuthority` adds persistent gateway ownership for clients with independent filesystems. Reset, action and quiescence require the current generation's credential; action admission also checks expiry, gateway boot and System 1 revision. Revocation remains reachable during driver execution, and `Context.check()` carries the cooperative ownership guard into the driver. Expired or revoked claims require verified quiescence before handoff. The [ownership guide](../docs/controller-ownership.md) covers crash recovery, credential-free evidence, and the distinction between a gateway claim and a hardware watchdog.

`infra.actuation.ActuationGuard` adds a separate authority at the actuator boundary. Its independent device process schedules bounded commands, inhibits output when a command ends or is interrupted, and requires measured settling before release. `infra.actuation_rpc.ActuationDriver` connects that backend to the existing gateway and experiment interfaces. The [actuation watchdog guide](../docs/actuation-watchdog.md) includes controller-process-death tests with continuously advancing native MuJoCo, a normal experiment path, and the hardware watchdog and local-clock limitations.

`infra.operator` provides optional Linux local operator sockets for both hosts. The dedicated `./physicalrsi operator inspect/prepare/recover/status` CLI binds an explicit recovery request to a saved inspection, implementation and ownership generation. Kernel UID authentication, immutable completed receipts and bounded dispatch keep recovery separate from ordinary policy RPC; controller stop calls retain their owning thread. The [operator guide](../docs/operator-recovery.md) covers timeout resolution, deployment isolation and a native fault/recovery demonstration that preserves the interrupted trial's uncertainty.

`infra.clock_sync` adds an explicit bounded mapping between controller and device monotonic domains. It preserves source capture intervals, drift/latency assumptions and immutable converted command deadlines through the actuator and experiment evidence. Shared-domain checks remain the default. The [clock guide](../docs/clock-domains.md) covers conservative freshness, replay, fault handling and native tests with positive/negative device-clock offsets; physical oscillator and network qualification remain external.

`infra.rpc.tls` and `infra.rpc.authenticated_http` add explicitly configured mutual TLS to both control hops, with CA validation, certificate fingerprints, exact method grants, bounded handshakes/requests and authorization audit before dispatch. Plain control listeners and clients are restricted to loopback. Public client security identities join the experiment environment or actuator driver identity; private keys remain operator supplied. The [transport guide](../docs/control-transport-security.md) explains the separate roles of authentication, ownership, timing and local recovery, plus temporary-certificate fault tests. It does not establish worker isolation, physical network availability or hardware qualification.

## Self-Harness

Opt-in `self_harness.protocol.PreregisteredSuite` freezes development,
validation and report-only test cases before an experiment. Its optional
`self_harness.shadow.ShadowGate` admits temporal recovery triggers only after
read-only replay over verified failure and success-control evidence. Missing
features remain inconclusive; passing replay never selects a survivor. See the
[research protocol guide](../docs/research-protocol.md) for the Zetta design
review, public-observation boundary, integration and remaining limitations.

`infra.isolated_policy.IsolatedPolicy` and `infra.isolated_proposal.IsolatedProposal` reuse the Linux chroot/seccomp executor for untrusted stdlib-only candidate source. System 1 retains episode state and returns actions through a bounded data exchange; System 2 returns structured proposal data without device callbacks. `ExperimentRuntime` supports optional policy teardown with hashed cleanup evidence and retains unresolved device claims if cleanup fails. The [isolated candidate guide](../docs/isolated-candidates.md) covers the runtime, actual OS boundary tests and the optional native campaign. This does not automatically isolate trusted model transports, add GPU execution or establish physical qualification.

`self_harness.code_policy.IsolatedHarnessPolicy` binds an executable Python skill artifact to its verified candidate closure and exact skill dependency hash. Explicitly admitted source edits can use the existing structured proposer, isolated execution, independent evaluation and lineage. Source is never evaluated while loading the artifact on the host. CPU tests demonstrate reviewed code replacement and measured inheritance; external model code quality and hardware qualification remain unverified.

The original preview release's `self_harness/` is retained. Its injected proposal, evaluation and selection ports perform:

1. Development feedback, including failures.
2. Candidate materialization with parent and dependency identities.
3. Admission and a frozen comparison pool/profile.
4. Fresh validation cases and paired parent/candidate evaluation.
5. Selection with regression limits and evidence checks.
6. Compare-and-swap inheritance through `lineage.HarnessState`.

`/evolve` runs the CPU service example end to end. Completed stages resume from receipts. Unknown external effects require reconciliation. The current implementation fixes the foundation component during an iteration; it does not automatically improve foundation-model weights.

The experiment runtime is an adapter building block. Task-specific Self-Harness evaluators decide how to turn measured trials into comparison results. `self_harness.experiments.experiment_result()` binds each receipt to its candidate, comparison, cohort, case and scope. It rejects invalid, uncertain, missing or repeated trials. It exports binary success or an explicitly named measurement, retaining the raw evidence and receipt hashes. The selector also requires identical execution/scoring protocols across candidates when these results are used: environment, embodiment, budget, verifier and System 2 identities remain paired while System 1 changes. Existing evaluator ports retain their own protocol boundaries.

`self_harness/selection.py` checks paired cases, evaluator identity, native outcomes, evidence digests and per-task regression limits. It does not infer physical qualification from passing software checks.

`self_harness.evaluation.ExperimentEvaluator` implements a reusable evaluator for declared experiment suites, with separate development/validation plans and explicit recovery from verified trial receipts. `infra.trial_quota.TrialQuota` reserves the entire paired candidate pool before validation starts and preserves its accounting across restart. `self_harness.campaign.ImprovementCampaign` bounds sequential iterations of the same Self-Harness, audits retained rounds and recovers the lineage/checkpoint boundary. The [System 2 campaign guide](../docs/system2-campaigns.md) includes a native MuJoCo example that proposes a reviewed skill change, evaluates it and commits actual lineage.

`self_harness.proposals.StructuredProposer` provides declared JSON edits, a frozen development-only request, a durable strategy response and restartable candidate materialization. `EnumeratedEdits` and `infra.proposal_model.ModelProposal` share that contract; the model adapter has an owned process, a total call deadline and no tool execution. Unknown calls block automatic retries. Model-reported usage is provenance, and performance still comes from the independent evaluator. The [proposal guide](../docs/system2-proposals.md) documents the native simulation option, optional external-provider configuration and limits of the tested evidence.

## Other modules

- `contracts.py`: typed, versioned operations and cooperative execution context.
- `embodiment.py`: exact environment/System 1 binding and System 2 trial provenance.
- `timing.py`: control periods, chunk limits, source sensor samples and observation-bound commands.
- `infra/`: atomic storage, events, resource pools, process/service lifecycle, RPC and trajectory evidence.
- `lineage/`: current revision, historical evidence and explicit rollback records.

Core imports no CLI, baseline or concrete task implementation. Application adapters provide those dependencies. See [architecture](../docs/architecture.md) for System 1 and System 2.

`infra.artifacts.Artifacts` provides byte-bounded, digest-verified local reads for binary observation evidence and restricted NPY array decoding. Publication preserves existing content-addressed evidence, and consumers receive immutable verified bytes or read-only array views. The [observation artifact guide](../docs/observation-artifacts.md) covers limits, supported formats and the remaining transport/sensor responsibilities.
