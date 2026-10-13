# PhysicalRSI core development

The objective is a task-independent substrate for recursive self-improvement:
System 1 executes an explicitly versioned state; System 2 learns from measured
rollouts, creates candidates and selects with independent evidence. Interface
existence alone does not establish an effective improvement algorithm.

## Priorities and acceptance criteria

| Priority | Deliverable | Required evidence | Current boundary |
| --- | --- | --- | --- |
| P0 | Evolvable state, common Trial/Evaluation, reliable rollback | Fixed components rejected before evaluation and at commit; raw rollout attribution; recover from damaged active artifacts; restart without repeated effects | Implemented for local files and the existing experiment runtime; provider-owned assets still require provider recovery |
| P0 | Rollout-driven Self-Harness evolution | Changed proposals respond to observed transitions; fresh validation; independent final tests; cross-task integration; comparison with frozen/search controls on a native workload | CPU numerical and policy-gradient improvement across two action contracts measured; native global-search improvement and archive-fitted second-round proposals measured; benefit from the fitted distribution, broad transfer and physical qualification remain unproven |
| P1 | Candidate archive, asynchronous multi-fidelity evaluation and scheduling | Losing but useful branches remain retrievable; each fidelity has a distinct protocol; out-of-order completion cannot promote against a stale parent; resource reservations reconcile after worker loss | Owned branches, local asynchronous scheduling and development/full evaluation stages implemented; bounded native screening and archive reuse measured; distributed resource backends remain unproven |
| P1 | Experience-to-skill compilation and versioned memory | Compiler binds development evidence to a skill's preconditions, implementation and checks; transfer succeeds on a second task; contradicted lessons suspend a skill revision | Owned experience, checked compilation, pinned runtime binding and memory suspension implemented; local transfer between two action contracts measured, broad/native transfer unproven |
| P2 | Harness Tree, capability routing and retention | Select specialized branches by declared capability; evaluate unseen scenario shifts; retain old capability scores under a shared cost budget | Owned trees, public-observation routing and historical score floors implemented; local four-capability study measured, native and open-world scenario shifts remain unproven |
| P2 | Optional RL/checkpoint evolver | External training job has resumable receipts; checkpoint is content-addressed; evaluation and rollback bind its complete dependencies; ordinary harness updates need no training backend | Explicit checkpoint transitions and durable provider jobs implemented; two CPU policy-gradient updates measured; native/GPU framework adapters and broad learned transfer remain unproven |

The selected native acceptance task is Isaac automatic engineering. The
`auto-engineering --core-research` path connects its real worker to the common
Trial/Evaluation contracts. Its default bounded clearance repair is a control
algorithm; acceptance of a non-scripted proposal provider remains separate work.
P1 archive/scheduler work must preserve the same state, trial and evidence
identities. It must not turn partial or low-fidelity scores into full promotion
evidence. P2 training stays optional and must not become a core import dependency.

## Reference decisions

These are design references, not dependencies or claims of reproduced results:

- [A-Evolve](https://github.com/A-EVO-Lab/a-evolve) exposes an agent workspace,
  evolution engines, trial execution and version control. PhysicalRSI keeps
  component mutation boundaries explicit and versions declared files locally.
- [GEPA's adapter](https://github.com/gepa-ai/gepa/blob/main/src/gepa/core/adapter.py)
  separates evaluation batches, trajectories and reflective dataset construction.
  PhysicalRSI exposes verified rollouts to a replaceable evolver and keeps
  uncertain/invalid physical outcomes separate from task failures.
- [DGM's outer loop](https://github.com/jennyzzt/dgm/blob/main/DGM_outer.py)
  samples parents from an archive and supports staged evaluation. Its archive
  is the reference for P1 branch retention; a selected-version snapshot alone
  is not an evolutionary candidate archive.
- [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve) and
  [RLinf](https://github.com/RLinf/RLinf) inform the separate search and execution
  scheduling layers. PhysicalRSI separates its Trial-count quota from a durable
  local JobScheduler and DeviceRegistry that reserve declared resources and
  coordinate native workers. GPU memory allocation, token budgets, model-server
  capacity and distributed scheduling remain outside this implementation.
- [Evo-Harness](https://arxiv.org/abs/2608.15071) studies compilation of execution
  context into reusable skills. [Hermes](https://github.com/NousResearch/hermes-agent)
  is a reference for persistent agent memory and skills. P1 needs executable,
  evidence-bound transfer, beyond saving textual lessons.
- [Adaptive Auto-Harness](https://arxiv.org/abs/2606.01770) motivates a harness
  tree and solve-time routing for changing task streams.
- [Agent Lightning](https://github.com/microsoft/agent-lightning) and RLinf
  inform optional training adapters. Neither is installed or invoked by this
  implementation.

## Evolvable state and version recovery

`self_harness.artifacts.EvolutionSpec` declares mutable components before any
proposal. The closure includes foundation, skill selection, skills, tools,
control, prompts, memory rules/content, dependencies, weights, assets and
configuration. It binds file identities, not informal change descriptions.

```python
from PhysicalRSI_core.self_harness.artifacts import EvolutionSpec

evolution = EvolutionSpec(("control", "prompts", "skills", "memory_rules"))
# Pass evolution=evolution to SelfHarness(...).
```

The default allows harness components except foundation and weights. The
specification is part of both the frozen round and the stable campaign policy.
Self-Harness checks actual component differences before admission; lineage
checks again before inheritance. The separate `CheckpointEvolutionSpec` below
defines a model transition contract; ordinary evolution keeps weights frozen.

Frozen evaluation artifacts compare canonical JSON identities on reuse. Boolean,
integer and floating-point values retain their distinct serialized identities;
Python equality such as `True == 1` cannot acknowledge a changed request.
Dictionary key order and tuple/list representations of the same JSON array
remain equivalent. A mismatch leaves the original artifact unchanged.

New `HarnessState` bootstrap and selection records own a snapshot under
`state/snapshots/<content hash>/`. Files are copied, not hard-linked. Snapshot
publication verifies content, synchronizes files/directories and atomically
renames the completed directory before the lineage pointer advances. Existing
snapshots are checked and never silently overwritten. Historical records with
external roots remain readable under their original integrity checks.

The original fault-injection audit at
`/cpfs02/user/yanming/physicalrsi-core-checks/harness-mode-audit-001`
changed a captured script from mode 0755 to 0644: execution failed with EACCES,
but content verification passed and restoration reported no repair. New harness
declarations can now bind `file_modes` for every declared file. This map uses
integer POSIX permissions (0000–0777), contributes to the version identity and
is checked alongside bytes. `bind_harness_modes(harness)` explicitly binds a
verified source before a new run; it cannot rebind a declaration to hide drift.
The bundled CPU, Isaac automatic-engineering and real-robot manifest builders
bind modes by default. Historical manifests without this field retain their
original content-only hashes; they do not attest permissions. Start a new lineage
to migrate a historical manifest, rather than changing an existing run's identity.

Mode changes belong to every component owning that file. Frozen components reject
them, and evolution cannot add or drop mode coverage. Snapshot repair detects
mode-only drift, verifies an independent source with the same declared modes,
quarantines the damaged tree and restores the original version and permissions.
Interrupted repair resumes through the existing durable rename protocol.

Parameter mutation and archive reuse preserve the full declaration. New owned
file records use `physicalrsi.owned-files/v2`, preserving and verifying modes;
v1 records remain readable with their original semantics. Mode-bound harness
trees use v2 and distinguish objects with the same bytes but different modes.
Reconstructed leaves, checkpoint packages, training experience and loaded harness
bindings preserve these declarations. The scope is declared regular-file modes:
ownership, ACLs, directory permissions and the operating-system environment remain
adapter/runtime inputs, not claims of this file snapshot contract.

The acceptance at
`/cpfs02/user/yanming/physicalrsi-core-checks/harness-modes-001`
used frozen source manifest
`adc74c759d42da38229045b4581489be701017e49020cda19d2845d8908c1f7a`.
All 637 tests, compile checks and both public CLI smoke commands passed.
On CPFS, a real executable supplied actions to two common-runtime counter cases.
After two successful Trials, changing its snapshot mode from 0755 to 0644 caused
direct execution to fail with EACCES; runtime binding rejected the next attempt
before creating a Trial or resetting the environment. Explicit recovery restored
0755, preserved the lineage head and quarantined the 0644 copy. Both cases then
passed again: four successful Trials and eight executable actions in total.
An independent restart prohibited reset and executable invocation, reproduced
the same evaluations and repair receipt, and preserved the bytes and modes of
65 evidence files. No new Trial or action occurred. The source, fault evidence,
repair journal and restart audit are retained; this is software recovery evidence,
not a new Isaac or physical qualification result.

A completed paired comparison that retains its parent publishes an `assessment`
lineage record. It advances the CAS revision and binds the full decision and
evidence while preserving the same harness snapshot and `freeze_sha256`.
An inherited candidate publishes a `selection` record. Consumers must distinguish
an artifact revision from a lineage revision: a changed lineage head may record
new observations without changing the executing controller. Publication recovery
reuses the exact assessment after interruption and does not rerun completed trials.
Commit retry acknowledgment is limited to `selection` and `assessment` records.
A rollback may retain the target's old decision and evidence, but cannot be
acknowledged as a new commit from its preceding head. Such a request raises a
parent conflict and preserves the rollback head. Archived branches remain
available for new proposals against the restored harness; their old decisions
do not authorize promotion.
No-candidate and all-admission-rejected rounds have no complete paired scores;
they retain their ordinary campaign/archive receipts without an assessment.

Paired selection uses exact-ratio accumulation (`statistics.mean`) for normalized
episode scores and measured costs before converting their means to floats.
Constant costs at their declared limit remain admissible, including finite costs
whose intermediate sum would overflow. Episode ordering cannot change these
means; raw evidence hashes still preserve the submitted ordering. Promotion
thresholds, statistical confidence rules and cost limits remain unchanged.
Task weights are normalized before multiplying scores in selection and adaptive
search, matching screening. This prevents subnormal positive weights from
rounding away measured improvements and incorrectly disabling archive-guided
proposals. Weights remain relative priorities, independent of episode counts.
All three stages first scale weights by their maximum, then normalize their
sum. Individually finite positive weights remain usable even when their raw
sum would overflow; the statistical confidence radius uses the same ratios.
The multiple-comparison correction is evaluated in log space. Strict positive
alpha values can produce finite, persistable rejection evidence even when
`max_comparisons / alpha` would overflow; the confidence rule is unchanged.
A two-round integration check with strict alpha retains the parent despite
observed improvements on both action contracts, persists finite rejection
bounds, and reopens without repeating any of its 22 Trials.
Screening also uses exact-ratio means within each task, so repeating identical
normalized feedback cannot perturb its task score or relative task priority.
Native score bounds must also have a finite positive span, as in screening;
overflowing spans are rejected before any rollout or trial reservation instead
of silently mapping distinct observations to zero during normalization.

```python
restored = state.rollback(
    expected=current_revision,
    target=known_good_revision,  # Omit to select the previous generation.
    reason="position error regressed in deployment",
)
```

Rollback verifies the current history record and ancestry without requiring
the damaged active files to pass. The target still requires complete file and
evidence verification. It records a new CAS revision and reason; retrying the
same completed rollback is idempotent. Consecutive rollbacks step through
selected generations rather than toggling back to a broken revision. Default
rollback skips assessment records for the unchanged active artifact. An explicit
target can still name an audited ancestor. Assessment records do not turn a
bootstrap into a promoted policy or upgrade an offline qualification to physical.

A rollback changes the lineage head, not a running controller or a completed
research campaign cursor. Restart/reload the controller at its declared episode
boundary. An existing campaign rejects that external head change; start a new
campaign with the restored harness and an explicitly chosen research-memory
revision. Rolling back research knowledge is a separate decision from restoring
System 1 behavior.

`state.history(revision)` returns the audited ancestry, newest first. It checks
each record's content address and each declared evidence hash. Shared snapshots
and evidence paths are read once within this call, while every declaration is
compared against the checked content. The retention selector uses this read to
construct its capability floors. The next read starts a new audit; matching
file size or modification time never substitutes for checking bytes. Returned
records describe the files checked during that audit, not a lock on a changing
filesystem.

Rollback alone does not repair a damaged historical snapshot. Retention checks
still audit that ancestor, and ordinary snapshot capture rejects a damaged
directory even when a newly proposed candidate has the same content hash.
`HarnessState.restore_snapshot` explicitly restores the original declared bytes
from an independently verified replica, such as a candidate archive snapshot:

```python
repair = state.restore_snapshot(
    expected=restored["revision"],  # CAS against the current head after rollback.
    revision=damaged_revision,     # Must be the current revision or an ancestor.
    source=verified_archive_candidate["harness"],
    reason="restore exact original bytes from the candidate archive",
)
```

The operation requires an owned snapshot and exactly matching component hashes.
It copies and verifies the replacement, records the request and preparation,
moves damaged bytes into `state/snapshot-repairs/<request hash>/quarantine/`,
and publishes the original snapshot path. A completion receipt binds the target,
reason, implementation and quarantine inventory. A missing snapshot has no bytes
to quarantine. A healthy snapshot records a no-op without requiring source bytes.
Retry the identical request after interruption; once preparation is recorded,
the staged copy is sufficient even if the original replica is unavailable.
Completed retries verify the snapshot and repair evidence without copying again.
Changed quarantine bytes or a newly damaged destination require reconciliation.

Restoration preserves both the lineage head and artifact hash. It neither
reselects a policy nor resumes a completed campaign. A subsequent explicit
Self-Harness round must use fresh validation cases and retain the existing
capability floors. The regression study grows a four-task tree, corrupts its
selected snapshot, rolls back, restores it from the archive, and independently
reevaluates the same artifact in a new round. Historical trial evidence remains
mandatory; restoring harness files cannot repair missing or changed evaluations.
Quiesce affected readers at their declared boundary before maintenance; the two
directory renames are journaled for recovery, not atomic for running readers.

Snapshot scope is declared local files. Referenced checkpoints, providers and
external asset manifests still need their own retention policy. Selection
evidence remains in the experiment workspace and must be retained. Corrupt
history metadata, a corrupt current pointer or missing target evidence require
separate recovery. This is a cooperating-process POSIX store, not a sandbox or
a distributed controller fence.

## Common Trial and Evaluation

`trials.Trial` freezes task, public case, scope, budget and optional System 2
candidate/split attribution in `physicalrsi.trial/v1`. `trials.Evaluation` reads
verified runtime receipts into `physicalrsi.evaluation/v1`, including reset,
actual observation/action trajectory, verdict, measurements and evidence hashes.
Returned JSON is a copy; caller mutation cannot alter an existing value.
Run IDs are unique within an `ExperimentRuntime` root. Paired candidates or
different rounds can reuse the same local ID; cross-run reports must retain
the runtime location and verified receipt identity instead of joining on ID alone.

An environment may implement `export_evidence()` to return a mapping from
relative artifact names to `{path, sha256}`. Before publishing completion the
runtime copies those declared bytes into the trial's `artifacts/` directory,
checks the expected hashes, and binds `artifacts.json` and every copied file
to the receipt. Subsequent reads check all artifacts. Source deletion does not
remove retained evidence; an interrupted copy requires reconciliation. This
interface does not retain undeclared dependencies or referenced external assets.

```python
from tempfile import TemporaryDirectory
from PhysicalRSI_core.experiments import Budget, ExperimentRuntime
from PhysicalRSI_core.trials import Trial
from PhysicalRSI_demos.experiment import CounterEnvironment, CounterPolicy, CounterVerifier

with TemporaryDirectory(prefix="physicalrsi-trial-") as workspace:
    trial = Trial("counter-1", task="counter", case={"initial": 0, "target": 3},
                  scope="software", budget=Budget(5, 30))
    result = ExperimentRuntime(workspace).evaluate(
        trial, environment=CounterEnvironment(), policy=CounterPolicy(2),
        verifier=CounterVerifier())
    print(result.outcome)  # failure: measured position 4 differs from target 3
    print(result.to_dict()["trajectory"])
```

The legacy `ExperimentRuntime.run(...)` remains the execution implementation;
`run_trial(...)` invokes that same path. Development and paired validation use
the common contracts. Unknown effects still require reconciliation; a common
interface does not authorize retries. Success, failure, uncertain and invalid
remain distinct. Final tests remain report-only.

Paired selection hashes each shared physical evidence file once per call,
while checking every reference's declared digest and containment. Conflicting
digests are rejected, and every later selection call reads the files afresh;
this does not cache evidence verification across decisions.

A local CPFS measurement used 32 paired cases sharing one approximately 1 MiB
evidence file. Six alternating baseline/updated pairs after filesystem warmup
produced identical decisions; median selection time was 0.2006 s before and
0.0060 s after, with 64 versus one file hash per call. This single shared-file
workload does not establish end-to-end RSI throughput or native performance.
The comparison script, source hashes and individual measurements are retained in
`/cpfs02/user/yanming/physicalrsi-core-checks/core-selection-evidence-001/benchmark.json`
and its adjacent `benchmark.py` and `baseline_selection.py`.

The Trial runtime and `ExperimentEvaluator` share `runtime_implementation()`,
an explicit hash map of their owned execution dependencies: runtime, contracts,
Trial types, artifact capture, evidence reading, embodiment, timing, device
ownership and storage. The evaluator additionally binds its own planning,
scoring, rollout and harness-artifact code. Every identity check reads current
source bytes; changing a helper cannot preserve the old evaluator revision.
The evaluator compares the canonical JSON identity digest to its frozen revision,
so boolean, integer and floating-point substitutions cannot pass as unchanged
configuration merely because Python considers their values equal.
The same canonical identity comparison protects training coordinators and
experience compilers, including resumed jobs and completed skill acknowledgements.
Round configuration reopening, live port/evolution checks and research reflector
checks also use canonical identity digests. A stored numeric/boolean type change
is rejected without overwriting the conflicting configuration or spending Trials.
A change before reservation blocks the whole plan; a change during adapter
construction blocks reset while retaining the already charged reservation.

This versions the declared execution implementation, not external Python,
simulator, model-server or asset dependencies. It is a trusted source contract,
not attestation of loaded machine code or a malicious-process sandbox. A new
implementation requires a new frozen campaign; ordinary `Evaluation.read()` can
still inspect completed historical evidence without executing adapters or
silently migrating its evaluator identity. Source-mutation tests run in isolated
copies and never edit an active experiment's implementation.

New evaluator cohorts declare `case_identity: task-case/v1` and hash the task
with its public case. Different task domains can use the same seed or payload
without colliding. Within-task duplicates and semantic reset reuse remain
rejected. The result bridge still reads legacy unqualified `case/v1` cohorts.
Task names must represent actual task domains; renaming one task does not make
its examples statistically independent or justify reusing its validation data.

### Explicit replacement after an interrupted reset

An SDK startup failure can stop a paired comparison before a task outcome
exists. `ExperimentEvaluator(max_replacements=N)` declares an optional bound
of zero to ten replacement attempts per planned Trial; the default is zero.
The allowance is part of evaluator identity and must be frozen before the
campaign starts. `TrialQuota` must also have capacity for the additional attempts.
An authorization consumes a reservation even if execution never starts; neither
the original failed attempt nor a prepared replacement is automatically refunded.

The runtime records `execution_phase` before entering reset and after saving its
result. Replacement requires an interrupted reset, no returned reset record or
dispatched action, inactive local ownership, and an explicit operator attestation
with hashed stopping evidence. If the Trial leased a device, its original lease
must first be reconciled through the shared `DeviceRegistry`. A missing receipt
with other retained execution files also blocks execution. File absence, elapsed
time and a dead process alone never authorize a replacement.

```python
# Construct the evaluator with max_replacements=1 and sufficient TrialQuota.
# Independently stop owned workers and reconcile any device lease first.
replacement = evaluator.authorize_replacement(
    candidate, output=round_root, split="validation", run_id=logical_trial_id,
    failed_run_id=failed_execution_id, operator=operator_name,
    reason="Owned startup process stopped; reset did not become ready",
    evidence=stopping_evidence_paths_and_hashes, quiescent=True,
)

# For ScheduledEvaluator, explicitly release the failed queue reservation for
# its already authorized replacement work. This verifies the whole Trial plan.
evaluator.resume_replacements(
    failed_job_id, operator=operator_name, reason="Replacement Trial authorized",
    evidence=stopping_evidence_paths_and_hashes,
)
result = campaign.run(initial)
```

For a `MultiFidelityEvaluator` screening interruption, use
`authorize_screening_replacement(candidate, output=round_root, run_id=logical_trial_id,
failed_run_id=failed_execution_id, operator=operator_name, reason=reason,
evidence=stopping_evidence_paths_and_hashes, quiescent=True)`, then
`resume_screening_replacements(failed_job_id, operator=operator_name, reason=reason,
evidence=stopping_evidence_paths_and_hashes)`, before calling `campaign.run(initial)`.
The output is the round's execution root, just as for full validation; the API
resolves the nested screening directory. The low-fidelity evaluator's frozen
`max_replacements` applies. Screening and full validation share quota, but their
queue handlers remain distinct and reject recovery through the wrong handler.
Both use the same inactive-plan and authorization audit. A replacement screening
measurement only decides admission; survivors still undergo full paired validation.

Authorization does not execute adapters. It assigns a distinct Trial ID, retains
the original failed directory, and owns copies of the failed specification,
receipt, other retained files, stopping evidence and device reconciliation. The
new `Trial` carries the authorization hash. `Evaluation.read()` verifies this
owned closure, so a completed replacement remains auditable after its original
failure source is lost. Candidate, task, case, execution budget, adapters,
scoring, contracts and System 2 attribution must match the predecessor.

The plan continues to identify its original logical Trial. Development feedback
and paired results expose the actual replacement execution and bind its logical
identity to that plan. Only completed replacement measurements reach proposal,
selection or archival scoring; failed startup records remain recovery evidence.
The candidate archive retains that closure. Final tests stay report-only and
retain their original case denominator. Additional attempts consume quota and
appear in development cost accounting.

The queue preserves each interrupted dispatch's error, operator decision and
replacement proof. Repeated identical authorization or queue-resume calls are
idempotent. Another interrupted startup needs a new explicit authorization and
reservation. Completed or abandoned jobs cannot be reexecuted through this API.
Read-only collection and ordinary restart never authorize a replacement.

This is a cooperating-process recovery contract. Operators and providers remain
responsible for independently establishing that effects have stopped. It does
not automatically diagnose SDK crashes, stop devices, repair corrupt evidence,
or reconcile actions already dispatched. Completed success, failure, invalid and
uncertain outcomes cannot be replaced to improve a score. Historical interrupted
receipts without the new phase evidence remain unresolved. Updating core requires
a new frozen campaign; an old interrupted study is not silently migrated.

Cross-task software tests inject startup faults into development and validation
for both scalar and structured-action tasks. They exercise the complete research
campaign, candidate archive, held-out report, extra reservation and restart, plus
live ownership, device reconciliation, repeated interruption, quota exhaustion,
interrupted authorization publication and evidence corruption. These are software
recovery checks; native evidence is reported separately below.

The recovery implementation published as `d6ccf65` passed 722 software tests.
Four separate CPU campaigns covered interrupted development and validation
for both action interfaces: 56 completed Trials, four interrupted attempts and
eight successful final cases. A separate process resumed all four campaigns
without new execution, preserving the bytes and modes of 1,280 evidence files.

A native Isaac Sim 5.1 startup check then interrupted one owned worker after
scene layout creation and before reset readiness. After checking process-group
quiescence and reconciling its device lease, an explicitly authorized new Trial
completed 1,576 physics steps and passed all seven transfer checks. Both attempts
consumed reservations, and the original failure evidence remained unchanged.
This used known development seed 5000 and fixed transit height 0.36 m; it proves
the native replacement path, not generalization or a fix for the earlier USD
crash.

A separate two-round native campaign completed five full Trials (7,880 physics
steps) and one deliberately interrupted validation startup, consuming six
reservations. On validation seed 681000 the parent failed and the 0.36 m
clearance candidate succeeded; selection inherited the child. The second round
inherited both the selected harness and the updated research memory, succeeded
on development, and retained the policy without a changed candidate. The final
report on held-out seed 691000 succeeded (1/1). Independent calculations from
the retained object measurements agreed with the seven task checks. This
establishes native recovery and recursive state/memory integration for the
declared control and small panels; it does not establish search superiority or
benchmark-wide generalization. No model weights were updated.
Evidence is retained under
`physicalrsi-core-checks/native-trial-recovery-001/` (including `campaign-study/`) and
`physicalrsi-core-checks/core-trial-recovery-002/`.

`auto_engineering.research.setup()` exposes the same `max_replacements` option
and `extra_trials` for explicit additional total Trial capacity. Both default
to zero; full evaluation and staged screening share the declared allowance and
quota. These options do not retry automatically or provision simulator workers.

## Rollout-driven proposals

`self_harness.rollout.RolloutProposer(evaluator, evolver)` obtains development
trials from `ExperimentEvaluator`. Before calling the evolver it rehydrates
each receipt, verifies the candidate and split, checks every evidence hash,
and requires complete coverage of the frozen development plan. It rejects
altered traces, substituted outcomes, duplicate trials and omitted episodes.
The evolver receives `propose(parent, evaluations, context, output)` and returns
materialized candidates with the existing Self-Harness provenance. The parent
and evidence are rechecked after the call. Interrupted proposal calls require
the provider's explicit `resume_proposal(...)` reconciliation implementation.

The CPU example estimates actuator response with least squares from actions
and measured position changes. The same algorithm updates a scalar command
task and a structured joint-command task. It evaluates new target positions,
inherits the accepted controller, then uses that controller in the next round.
An independent final test uses targets absent from development and validation.

```bash
PYTHONDONTWRITEBYTECODE=1 python - <<'PY'
from tempfile import TemporaryDirectory
from PhysicalRSI_demos.rollout_evolution import run
with TemporaryDirectory(prefix="physicalrsi-rollout-") as workspace:
    result = run(workspace)
    print([row["outcome"] for row in result["rounds"]])
    print(result["final_test"]["metrics"])
    print(result["quota"])
PY
```

Expected decisions are `inherit_child`, then `no_changed_candidate`. Each of
the two final cases succeeds; the complete run reserves 14 trials. Reopening
the same workspace verifies receipts without more environment calls. The
tests also change the hidden dynamics and require correspondingly different
fitted controls, avoiding a preconfigured corrective parameter.

## Candidate branches and search history

`self_harness.archive.CandidateArchive` owns every candidate in a completed
round, including admission rejections and evaluated candidates that did not
win. Each immutable entry binds the parent and resulting lineage revisions,
search policy, admission reasons, per-task measurements and selection decision.
Candidate files are copied into content-addressed snapshots. Declared raw
development and validation evidence, including binary media, is copied into
the archive. Reads verify the entire retained closure, including ancestor
entries referenced by frozen search views. Missing or corrupted historical
evidence invalidates dependent entries. Published entries are never repaired
silently from potentially changed sources.

Each archive read traverses an ancestor only once and shares verification of
identical owned snapshots within that traversal. `fork` uses one audited view
for its source and ancestry before copying the branch. This reuse does not span
public calls, provider execution or separate reads. Historical search views are
parsed from the exact bytes whose hashes were verified; the entry's publication
record is checked again after reading its evidence. Directory enumeration rejects
links and non-regular files, including undeclared FIFOs. The archive identity pins
the snapshot and file-enumeration helpers as well as its own implementation.
Existing frozen studies must be inspected with their recorded source version.

Campaign recovery batches completed-round archive receipts by owned archive.
Every receipt still checks its implementation and source-round binding, while
shared historical entries and candidate snapshots are verified once within that
call. Separate `run`, `report_test` and archive reads start fresh audits; no
verification state survives provider execution or a later restart. Within an
entry, physical directory checks are shared across evidence files and directory
identities are checked again before returning. Every leaf is checked as a regular
file and all evidence bytes are hashed. Directory replacement, new entries and
symlinks observed during the audit invalidate it. These checks assume cooperating
POSIX storage and do not provide an atomic snapshot against hostile concurrent
filesystem changes.

The retained `archive-restart-001` comparison uses independently frozen baseline
and updated sources. Both four-round CPU adaptive-control campaigns executed 172
Trials. After matching candidates by executable revision, the task/case/split,
trajectory digests, verdicts, quotas and selection outcomes agree. Five repeated
reopen measurements of `run` plus `report_test` had median elapsed times of 2.699
and 0.782 seconds (71.0% lower). Each reopened 2,979 files without new Trials or
actions, and all file bytes and modes remained unchanged. A profiled reopen
reduced archive entry reads from 20 to 8, across two separate fresh audits.

Synthetic shared-history fixtures at 8, 32 and 64 retained rounds checked entry
reads of 36/528/2,080 before batching and 8/32/64 afterward. The 64-round fixture
took 25.58 versus 0.28 seconds. This counts full entry audits; stored ancestor-view
references still grow with history. Timing is local CPU/RAM-backed filesystem
evidence, not a distributed-storage guarantee or an improvement in task success.
All 646 tests passed. Sources, profiles, raw reports and byte/mode-verified rollout
archives are retained at
`/cpfs02/user/yanming/physicalrsi-core-checks/archive-restart-001`.

Pass the same archive to `SelfHarness(..., archive=archive)` and
`RolloutProposer(evaluator, evolver, archive=archive)`. Keep its directory
outside all campaign round evidence. The rollout-calibration and native Isaac
research adapters enable an archive at `<workspace>/archive`.

Before development, Self-Harness freezes `archive-view.json` for the declared
scope and search policy. The proposer receives that view and candidate
summaries through the optional evolver method
`propose_from_archive(parent, evaluations, research_context, search_context, output)`.
The existing research context remains unchanged; `search_context` contains
`view` and `candidates`. Interrupted calls need the matching
`resume_proposal_from_archive` method. Providers using ordinary `propose`
continue to receive development evaluations and their original context.
Later entries cannot enter a resumed proposal. Archive summaries are historical observations across cohorts, not
fresh promotion scores. Preregistered test payloads and final-test reports
are excluded from archive evidence. Providers remain trusted code rather
than processes isolated from other files on the host.

`archive.fork(view, revision, candidate_id, parent=..., output=..., name=...,
evolution=...)` copies an old branch into a new proposal directory. It binds
the current parent, the archived source and the frozen view; fixed components
remain protected. A research proposer must add its hypothesis, expected effect
and required checks, and update component hashes after any further edits.
The proposal then follows ordinary admission, a fresh paired comparison and
CAS selection. Retrieving or materializing a branch never changes the head.

Publication is idempotent. If selection finishes before archive publication
is interrupted, a resumed round completes the archive without repeating
rollouts. Campaign restarts also verify archived files. Incomplete rounds and
stale-parent comparisons remain in their original workspaces for recovery;
this archive is not an asynchronous job ledger. Archive retention can recover
candidate files and historical observations after source loss, but it does
not replace the original lineage/campaign store or its recovery procedure.
External assets referenced by manifests still belong to their providers.

The CPU branch example deliberately searches individual task repairs in round
one. The vector specialist wins the weighted comparison; the axis specialist
remains archived. Round two retrieves the losing axis branch and combines it
with the incumbent's successful vector control. Fresh validation selects the
composed candidate; separate tests cover both action contracts.

```bash
PYTHONDONTWRITEBYTECODE=1 python - <<'PY'
from tempfile import TemporaryDirectory
from PhysicalRSI_demos.branch_evolution import run
with TemporaryDirectory(prefix="physicalrsi-branches-") as workspace:
    result = run(workspace)
    print([row["outcome"] for row in result["rounds"]])
    print(result["final_test"]["metrics"])
    print(result["quota"])
PY
```

Expected: two `inherit_child` decisions, both final test cases successful, and
26 reserved trials. This verifies branch retention and reuse on actual CPU
transitions, not an optimal search algorithm or OpenEvolve/DGM reproduction.
The following evaluation stages extend this same candidate and evidence path.

## Asynchronous evaluation and fidelity stages

The scheduler owns JSON snapshots of submitted requests and each callback's
returned value. A collector that updates a provider's shared cache cannot
retroactively change what `execute()` returned. Equality uses canonical JSON
hashes, matching durable result identity; Python's equality between `True` and
`1` is insufficient. Resource requests are copied before validation and handler
callbacks. Every execution or collection checks the handler identity on both
sides, including recovery and reads of already completed jobs. A changed
collector cannot return evidence under an old handler revision or cause pending
jobs to be dispatched first.

Execution/collection disagreement leaves the job in `needs_reconciliation` and
keeps its reservation. Verified complete evidence may recover it without a new
execution. A failed read of a completed job preserves its original receipt.
Snapshots do not make a provider's `collect()` side-effect-free; the provider
still owns that contract and the completeness of its evidence checks.

`infra.scheduler.JobScheduler` persists queued inputs, verified results and
resource reservations outside campaign evidence. `submit` is idempotent for
identical payloads, handler identities and resource requests. Each submitted
batch snapshots one handler identity before consuming its inputs and rechecks
it under the queue lock before adding any jobs. A configuration change during
input iteration rejects the whole batch without changing the existing ledger.
`work_once` lets
independent cooperating local processes claim fitting work; `run_until` provides
a bounded thread dispatcher and returns verified results in requested order.
Higher numeric priorities dispatch first among requests that fit. A large
blocked request does not prevent smaller fitting requests from running.
Each claim scans queued jobs against one freshly observed handler identity;
unrelated jobs do not cause repeated handler implementation hashing. This
observation is not cached across claims. Execution and collection still check
the handler identity before and after each callback.

Configure capacities explicitly, such as `{"cpu": 2}`. These are scheduling
allowances, not inferred hardware capacity, enforced memory limits or cluster
allocations. All cooperating controllers must share one queue on one POSIX
host. Suite factories must create independent adapters and support concurrent
calls. Physical device ownership still requires the runtime's `DeviceRegistry`;
a queue reservation does not replace its leases or native cleanup.
Each observation scan checks one shared handler identity against every selected
request, while subsequent scans and callback boundaries revalidate it. A local
CPFS microbenchmark with 64 waiting jobs and the native Isaac Suite identity
reduced identity observations from 64 to 1 and median scan time from 1.811 s to
0.852 s (five alternating pairs after one warmup pair). Jobs remained queued;
this measures observation overhead, not native rollout or end-to-end speedup.
Evidence: `physicalrsi-core-checks/core-scheduler-observation-001/benchmark.json`.
Queue identities use canonical JSON comparison, and Trial quota ledgers require
integer allowances when read as well as when created. Boolean or floating-point
substitutions for integer capacities/allowances stop reopening and dispatch or
reservation without rewriting the conflicting ledger.

A job's owner lock and durable state are distinct. A dead worker retains its
capacity and cannot be retried automatically. `recover_completed` invokes only
the handler's read-only `collect`: all required trial receipts and evidence
must already exist. Successful recovery releases the reservation without new
rollouts. Partial effects remain unresolved. `abandon` requires an operator,
reason and hashed quiescence evidence; it releases queue capacity while retaining
the failed input and does not clear a device lease or authorize replay. Live
owners cannot be recovered or abandoned. Observation timeouts do not expire
ownership; owned threads are joined before the dispatcher returns, so adapters
must honor their own execution budgets.
Each `run_until` claim rechecks its observation deadline while holding the queue
lock and again after acquiring the owner lock, before reserving a new job. Slow
queue I/O or lock acquisition cannot authorize another dispatch after expiry.
Already reserved work is joined; untouched jobs stay queued and a later explicit
`run_until` can finish them without rerunning completed inputs.
An executor submission error does not prove that work never started. The
dispatcher joins the executor before releasing that claim's owner lock; verified
completed evidence remains recoverable without replay. If no complete evidence
exists, the durable reservation still requires explicit reconciliation.

`self_harness.scheduled.ScheduledEvaluator` adds the optional
`evaluate_pool`/`resume_evaluation_pool` port to `ExperimentEvaluator`. The
complete paired pool is reserved before dispatch. Candidate comparisons can
finish out of order, but every candidate still receives its own verified
evaluation receipt. Missing, uncertain or changed evidence prevents selection;
lineage CAS rejects completion against an outdated parent revision. Completed
queue receipts are rechecked against raw trials when reused.

`self_harness.fidelity.MultiFidelityEvaluator` adds one development screening
stage before full validation. Both stages share the preregistered case splits
and trial quota, and have distinct evaluator/System 2 identities. The cheap
stage runs all ordinarily admitted children on development cases, using its
declared adapters and budgets. It requires the frozen minimum success rate on
every task. The incumbent remains available for full paired comparison and is
recorded as `measured: false, per_task: null` at screening. No measurement is
invented or reused for it. Set `screen_incumbent=True` to measure the entire
pool for diagnostics; this policy is frozen before effects. An uncertain
screening outcome blocks full dispatch.

An optional `survivor_limit` caps the number of children sent to full validation.
It requires `ranking_profile={task: {"weight": positive_number,
"score_range": [low, high]}}` covering every screening task. The screening
`score_measurement` supplies native scores (or binary success when omitted).
Each score is normalized within its declared range, averaged within its task,
and then averaged using the declared task weights. Higher scores rank first;
exact ties use executable content hashes, independent of candidate names or
scheduler completion order. Only children that satisfy every success threshold
compete for slots; the incumbent is always retained outside the cap, including
when it has no screening measurement.
An integration regression renames and reverses the complete child pool across
two independently executed calibration campaigns. Ranked content identities,
admitted branches, the selected harness and Trial consumption stay identical;
reopening either campaign performs no additional rollouts. This checks the
declared tie policy, not a performance advantage from screening.

A bounded screen rejects missing, nonfinite or out-of-range measurements before
full dispatch. Profiles, limits and measurement mappings are frozen in evaluator
identity. Screening records retain per-task normalized means, aggregate scores,
ranks and cutoff reasons; rejected branches keep their rollout evidence in the
archive and remain available as development feedback. This is a fixed allocation
policy, not a guarantee that the best validation candidate survives: noisy or
biased development scores can discard useful children. The default remains
unlimited. Full validation still runs fresh paired cases and enforces its own
scores, success and regression constraints. For example, the CPU control demo
accepts `adaptive_control.setup(path, survivor_limit=2)`.

Only children passing screening enter the frozen full pool, whose independent
validation cases are then generated. Cheap scores never become promotion
scores. This implementation supports development screening followed by full
validation; arbitrary fidelity ladders and adaptive allocation policies are
future work. Screening decisions, rejected candidate snapshots and declared raw
evidence remain in `CandidateArchive`, with screening measurements separate
from full validation metrics. Test cases stay report-only.

The CPU acceptance example compares the same five-member proposal pool under
three screening policies:

```bash
PYTHONDONTWRITEBYTECODE=1 python - <<'PY'
from tempfile import TemporaryDirectory
from PhysicalRSI_demos.staged_evolution import run
for screened, screen_incumbent in ((False, False), (True, True), (True, False)):
    with TemporaryDirectory(prefix="physicalrsi-stages-") as workspace:
        result = run(workspace, screened=screened, screen_incumbent=screen_incumbent)
        print(screened, screen_incumbent, result["quota"], result["final_test"]["metrics"])
PY
```

Expected: 26 total reserved trials without screening, 24 with the entire pool
screened, and 22 with the default children-only screening, including development
and the two report-only tests. All three select the fitted calibration and
succeed on the scalar and structured action tests. Skipping the incumbent's
redundant screening saves two trials while preserving its full paired scores.
Three
overcorrections are deliberately included to exercise rejection; this small
software example does not establish general screening efficiency or native
simulator performance. Process-exit tests additionally cover retained uncertain
reservations, complete-evidence recovery and live-owner protection.

### Observed ranked-screening comparison, 2026-10-13

The preregistered `ranked-screening-001` study compared unlimited screening
with `survivor_limit=2` on seeds 30–39. Both arms used diagonal adaptive search,
five-observation warmup, four rounds, six proposals per round, the same 228-Trial
cap, task weights, success thresholds and independent promotion gates. Fresh
case panels changed effective target displacements and load scales relative to
the earlier search studies. The two tasks share a force action contract and
have different load dynamics; they are numerical CPU rollouts, not Isaac runs.

| Screening policy | Mean final quality | Final successes | Total Trials |
|---|---:|---:|---:|
| Unlimited | 0.822302 | 40 / 40 | 1,800 |
| Highest two passing children | 0.822302 | 40 / 40 | 1,620 |

All ten pairs selected identical final gains and obtained identical final
quality. The bounded policy saved 180 Trials (10%), with per-seed savings of
4–24 Trials. It passed the preregistered requirements: at least 5% fewer Trials,
mean quality difference no worse than -0.005, worst paired difference no worse
than -0.02, and no decrease in final successes. These are descriptive results
for the declared task distribution, not a statistical guarantee or wall-clock
speedup claim. The default remains unlimited; no final-test-driven retuning was
performed, and these now-observed final panels cannot serve as fresh holdouts.

An independent audit reconstructed all 3,420 trajectories and verdicts from
controller gains and dynamics, recomputed development rankings, and checked
that full-validation pools contained exactly the incumbent and eligible
survivors. All screening branches remained available as development feedback.
Twenty independent restarts created no Trials or actions and left 59,218 files'
bytes and modes unchanged. All 755 scheduler jobs completed with no retained
capacity. A separate CPFS interruption after screening completion recovered the
identical ranking without repeating any screening Trial. The full suite passed
681 tests. Frozen source, preregistration, raw artifacts, audit results and
verified byte/mode-preserving archives are retained at
`/cpfs02/user/yanming/physicalrsi-core-checks/ranked-screening-001`.

## Compiled skills and versioned memory

`self_harness.skill_compiler.ExperienceArchive` accepts the same verified
development feedback used by `RolloutProposer`. It independently retains the
source harness, complete trial receipts and declared evidence files, including
binary artifacts. Reads reconstruct `Evaluation` values from those owned files
and verify their candidate, task, split and plan attribution. Compilation can
continue after the original experiment workspace has been removed; test and
validation results cannot be substituted for development inputs.

`ExperienceCompiler` separates three explicit provider responsibilities:

- A compiler materializes an executable file bundle from owned evaluations and
  a frozen request containing the skill name, applicability conditions and
  required checks. It declares the exact file closure and a testable hypothesis.
- A backend builds a typed `Operation` from that bundle. It is selected by the
  caller and pinned by identity; memory cannot name an import or select code to
  execute. The provider owns review, dependency completeness and any necessary
  process isolation.
- A checker actually exercises that operation and binds each required report
  to source trial identities. A failed check preserves its report without
  publishing an eligible skill. Trusted checkers still own scientific validity;
  a passing offline check does not establish transfer or deployment success.

`compile` records materialization and checking separately. Final report
validation uses the frozen required-check list; later edits to the caller's
list belong to a new compilation request. Interrupted provider calls require
`resume_compile` or `resume_check`; source files and provider
identities are checked again before publication. Published skills bind the
owned experience, compiler/backend/checker identities, operation contracts,
applicability conditions and check results. `infra.owned_files.OwnedFiles`
publishes metadata and copied files atomically and rejects changed, missing,
extra or linked files. It never repairs a published version silently.

New compilations freeze `bundle.json` before backend construction. It binds the
declared file hashes and regular POSIX permission bits to the check-stage inputs.
Both backend construction and checking must preserve that exact bundle, and a
resumed stage cannot reuse a completed report after permission drift. Publication
passes the checked modes to `OwnedFiles.capture(expected_modes=...)`; it must not
adopt a later filesystem state. Nonregular or linked compiler output is rejected
before backend construction. These checks cover declared local files, not
provider-owned external assets or process isolation.

`physicalrsi.compiled-skill/v2` retains the checked bundle in the owned skill's
metadata. Reads require an exact match with the published files and modes before
binding or installation. Historical v1 skill records remain readable under their
original contract. Existing compilation workspaces and memory references require
their frozen implementation; they are not silently migrated to new check inputs.

The `compiler-inputs-001` fault injection reproduced a checked mode-0755 script
being published and bound at mode 0644 by the prior compiler; actual invocation
then failed with EACCES. The updated compiler rejected both post-check drift and
drift immediately before storage capture without publishing a skill. After an
interruption following completed checks, changed permissions blocked recovery;
restoring the exact checked mode resumed publication without repeating synthesis
or checks.

A fresh executable response skill was compiled from two axis development Trials
and used by the ordinary Self-Harness on axis and vector tasks. This CPFS study
completed 16 Trials, including independent final success on both tasks, and 14
actual executable invocations (six offline-check calls and eight target actions).
Independent process recovery added zero Trials/actions and preserved all 378
tracked files and modes. All 654 tests passed. The old failure, new evidence,
source snapshots and restart audit are retained at
`/cpfs02/user/yanming/physicalrsi-core-checks/compiler-inputs-001`.
This demonstrates executable compilation/transfer integrity on a software
actuator family; it is not native Isaac transfer or physical qualification.

`skills.SkillLibrary.materialize` copies a skill package into a candidate's
declared `skills` component. `bind` requires an explicit memory revision,
lesson ID, applicability context and matching backend. It verifies evidence
and contracts before construction, and checks the pinned memory and executable
files on invocation. Source evidence is fully audited at binding rather than
replayed on every action. Retain the library and research memory stores alongside
lineage: the candidate owns its executable copy, while those stores retain its
provenance and lesson history. Provider-owned external assets are not copied
implicitly.

Each `ResearchMemory.read` hashes a shared evidence path once across all lessons
and checks every declared digest against that result. The cache belongs only to
that call: later reads recheck the files, and conflicting digest declarations
are rejected. This reduces repeated evidence I/O without a persistent cache.
Snapshot writes likewise verify each successor evidence path once. Parent
verification remains separate, and repeated writes recheck the evidence before
acknowledging an existing revision. Changed, missing or conflicting evidence
cannot publish a successor or modify an existing snapshot.
A local CPFS benchmark with 32 lessons sharing two 1 MiB files measured six
alternating old/new read pairs. Median read time decreased from 242 ms to 10 ms,
with 64 versus two file hashes and identical returned snapshots. This used
warmed filesystem caches and measures one read, not end-to-end RSI throughput.
Source hashes and individual observations are retained in
`/cpfs02/user/yanming/physicalrsi-core-checks/core-memory-shared-evidence-001/benchmark.json`.

`install` creates a new `ResearchMemory` snapshot without replacing prior
lessons. A lesson marked active is eligible for an experiment; normal
Self-Harness admission, independent paired validation and CAS still select the
runtime harness. System 1 declares both the skill revision and the memory
revision in its dependencies. `suspend` accepts only a definite failed
development trial whose execution binding names that exact skill and memory.
The new memory revision retains the supporting and contradicting evidence.
Prose refinement alone cannot re-enable the contradicted executable: recompile
and install a new skill revision under a new lesson ID. Historical memory
snapshots remain unchanged. Suspension blocks bindings using the new snapshot;
it does not silently mutate already pinned historical executions or stop a
robot. Runtime withdrawal requires an explicit state transition or rollback.

A multi-skill integration regression compiles independent response gains of 2
and 0.25 into one library under distinct actuator-family conditions. With both
lessons installed, a failed development Trial suspends the first skill while
preserving the second lesson exactly. Fresh vector Trials using the independent
skill succeed under both historical and current memory revisions; their receipts
bind the corresponding skill and memory. Wrong-context bindings are rejected.
The test executes 22 software Trials in total and reopening the completed campaign
adds none. This checks retention and version isolation, not native transfer.

The CPU acceptance example learns an inverse response from two scalar-axis
rollouts, compiles a vector action program and installs its skill lesson. The
target Self-Harness uses that same program for scalar and structured vector
actions, with explicit shape adapters and unchanged baseline control files.
No target-task, validation or test trajectories enter compilation. After
selection and separate final tests, a new development-only dynamics shift
provides a counterexample. The example suspends the lesson, rolls back the
selected harness, and executes a further trial without the withdrawn skill.

```bash
PYTHONDONTWRITEBYTECODE=1 python - <<'PY'
from tempfile import TemporaryDirectory
from PhysicalRSI_demos.skill_transfer import run
with TemporaryDirectory(prefix="physicalrsi-skill-transfer-") as workspace:
    result = run(workspace)
    print(result["final_test"]["metrics"])
    print(result["quota"])
    print(result["restored"]["harness"]["id"])
    assert run(workspace) == result
PY
```

Expected: one successful report-only test per action contract, 18 total trials,
and rollback to `uncompiled`. The post-withdrawal trial still fails on the
changed dynamics; withdrawal is not presented as a repaired controller.
Completed studies audit their retained files on restart. A crash after rollback
can finish the pending withdrawal report without repeating earlier trials.
Tests also compile a distinct revision from the counterexample and preserve the
old suspended lesson. That revision remains a candidate requiring fresh
validation. These results demonstrate local transfer within one linear actuator
family, not transfer between unrelated robot tasks or learned model checkpoints.

## Harness Tree and capability retention

`harness_tree.HarnessTree` packages several complete harnesses into one normal
Self-Harness candidate. Each node has a parent, exact capability conditions and
a full component declaration. The root is the unconditional fallback. Children
must strictly narrow their parent's conditions; overlapping siblings, cycles,
unknown capability keys and disconnected nodes are rejected. Trees are bounded
to 64 nodes and depth 16. Routing chooses the deepest matching node without a
priority heuristic or reading evaluation scores.

Packaging copies bytes into content-addressed objects and records the original
logical file names separately for every component. Identical component manifests
are shared. A branch's foundation and weights must match the root; its other
changes must obey the declared `EvolutionSpec`. The resulting tree declares all
objects and manifests in their original components, with routing in
`skill_selection`. This allows ordinary component comparison to reject a hidden
model, tool or configuration change, including a renamed logical input. Adding
a control branch does not invent changes to unchanged foundation files.
`materialize` reconstructs each branch's original file layout for an explicitly
supplied backend. Neither publication nor branch membership authorizes deployment.

`RoutedPolicy` is a System 1 adapter. Its backend builds declared branch policies
without starting workers; a separately pinned extractor receives only the
initial public observation. Core supplies the task identity. All possible
branches for that task must agree on observation/action contracts and timing.
The wrapper declares their versioned dependencies and the whole-tree revision,
pins one route for the episode, passes the selected leaf its own revision, and
delegates cleanup only to that leaf. Cleanup uses the originally bound method
and revision even when an integrity failure prevents any further action.
Provider constructors and extractors remain
trusted code; routing is not a sandbox or an automatic contract converter.
Full-tree file audits bracket execution. Per-action checks verify the routing
manifest, selected branch files and provider identities; inactive branch bytes
are audited before the first action and before a completed receipt is published.

The experiment runtime's optional `execution_evidence()` policy port returns a
nonempty JSON object. Its presence is frozen before reset. Evidence is written
after `begin_episode`, before the first environment action, and collected again
at completion. `policy-evidence.json` belongs to the Trial's verified closure,
so archive and experience capture retain it too. Routed policies record the
public capabilities, traversed nodes, tree revision, leaf revision and extractor
identity. An uncertain action retains the selected-route evidence without
authorizing a replay. Route identity does not change halfway through an episode.

`self_harness.retention.RetentionSelector` wraps the existing paired selector.
Before a research round, it freezes a view of the declared parent's audited
lineage and the highest observed normalized score for each capability, including
past incumbents, retained-parent assessments and selected survivors. All generations must share the scoring
policy and task coverage. Candidate scores must pass both ordinary paired
selection and the declared historical regression limits. This prevents a series
of individually tolerated regressions from silently erasing an older capability.
If the provisional winner violates a floor, selection considers the remaining
eligible candidates; it does not deploy the violating winner. The view and
effective floors are bound into round identities and the selection receipt.
Lineage CAS still rejects commits against a stale parent.
The four-capability tree regression also checks rollback from three branches to
two: all four historical score floors remain at 1.0, the frozen pre-rollback view
remains unchanged, and the post-rollback selector rejects the old comparison's
parent binding. Reading these views and rejecting stale evidence adds no Trials
or quota reservations. Rollback therefore does not erase measured capabilities
or turn previously collected validation into a fresh comparison.

These floors are empirical scores, not population confidence bounds. Fresh
paired cases, optional statistical constraints and complete coverage remain
necessary. A previously selected parent may itself fail a historical floor on
new cases; retaining it is not a claim that the regression has been repaired.
Changing task meanings, score scales or the task catalogue requires an explicit
new research policy. Keep the lineage and its evidence available when reusing
historical retention views. This implementation does not silently relax floors,
add new task domains to a frozen campaign, or certify unseen dynamics.

The retained-observation regression study exercises the case in which no child
is promoted in the first round. The axis controller initially scores 1.0; fresh
second-round cases expose its limited action range and reduce its score to 0.5.
A vector calibration deliberately applied to both action contracts improves the
paired aggregate but still scores 0.5 on the axis task. The recorded first-round
assessment keeps the axis floor at 1.0 and
rejects this candidate. Discarding that assessment would lower the floor to 0.5
and incorrectly admit it. The diagnostic tie and global application are authored
controls; the vector coefficient is fitted from development transitions. Four
additional diagnostic trials replay the first round's two known axis cases:
the incumbent succeeds on both while the global repair fails both. These replays
verify actual old-case loss, distinguish it from harder fresh cases, and never
enter selection or proposals. They are explicitly not fresh held-out tests.

```bash
PYTHONDONTWRITEBYTECODE=1 python - <<'PY'
from tempfile import TemporaryDirectory
from PhysicalRSI_demos.retention_observations import run
with TemporaryDirectory(prefix="physicalrsi-retained-observations-") as workspace:
    result = run(workspace)
    print([row["outcome"] for row in result["research"]["rounds"]])
    print(result["quota"])
    assert run(workspace) == result
PY
```

Expected: two `retain_parent` decisions and 24 actual CPU trials: 20 for
development/paired validation and four for post-selection known-case diagnostics.
Both assessments and rejected
candidates remain auditable; independent restart adds zero trials. This study
does not repair the incumbent's range failure or run final test cases. Its
purpose is to demonstrate that retaining a parent cannot erase an observed
capability floor. Tests also cover interruption before/after the lineage pointer,
the outer commit and archive receipts, stale-parent rejection and damaged
historical evidence.

The CPU study starts with one fallback controller across scalar and vector
actions under standard and shifted actuator responses. An evolver fits only
failed development transitions, adds a drive-specific branch, then specializes
that branch for vector actions. The source harnesses, routing, candidate pool
and retention policy all pass the ordinary Self-Harness path under one trial
quota. A global repair is included as a forgetting control: it improves the
weighted aggregate score but destroys three earlier capabilities. The historical
gate rejects it even though the study deliberately allows ordinary per-round
regression, isolating the contribution of the retention gate.

```bash
PYTHONDONTWRITEBYTECODE=1 python - <<'PY'
from tempfile import TemporaryDirectory
from PhysicalRSI_demos.harness_tree import run
with TemporaryDirectory(prefix="physicalrsi-tree-") as workspace:
    result = run(workspace)
    print(result["final_test"]["metrics"])
    print(result["quota"])
    assert run(workspace) == result
PY
```

Expected: two inherited tree revisions, one unchanged round, 60 trials including
eight report-only tests, and 2/2 successes for each of four capabilities. Final
tests use unseen target magnitudes; standard-actuator tests also use a previously
unseen public label and exercise the fallback. This is not an unseen-physics
result. Completed campaigns audit evidence on restart without another reset.
The selected three-node tree and earlier two-node tree use normal snapshots and
rollback. Native routing, broader transfer, learned capability extraction and
open-world task-catalogue evolution remain research work.

## Optional training and checkpoint evolution

`checkpoints.CheckpointStore` owns content-addressed inference bundles and their
complete declared files. Model architecture, backend identity, operation input,
output and effects are fixed before training. `self_harness.training` provides
`CheckpointEvolutionSpec`, `TrainingCoordinator` and `CheckpointProposer`.
Ordinary `EvolutionSpec` still rejects weights mutations. The explicit model
specification changes only weights; foundation, runtime control and all other
components remain fixed. Inheritance rechecks this specification, including
checkpoint provenance. An imported seed is not a trained update; restoring a
verified ancestor is allowed.

System 1 loads the selected inference bundle through a caller-supplied backend
and records its checkpoint revision as a model dependency in each Trial.
The backend must own that frozen loaded version, including any remote inference
handle. It must not silently follow a mutable `latest` model. Full bundle bytes
are verified at binding; core does not hash a large model at every token. These
ports are trusted implementations, not a security sandbox or remote attestation.

System 2 freezes a planner result based only on verified development Trials.
The proposer owns a deep snapshot of both new and recovered planner replies
before subsequent callbacks. Candidate hypotheses and required checks therefore
retain the acknowledged plan even if a plugin later updates its reply object.
Planner identity and context are independently snapshotted before planning; a
planner that mutates its declared configuration is rejected before submitting
training, including when recovering an interrupted plan.
`DevelopmentExperience` retains the parent files and complete source evidence
independently of the original workspace; skill compilation shares this storage
contract. Training requires every source Trial to name the exact parent model.
A request records that experience revision, parent harness/checkpoint, fixed
model, provider, method (`rl` or `supervised`), parameters and update/sample/time
budget. A provider implements these explicit methods:

| Port | Contract |
| --- | --- |
| `identity()` | Stable implementation and execution configuration |
| `submit(request, output)` | Dispatch once and return a request-bound job handle |
| `recover_submission(request, output)` | Read-only lookup of the original submission after a lost reply; an unknown result cannot authorize resubmission |
| `poll(handle)` | Read-only status for that same job; terminal results require explicit quiescence |
| `collect(handle)` | Read-only completed inference bundle, file hashes, measured costs and `training_only` metrics |

Known queued/running jobs raise `TrainingPending`; the caller resumes the same
proposal after observing progress. Unknown jobs require reconciliation.
Submission and planner receipts prevent blind repetition. Completed checkpoints
own the training request, job handle, terminal status, output hashes and costs.
The coordinator owns deep snapshots of requests and provider replies before the
next callback. Later provider mutations of nested handle/status/result objects
cannot rewrite the provenance being published. Normal submission and recovery
after a lost reply both retain the values recorded in the durable job receipts.
Resuming a completed training workspace checks its submission, terminal and
collection receipts against the checkpoint's owned provenance without calling
the provider. These comparisons use canonical JSON identities, so boolean/numeric
substitutions in quiescence or measured-cost fields are rejected. Missing or
conflicting receipts stop acknowledgement; they never
trigger retraining or silently overwrite the conflicting evidence. Reading or
binding an owned checkpoint separately still works after source workspace loss.
Reported costs must fit the frozen budget; providers enforce actual process/GPU
limits and cancellation. Core does not infer spare cluster capacity or provision
instances. A failed, quiescent training job produces no candidate. Training loss
or reward never becomes fitness: every completed model is an ordinary candidate
requiring fresh paired validation and per-task nonregression before selection.

Run the standalone CPU acceptance in a new workspace:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m PhysicalRSI_demos.checkpoint_evolution \
  /tmp/physicalrsi-checkpoint-study
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  tests/test_checkpoint_evolution.py tests/test_skill_compiler.py
```

The demo starts one local worker process over `JobScheduler` with logical CPU
capacity one. A two-feature categorical policy samples actions on development
targets; its worker checks behavior probabilities, recorded actions and parent
attribution, then executes one REINFORCE gradient update per round. It learns
the scalar action first, then the structured vector action while retaining the
first capability. There are two training jobs and 52 CPU Trials: 32 development,
16 paired validation and four report-only final tests. Both final tasks succeed
on 2/2 new targets. Restart reads the same jobs and Trials; explicit lineage
rollback restores the preceding model and its actual inference behavior.
Fault tests cover lost replies, unknown workers with retained reservations,
wrong attribution, damaged/missing files, exceeded budgets, and misleading
training metrics that fail independent selection.

This is a small on-policy, finite categorical model, not LLM fine-tuning,
simulator/robot learning or a reproduced Agent Lightning/RLinf result. No training
framework is imported by core. The first checkpoint binding is a flat model
package; composing model updates with a Harness Tree needs its own declared
transition and acceptance. Distributed trainers, native learned improvements,
cross-family models and broad transfer remain separate work. Historical studies
must use their frozen source because implementation identities are part of their
receipts. Current source changes cannot silently resume older campaigns.

## Evidence read cost and integrity

Checkpoint ancestry and compiled skills may refer to many past Trials. Reopening
each small file at every wrapper layer made shared-filesystem metadata latency
dominate local CPU work. `ExperimentRuntime.read_evidence()` now returns the
receipt, parsed documents and raw evidence hashes from one audit. `Evaluation`
consumes that result. JSON semantics are checked against the same bytes that
were hashed; large binary artifacts are streamed once per Trial audit. The
publication receipt is rechecked after its dependent files, and artifact-manifest
hashes must agree with the receipt.

`DevelopmentExperience.read_with_record()` returns metadata and rehydrated
evaluations together. Reading owned experience no longer constructs each
Evaluation twice. Development feedback still binds every Trial to the exact
parent, split, evaluator and complete plan; auxiliary declared files also remain
subject to hash verification. Checkpoint ancestry reads each ancestor once and
requires one fixed model family throughout. Owned file enumeration uses one
directory scan per directory and rejects symlinks and non-regular files.

This reuse is confined to one read-only audit. There is no process-wide digest
cache, no timestamp shortcut and no reuse across an adapter call. Every new
public read checks the actual bytes again, including same-size modifications
with restored timestamps. Returned values describe verified bytes, not a lock
on a mutable filesystem: a later change cannot alter an already serialized
Evaluation and is rejected by the next read. Readers remain local POSIX tools,
not storage authentication or adversarial process isolation.

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  tests/test_evidence_audit.py tests/test_trial_contract.py tests/test_trial_artifacts.py
```

These checks cover actual file-open counts, binary corruption, JSON/receipt
replacement during reading, auxiliary evidence, and symlink/non-regular-file
rejection. Comparing reader implementations on the same retained experience
requires equal Evaluation digests, as well as elapsed time and file-open counts.
Read-only historical Trial evidence can be inspected independently; a frozen
campaign or checkpoint provider still requires its original implementation
identity to resume effects. Performance work must not rewrite historical source
or training receipts to make an old campaign accept new code.

## Isaac native acceptance path

```bash
./physicalrsi auto-engineering --core-research \
  --workspace /path/to/new-isaac-research \
  --isaac-python /path/to/isaac51/bin/python \
  --device-registry /path/to/shared-local-device-registry \
  --resource isaac-gpu-0 --episodes 3 --rounds 2
```

Use the same device registry and resource name for all cooperating clients on
the host. Resource names do not select CUDA devices or coordinate unrelated
processes. No cluster instance is provisioned by this command.

The native worker first constructs and settles the scene and publishes public
RGB-D detections and joint positions. It waits up to 120 seconds for complete
RGB-D and presentation frames, using render-only updates after settling; empty
annotator arrays during shader compilation cannot become reset observations.
The camera evidence records this warmup. Only then does the runtime dispatch the
System 1 pick/place plan. One Trial action represents that complete plan, not
one servo tick. The fixed Isaac controller executes at 60 Hz and records every
actual joint target and subsequent joint observation in `public-rollout.json`.
The runtime also retains images, depth, video, and private verifier evidence.
An independent adapter re-runs the task judge on measured final object states
and checks that the native plan and rollout match the dispatched action.

`RolloutProposer` supplies verified development Evaluation values to the
clearance evolver. Fresh paired validation compares parent and child; the next
round executes the selected revision. A frozen final candidate is evaluated
on report-only test seeds. The maximum reservation is
`rounds * (1 + 2 * episodes) + episodes`; unused validation reservations are
not consumed when no changed candidate exists. Completed trials resume by
reading evidence. A worker timeout stops its owned process group, retains the
unresolved receipt and device claim, and blocks automatic reexecution.

`PhysicalRSI_demos.auto_engineering.research.setup(..., evolver=provider)` accepts
another `RolloutEvolver`. The default only edits `memory_rules`; changing that
search space requires a new declared evolution policy. The suite, algorithm,
fixed source closure, splits and limits are recorded before effects. The native
interpreter and referenced simulator assets remain externally managed; this is
not a complete simulator installation or asset snapshot. Adapters are trusted
code, and private verifier files are not a security sandbox against a malicious
proposal provider.

Results are in `result.json`, per-round receipts and
`campaign/final-test/report.json`. The software contract tests use explicitly
named doubles and do not establish native simulator success. Native success
likewise establishes only the declared simulation-development scope, never
physical qualification or broad engineering generalization.

### Native development screening

Add `--staged` to the command above for a bounded native multi-candidate study.
Its pool contains one repair inferred from public development observations and
two authored negative controls with transit heights of 0.10 m and 0.20 m. The
controls test screening behavior; they do not demonstrate learned search.
Development seeds are `5000 + round_index`, validation seeds are
`20000 + 100 * round_index + case_index`, and final report-only seeds are
`30000 + case_index`. These cases are separate from the earlier acceptance below.
The complete split protocol is frozen before that round executes.

The first fidelity evaluates one native development layout per child; the full
fidelity uses all declared fresh validation layouts. A child must pass the
development check to enter validation. The incumbent is marked unmeasured at
screening and always remains available for full paired comparison. Add
`--screen-incumbent` with `--staged` to also collect its diagnostic screening
rollout. Screening scores cannot promote a candidate, and final
test outcomes do not feed proposals. If the repair also fails screening, no
improvement is inferred. Rejected candidates and their evidence are archived.

`TrialAdapters` binds each policy and verifier to its own environment. Native
development and screening own separate output directories even when they repeat
the same case and candidate. The durable local scheduler allows one native
worker at a time and shares the physical device registry. Its logical resource
capacity does not reserve GPU memory against unrelated processes. The command
does not provision instances. Full native evidence, interrupted-work recovery
and device release use the ordinary core contracts.

For this pool the maximum reservation is
`rounds * (1 + 3 + 4 * episodes) + episodes`, or one more screening trial per
changed round with `--screen-incumbent`; stages with no changed candidate
consume no screening or validation trials. With three validation cases, two
rejected controls avoid six validation trials while screening costs three extra
trials by default (four with incumbent diagnostics). That is an accounting
comparison for this declared pool, not a measured
speedup or proof that screening generally improves sample efficiency. Software
tests use a named contract double; actual native receipts are required to report
simulator outcomes.

### Observed native staged acceptance, 2026-10-13

The same CPFS Isaac installation completed 15 native trials with one local
worker at a time. Evidence, videos, archived candidates and independent restart
checks are retained at
`/cpfs02/user/yanming/physicalrsi-core-checks/native-staged-001`.
Its `source-manifest.json` SHA-256 is
`e0da755c0b821b0fd60104580a1fbdb3812a6181809b21c08a42ca8e3c916d7d`. Use that frozen
source to reopen the study. It predates the default children-only screening
optimization and therefore measures the incumbent at screening too.

| Stage | Policy | Successful trials |
| --- | --- | --- |
| Development, seed 5000 | Initial | 0/1 |
| Screening, same development seed | Initial / clearance repair / two low-clearance controls | 0/1 / 1/1 / 0/2 |
| Paired validation, seeds 20000–20002 | Initial | 0/3 |
| Same validation cases | Clearance repair | 3/3 |
| Next-round development, seed 5001 | Inherited repair | 1/1 |
| Report-only test, seeds 30000–30002 | Frozen final repair | 3/3 |

The decisions are `inherit_child` and `no_changed_candidate`. Both authored
negative controls were rejected before full validation and retained in the
archive with their screening evidence. Six scheduler jobs completed, all local
worker reservations and device claims were released, and independent restart
created zero workers or trials while preserving the result and evidence files.
These assertions are bound to raw Trial/Evaluation reads in
`acceptance-audit.json`, `restart-audit.json` and `validation.json`.

The repair uses public development RGB-D observations and an authored clearance
rule; this is not a learned proposer or model update. Previously observed failed
seeds 1000 and 100001 were excluded from this fresh protocol, so the study does
not demonstrate that those failures were repaired. Scores on these new cases
must not be treated as improvement over the earlier study's different cases.
This is bounded simulation evidence, without broad task or physical qualification.

The newer children-only screening policy is separately checked by native
contract doubles and a CPU comparison with a common 26-trial cap: 26 trials
without screening, 24 with full-pool screening and 22 with children-only
screening. All three CPU arms select the same candidate, retain the same paired
parent/child scores and final metrics, and restart with zero new trials. Their
record is at
`/cpfs02/user/yanming/physicalrsi-core-checks/screening-budget-001/validation.json`.
Those CPU savings are not a measured native speedup.

### Observed local native acceptance, 2026-10-13

An isolated Isaac Sim 5.1.0.0 / Python 3.11.17 installation on cpfs02 executed
11 native trials on one RTX PRO 5000 Blackwell GPU. Evidence is retained outside
the repository at
`/cpfs02/user/yanming/physicalrsi-isaac51-p0/acceptance-001`; installation and
renderer/physics preflight records are in its parent directory. The frozen
`source-v002/source-manifest.json` SHA-256 is
`58532a2bf46a16a6853381b6fc967cf9b7340c4df3e561e3d7c61efd10bab92b`.
This snapshot predates the candidate archive integration; use that frozen
source to reopen its receipts. New source identities require a fresh campaign.

| Stage | Policy | Successful trials |
| --- | --- | --- |
| Development, seed 17 | Initial, 0.13 m transit | 0/1 |
| Paired validation, seeds 1000–1002 | Initial | 0/3 |
| Same validation cases | Clearance candidate, 0.36 m transit | 2/3 |
| Next-round development, seed 17 | Inherited candidate | 1/1 |
| Report-only test, seeds 100000–100002 | Frozen final candidate | 2/3 |

The two recorded decisions are `inherit_child` and `no_changed_candidate`.
The evolver changed only the declared memory rule, using development RGB-D
evidence; selection used fresh paired validation. Reopening the workspace
returned the identical result, kept the trial reservation at 11 and created no
new native workers. All device leases were released. The parent directory's
`acceptance-001-audit.json` binds these checks to the retained reports.
One validation case and one
test case still failed object-placement or disturbance checks. This is a small
local control experiment, not evidence of a generally capable engineering
agent, model-driven improvement, statistical generalization or physical
qualification. The native task scores are distinct from the software contract
tests, which use doubles.

The numerical calibration CPU example proves execution/proposal/evaluation/selection integration on
CPU tasks. It does not reproduce GEPA or DGM, train weights, use an LLM, prove
statistical generalization or qualify a simulator/robot. Model-driven native
gains, search ablations and transfer remain separate acceptance work.

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  tests/test_core_versions.py tests/test_trial_contract.py tests/test_rollout_evolution.py
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/test_task_case_identity.py
```

## Development feedback parameter search

`CandidateArchive.development(view)` audits the frozen archive view and returns
owned common Trial/Evaluation observations from development and screening.
Each observation is bound to its candidate closure, task, case, scope and archived
receipt. Validation and final-test evaluations are excluded. Rejected screening
probes remain usable observations after their original execution folder is lost.
Legacy archives without common Trial receipts return no observations.

`ScalarRefinementEvolver` is an optional task-independent numerical search port.
A `ScalarParameter` declares one JSON field, its mutable component, bounds,
precision and direction. A task adapter classifies development observations as
`satisfied`, `insufficient` or `unresolved`. Search combines only the same task,
case, scope, other controller inputs and verified execution conditions. The shared
`feedback_context.measurement_context()` verifies the bound experiment file against
its evidence digest, Trial and receipt, then identifies environment/verifier/runtime
adapters, execution budget, embodiment and contracts, artifact contract and policy
lifecycle. Policy parameters and the System 1 revision can vary within the declared
search space. Evidence paths and System 2 stage labels are not directly included;
paths explicitly declared inside adapter identities remain bound. Different
budgets remain different conditions, even when
both executions terminated within those budgets.

Scalar refinement computes bracket midpoints with exact-ratio averaging before
rounding to the declared grid. Finite same-sign bounds cannot overflow during
their addition and falsely report `declared_bound_exhausted`. Increasing,
decreasing and cross-zero intervals retain the same directional and grid rules.

Each current case freezes one measurement context. Duplicate current cases or
unbound/changed experiment specifications fail before a proposal is published.
Historical conditions that differ are recorded with
`reason="measurement_context_changed"` and excluded before the feedback classifier
is called, so their failures cannot exhaust a current bracket or create a false
contradiction. Compatible screening feedback remains usable across System 2 stage
labels. This is exact declared-condition matching, not automatic calibration or
a guarantee that an adapter truthfully declares all external dependencies.
Both scalar and adaptive evolver identities bind the shared helper implementation.
Old campaigns still require their frozen source; existing evidence is not rewritten.

Every used Trial, receipt identity, per-case measurement context and excluded
condition is recorded in immutable `search-fit.json`. It advances beyond failed values toward
the observed success bound or declared domain limit. Contradictory per-case
feedback, exhausted bounds and unclassified failures produce no candidate.
Publication is atomic and repeatable after interruption; the committed parent is
unchanged until normal admission, fresh paired validation and selection succeed.

The monotonicity assumption, mutation domain and feedback classifier are authored.
This is an empirical search algorithm, not a model update or a causal guarantee.
The CPU acceptance executes Newton iterations for two square-root tasks: a
five-iteration probe fails screening, its archived residuals drive a seven-iteration
proposal in round two, and normal validation determines inheritance. Run it in a
new temporary workspace:

```bash
python -c 'from PhysicalRSI_demos.parameter_refinement import run; print(run("/tmp/physicalrsi-parameter-study")["quota"])'
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/test_parameter_search.py
```

For Isaac, `--core-research --feedback-search` uses the same core with carrying
height bounded to 0.10–0.55 m. The adapter requires both a grasp-like aperture in
phase 3 of the actual joint trace and independently measured obstacle disturbance
before classifying a failure as potentially insufficient clearance. That is a
hypothesis to test; failures from missing grasps remain unresolved.
The protocol repeatedly uses known historical failure seed 1000 for development,
seeds `40000 + 100 * round_index + i` for fresh validation, and `50000 + i` for
report-only final testing. Seed 1000 is an explicit regression case, not held out.
The default two rounds and three evaluation cases reserve at most 19 trials;
one child is screened per round before any full paired comparison. This mode
implies staged evaluation and remains serial on one leased native worker.

### Observed feedback-search acceptance, 2026-10-13

The local study at
`/cpfs02/user/yanming/physicalrsi-core-checks/native-rollout-search-001`
executed 12 native Isaac trials from frozen source manifest
`16edef32a3c516158350dfbba81d3a160220210720521b88d3fa371529e20711`.
The implementation was pushed in commit `76466b6` after 564 software tests,
compile checks and both public CLI smoke commands passed. Use the retained
frozen source to reopen this experiment.

| Stage | Policy | Successful trials |
| --- | --- | --- |
| Development, known failure seed 1000 | Initial, 0.13 m transit | 0/1 |
| Development screening, same seed | Proposed 0.34 m transit | 1/1 |
| Paired validation, seeds 40000–40002 | Initial | 0/3 |
| Same validation cases | Proposed candidate | 3/3 |
| Second-round development, seed 1000 | Inherited candidate | 1/1 |
| Report-only test, seeds 50000–50002 | Frozen final candidate | 2/3 |

The two decisions were `inherit_child` and `no_changed_candidate`. The first
native proposal passed screening, so this native run did not exercise reuse of
a rejected probe. That mechanism was measured separately with two CPU numerical
tasks: access to rejected development feedback produced iteration budgets
`5 -> 7`, whereas the current-feedback-only ablation repeated `5 -> 5`.
The former passed both final CPU cases; the latter failed both. Both arms used
the same two-round protocol and 30-trial limit, consuming 18 and 10 trials
respectively because rejected pools skip paired validation. Independent CPU
restarts created no new trials and preserved all evidence.

The native driver stopped after nine completed trials. Process, device and
receipt reconciliation verified that both research rounds were complete and
final testing had not started; recovery ran only the three untouched test cases.
A subsequent independent restart forbade native worker launch, returned the
same result, verified 2,237 unchanged files and created zero trials or workers.
All three queued jobs completed and all local resource/device claims were
released. `acceptance-audit.json`, `resume-reconciliation.json`,
`restart-audit.json` and `final-review.json` retain these checks.

Test seed 50001 failed placement and obstacle-disturbance checks. This is
bounded simulation evidence, not broad generalization or physical qualification.
The search domain, classifier and monotonicity assumption remain authored.
The inspected failing test may inform a future development protocol, but it
must then be excluded from claims of fresh held-out evaluation.

### Observed execution-context acceptance, 2026-10-13

`scalar-feedback-context-001` reproduced a scalar-search defect with executed
CPU Newton Trials. The current controller requested one iteration. Historical
controllers requested five iterations in the compatible/budget/binding cases,
or nine iterations under a capped environment or stricter verifier. Every
historical Trial failed its own declared check. The former scalar implementation
pooled all those observations into the current bracket:

| Historical measurement conditions | Former next proposal | Guarded next proposal |
|---|---:|---:|
| Identical conditions, five-iteration failure | 7 | 7 |
| Environment executes at most one requested iteration | None: bounds exhausted | 5 |
| Verifier uses a tighter residual tolerance | None: bounds exhausted | 5 |
| Different declared execution budget | 7 | 5 |
| Different declared embodiment revision | 7 | 5 |

The guard excludes incompatible history before classification, leaving the
current failure to define a new probe. These values demonstrate corrected
inference, not that the five-iteration probe already solves the task. The
ordinary two-task campaign still refines five to seven iterations using a
compatible rejected probe, consumes 18 Trials, and succeeds on both independent
final cases. Reopening it adds no Trials or actions and preserves all 396
files' bytes and modes.

An independent audit reconstructed all 30 CPU Trials across the reproductions
and campaign. Five adaptive-search replays on identical retained inputs produced
identical fit fields and proposal vectors apart from the declared implementation
identity. Three previously verified native development/screening receipts also
remain comparable: their original first-round proposal and second-round stopping
decision are unchanged. This is read-only replay; 77 referenced native evidence
files remained unchanged and no new simulator episode ran. It does not repair
or reclassify the previously observed final failure on seed 50001.

All 692 tests passed. Frozen baseline and candidate source, bound Trial evidence,
native export, replay results, verified archives and the acceptance audit are at
`/cpfs02/user/yanming/physicalrsi-core-checks/scalar-feedback-context-001`.

## Adaptive parameter search

Both scalar refinement and adaptive parameter search compare canonical JSON
identity digests against their frozen configuration. Type changes in classifier
or objective settings are rejected before historical feedback is read or a fit
is published, even when Python considers the old and new values equal.

`ParameterSpace` declares up to 32 bounded numeric JSON fields across mutable
harness components. Fields must have existing keys, non-overlapping paths,
exact component ownership and a declared precision grid. Foundation and weights
remain fixed. Quantization clips finite unit coordinates to `[0, 1]` and
interpolates from the nearer endpoint, preserving declared bounds even when
their magnitudes differ enough for floating-point subtraction to lose precision.
Candidate publication verifies the entire closure, synchronizes
files and atomically publishes each immutable candidate; an interrupted proposal
can resume the same frozen fit without duplicate execution.

`AdaptiveParameterEvolver` reads actual current development evaluations and
`CandidateArchive.development(view)`. An explicit objective adapter returns a
finite score in `[0, 1]` or `None`. Unknown/invalid outcomes are not scored as
failures. The current task/case/scope panel defines the comparison set: each
parameter vector must have definite scores covering that entire panel before
it can enter the elite set. Replicates are averaged within a case, cases within
a task, and tasks by predeclared weights. Duplicate receipt observations do not
increase weight. Incomplete vectors and changed controller conditions are
recorded separately in the fit.

Observation compatibility checks the experiment specification bound to the
Trial receipt. Environment and verifier identities, runtime implementation,
execution budget, embodiment/System 1 contracts and dependencies must match.
The varying policy adapter's identity and System 1 revision are omitted from
this context because the parameterized harness closure is checked separately.
The frozen suite identity and admission remain responsible for attesting that
policy construction follows that closure. These are trusted adapter boundaries,
not a sandbox or proof of external-provider equivalence. Different fidelity
budgets or dynamics cannot silently train one shared sampling distribution.

At each round the algorithm ranks complete observed parameter vectors, estimates
an elite mean and diagonal spread, and proposes a bounded population. A fixed
fraction samples the whole domain; the remainder samples near the elite set,
with a declared minimum spread. Quantization respects the declared field grids.
Local sampling also requires `minimum_observations` distinct, complete parameter
vectors and an observed difference between their aggregate scores. The count
threshold defaults to `1`; increase it explicitly, or pass `None` to use
`2 * dimensions + 1`. While fewer complete vectors are available, or all their
scores are equal, the whole population explores globally. A flat score panel
does not provide an evidence-based direction for local search. All vectors tied
at the elite cutoff are included, rather than selecting ties by coordinate order.
The declared minimum spread also applies when only one elite is available.
Optional `elite_weighting="fractional"` shares the remaining top-rank slots
equally among observations tied at the elite cutoff. For scores `[1, .2, .2]`
and an elite fraction of `.5`, normalized weights become `[.5, .25, .25]`
instead of three equal weights. Weighted moments apply to both diagonal and
full covariance; equal-weight cases retain the original fit arithmetic. The
default remains `uniform`. This is an experimental proposal policy, with no
measured task-performance advantage established yet.
Proposal random streams bind the declared sampler seed, round, task cases and
controller condition. Runtime locations embedded in measurement identities no
longer perturb those streams. The complete measurement context still gates
which historical feedback may enter fitting; shared random draws do not make
incompatible feedback reusable. Fits record `sampling_panel` separately.
Within-case repeat averages and within-task case averages use exact-ratio
accumulation before conversion back to a float. Identical scores remain equal
when repeat counts differ, and archive iteration order does not alter those
averages. This removes numerical pseudo-contrast; it does not estimate uncertainty
or make genuine small score differences statistically significant.
Repeated receipts, partial task panels and incomparable observations
cannot satisfy this threshold. This is an explicit exploration heuristic, not
a confidence bound or a guarantee of geometric coverage. Every fit records
`model_ready`, `score_contrast`, `elite_cutoff` and the effective threshold;
changing the configuration requires a new run because
it is part of the evolver identity.
Archived cases outside the current development panel are explicitly listed in
`excluded` with reason `outside_current_development_panel`, their task/case,
Trial and receipt hashes, and archive origin. They never reach the objective or
count toward model readiness. These early exclusions omit parameter values and
measurement context because those are not inspected for an inapplicable case.
This makes a return to global warmup after a case-panel change auditable.
The optional `covariance="full"` mode preserves empirical correlations between
elite parameters. It keeps the diagonal mode's marginal spreads, shrinks
off-diagonal covariance terms by a fixed 10%, and samples through a Cholesky
factor. The shrinkage and minimum spread keep collinear or small elite sets from
collapsing the proposal distribution. Fits record the covariance matrix and its
factor. Global exploration, warmup, precision grids, complete-panel requirements
and selection rules are unchanged. `covariance="diagonal"` remains the default;
correlation estimates from a small development sample are not confidence bounds.
Already observed vectors are skipped, and a fixed attempt budget prevents
unbounded retries on a coarse or exhausted grid. A deterministic random seed,
ranked observations, exclusions, fitted distribution and proposed vectors are
stored in `adaptive-fit.json`. `strategy="uniform"` provides a control using the
same space, case protocol, population and quota. No monotonicity assumption,
LLM call or model checkpoint update is required.

The CPU study in `PhysicalRSI_demos.adaptive_control` executes a proportional/
derivative controller under two different load dynamics. Trials integrate motion
with force saturation, measure trajectory error and overshoot, and screen on
settling success. Development includes positive/negative moves and two load
scales. Validation and final tests use separate moves and load scales. Parameter
scores shape proposals; fresh paired validation decides inheritance. This is a
bounded software dynamics study, not simulator or robot qualification.

```bash
python -c 'from PhysicalRSI_demos.adaptive_control import run; print(run("/tmp/physicalrsi-adaptive-control", strategy="adaptive", seed=0)["quota"])'
python -c 'from PhysicalRSI_demos.adaptive_control import run; print(run("/tmp/physicalrsi-uniform-control", strategy="uniform", seed=0)["quota"])'
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/test_adaptive_search.py tests/test_selection_order.py
```

The multi-task study also exposed order-dependent selection receipts: rejection
lists differed after canonical JSON reordered task or cost-limit keys. Survivor
selection now uses a canonical task/measurement order for those lists and task
aggregation, allowing the same complete paired evidence to survive commit and
restart verification.

### Observed adaptive-search comparison, 2026-10-13

The preregistered study at
`/cpfs02/user/yanming/physicalrsi-core-checks/adaptive-parameter-search-001`
used source manifest
`ca8ea33d923a35f2531db2fd0a315bc7c968c201ae3ccaba344bc522be4a2f98`.
All 605 software tests, compile checks and both public CLI smoke commands passed.
Each strategy used seeds 0, 1 and 2, four rounds, six candidates per round and
a 228-Trial cap per seed. The first three global exploration vectors matched
between strategies in every round, and all preregistered case panels matched.

| Seed | Adaptive final quality | Uniform final quality | Adaptive Trials | Uniform Trials |
| --- | --- | --- | --- | --- |
| 0 | 0.8297 | 0.8222 | 136 | 156 |
| 1 | 0.7639 | 0.7870 | 144 | 160 |
| 2 | 0.8074 | 0.8414 | 176 | 152 |
| Mean / total | 0.8003 | 0.8169 | 456 | 468 |

Final quality averages the measured quality across both tasks and their two test
cases; higher is better. Both strategies passed all 12 final cases across the
three seeds. Actual Trial use differs because screening skips validation for
rejected pools. These are descriptive results from three seeds, not a significance
claim or an equal-consumption comparison. Adaptive search did not outperform
uniform search overall and remains an opt-in research algorithm. The study
supports its feedback, provenance and recovery behavior, not a performance gain.

An independent audit reconstructed all 924 trajectories from recorded gains,
actions and dynamics and recomputed every outcome and quality measurement.
Six independent restarts prohibited environment reset, created zero Trials and
preserved 16,442 evidence files. All 201 scheduled jobs completed and released
their resources. `acceptance-audit.json` and `final-review.json` record these
checks. The source and six byte-verified artifact archives are retained on CPFS;
reopening archived runs requires restoring their original absolute roots.
Final tests did not inform proposals. Future algorithm comparisons require a
new preregistered protocol and untouched final cases.

### Observed warmup comparison, 2026-10-13

The follow-up at
`/cpfs02/user/yanming/physicalrsi-core-checks/adaptive-warmup-001`
preregistered seeds 10–19 and three arms: a five-vector warmup, the previous
one-vector adaptive policy, and uniform search. Each arm retained four rounds,
six candidates per round and a 228-Trial cap. Development, validation and final
panels were new; the audit checked that neither their reset identities nor their
effective displacement/load pairs repeated the preceding study. The first three
global vectors matched across all arms in every round; warmup and uniform also
shared all six first-round vectors.

| Strategy | Mean final quality | Final successful cases | Actual Trials |
| --- | --- | --- | --- |
| Five-vector warmup | 0.8274 | 40/40 | 1,796 |
| Previous adaptive policy | 0.8263 | 40/40 | 1,684 |
| Uniform search | 0.8365 | 40/40 | 1,640 |

Warmup's paired mean quality difference against the previous policy was +0.0011,
with three wins, four losses and three ties across seeds. It used 112 additional
Trials. Against uniform search the difference was -0.0090, with two wins, four
losses and four ties. These descriptive results do not support changing the
default sampling policy; `minimum_observations=1` remains the default and warmup
is explicitly opt-in. Neither adaptive variant demonstrated an overall advantage
over uniform search on this software workload. Actual consumption differs because
screening determines which candidate pools receive paired validation.

The retained experiment source manifest is
`a7e71dbbb4301807b7c0748492ecbecf8742aadb3e5fde58206dc3f01d57de3c`.
The study explicitly supplied every arm's threshold. The release preserves the
one-vector API default; its function bodies match the measured implementation.
Both source versions and the default-only change audit are retained separately.
All 5,120 trajectories were reconstructed independently and all outcomes/scores
recomputed. Thirty independent restarts prohibited reset and created zero Trials.
All resource claims were released. `acceptance-audit.json`, `final-review.json`
and the byte-verified CPFS artifact archives retain the evidence. Ten seeds and
two software load tasks do not establish broad or native performance, and these
final cases must not be reused as fresh held-out evidence.

### Observed covariance comparison, 2026-10-13

The preregistered `adaptive-covariance-001` study used seeds 20–29, four rounds,
six candidates per round, a five-vector warmup and a 228-Trial cap per arm.
Development, validation and final panels differed from both prior studies,
including effective displacement/load combinations. Correlated sampling kept
the first global exploration points and initial warmup population identical to
the controls. Every comparison used the same case plans and selection rules.

| Strategy | Mean final quality | Final successes | Total Trials |
| --- | --- | --- | --- |
| Diagonal adaptive | 0.825627 | 40 / 40 | 1,828 |
| Full covariance | 0.826192 | 40 / 40 | 1,836 |
| Uniform | 0.826769 | 40 / 40 | 1,584 |

Full covariance exceeded diagonal sampling by only 0.000565 in mean quality,
with two seed wins, two losses and six ties, consuming eight additional Trials.
It was 0.000577 below uniform sampling and consumed 252 more Trials. This meets
the preregistered condition for retaining an optional mode, but establishes no
practical or statistically supported performance advantage. Diagonal sampling
and the one-observation threshold remain the defaults. The uniform control
remains essential for assessing whether feedback-guided sampling earns its cost.

All 665 tests passed. Independent replay reconstructed 5,248 trajectories and
recomputed their scores. It also reproduced the existing diagonal/uniform
sampling sequence from stored fit inputs and checked the full covariance factors.
All 30 independent restarts added zero Trials/actions and preserved 90,377 files
and their modes. The 1,162 scheduled jobs completed with no retained resource
claims. Frozen sources, complete reports, audit scripts and 30 byte/mode-verified
artifact archives (154,814,518 compressed bytes) are retained at
`/cpfs02/user/yanming/physicalrsi-core-checks/adaptive-covariance-001`.
This is descriptive evidence from one software dynamics family, not native
Isaac improvement, broad transfer or physical qualification. These final panels
are now observed and must not be reused as untouched tests for later tuning.

### Native two-parameter adaptive search

`PhysicalRSI_demos.auto_engineering.empirical.setup` connects the common
`AdaptiveParameterEvolver` to native Isaac Trials. It searches carrying height
(0.10–0.55 m) and grasp depth (0.15–0.40 of the declared sample height). Existing
policies without the new depth field retain the 0.25 default. Perception,
controller code, task geometry and the seven intended-task checks stay frozen.

```python
from PhysicalRSI_demos.auto_engineering.empirical import run
result = run(
    workspace, isaac_python, strategy="adaptive", seed=20261013, population=2,
    panels={"development": [1000], "validation": [[71000], [71100]], "test": [72000]},
    device_registry=shared_device_registry, resource="isaac-gpu-0",
    max_replacements=1, extra_trials=1,
)
```

Declare panels before running. Development can be reused across rounds; validation
panels are disjoint by round, and final seeds are withheld. Only completed
development measurements reach the search objective: success scores 1, and a
failure scores 0.9 times the fraction of satisfied checks. Uncertain/invalid
outcomes are excluded. Whole comparable case panels, at least three parameter
vectors and observed score contrast are required before elite proposals begin.
Uniform mode uses the same parameter bounds, screening and full-validation gates.
Screening only admits candidates; full paired validation controls inheritance.

The returned campaign exposes `status`; final results remain separately under
`final_test`. Search diagnostics are retained as `adaptive-fit.json` in each
round. The archive carries the measured proposal history into subsequent rounds.
Both fidelity stages receive the declared replacement allowance, with additional
attempts charged to the shared quota. Use the explicit screening or full-validation
recovery API for the interrupted stage; no automatic retry is enabled.

Contract-double tests cover two-round feedback reuse, independent cases, bounded
parameters, restart without repeated effects and a screening startup replacement.
They do not establish native performance or an advantage over uniform search.
A native study must retain its frozen implementation, declared panels, raw rollout
receipts and independently checked final measurements before making such claims.

`shared_cache=True` is an opt-in startup optimization on both native research
entry points. It gives all rounds and fidelity stages one workspace-local
`native-cache` for Kit, CUDA, GL, Warp and other XDG cache data. Each worker keeps
its own portable logs/data, configuration, temporary files, IPC and episode
artifacts. The normal shared-device lease and one-worker scheduling still apply;
this option does not reuse simulator processes or replay task observations.
Direct `Suite` users can declare `cache_root` explicitly. Configure it before
starting a workspace: existing local caches and conflicting links are rejected,
not overwritten or migrated. The default remains isolated caches.

This option has filesystem, adapter-contract and bounded native coverage.
The completed two-round study below used the original isolated-cache
configuration. A separate two-Trial shared-cache probe completed successfully;
its observed timing is reported separately below.


### Observed native adaptive-search closure, 2026-10-13

`/cpfs02/user/yanming/physicalrsi-core-checks/native-empirical-search-002`
completed with the frozen native adapter published in `da7fb89`. Its source
manifest SHA-256 is
`65394935919666e7f5fc2617e5e5426374c8cdaa2245f3056bd3493116fa4d9e`.
The study declared development seed 1000, validation seeds 2026101301 and
2026101302 in separate rounds, and final seed 2026101303 before execution.
Sampler seed 20261013 generated two proposals per round.

| Stage | Measured result | Decision |
| --- | --- | --- |
| Round 1 | Initial parameters `(0.13, 0.25)` failed; global proposal `(0.422, 0.309)` passed screening and the fresh paired case. Global proposal `(0.173, 0.171)` failed screening. | Inherit `(0.422, 0.309)` |
| Round 2 | Inherited parameters succeeded again on development. Global proposal `(0.447, 0.172)` passed screening and validation, tying the incumbent. Elite proposal `(0.258, 0.219)` failed screening. | Retain incumbent; no additional gain |
| Final test | Selected `(0.422, 0.309)` passed the one held-out case and all seven physical checks. | Report only; no further selection |

The tuples are carrying height in metres and grasp-depth fraction. Round 2 fitted
its proposal distribution using four verified development observations across
three parameter vectors, including the rejected first-round candidate. The
feedback audit matched each sample to its audited Trial and receipt identity;
no validation or test observation was used for fitting. The two successful
observations of the incumbent were grouped as repeat measurements of one vector.

There were **11 completed Trials: 7 successes and 4 failures**, **17,336 physical
steps**, and **11 charged reservations out of 14**. No replacement attempt was
needed. The two-round study and final report took about 40 minutes 36 seconds.
The shared device registry was empty after the driver exited. Raw receipts,
videos and measurements remain under the study workspace. `measurement-audit.json`
independently recomputed the intended-task checks from retained geometry for
every completed Trial; `interim-audits/round-2-feedback-audit.json` records the
feedback lineage, and `result.json` binds the final report and audit hash.

This establishes the native loop from proposal through screening, independent
validation, inheritance, archive-backed second-round fitting and final reporting.
The first improving candidate came from global warmup; the fitted second-round
proposals did not earn another promotion. One case per panel, one sampler seed
and no matched uniform arm do **not** establish adaptive-search superiority,
broad generalization, transferable skill acquisition or model-weight learning.


### Observed shared native cache, 2026-10-13

`/cpfs02/user/yanming/physicalrsi-core-checks/native-shared-cache-001`
ran two sequential development Trials with the same seed 1000, the same
`(0.422, 0.309)` policy, two different worker roots and one new shared cache.
The source manifest SHA-256 is `82103697a3d4e5720b0cac7b5225a944796dcb787928b5bc61307a3e1f64e5e2`.

| Measurement | First root, empty cache | Second root, reused cache |
| --- | ---: | ---: |
| Trial elapsed | 256.970 s | 175.838 s |
| Runtime reset phase | 214.991 s | 133.989 s |
| Runtime rollout phase | 40.499 s | 40.385 s |
| Runtime verification phase | 0.104 s | 0.101 s |
| Runtime finalization phase | 1.189 s | 1.175 s |
| Sensor-frame warmup | 65.528 s / 194 renders | 5.824 s / 17 renders |
| Shader-cache entries before launch | 0 | 61 |
| Task result | Success, 7/7 checks | Success, 7/7 checks |

Both Trials retained 1,576 physics steps, independently recomputed physical
checks, distinct IPC and evidence directories, and the same shared Kit cache
target. The two-reservation quota was fully charged, both worker processes
exited, and the shared device registry was empty afterward. `result.json`,
`cold-first-root.json` and `warm-second-root.json` bind the measured receipts,
worker logs, cache observations and phase timings.

The observed elapsed reduction was about 81 seconds (31.6%) in this pair.
This is one sequential cold/warm pair on one known case, with no randomized
control or repetition; OS caching and run-order effects are not separated.
It establishes successful cross-root cache reuse with lower observed reset cost,
not a general speedup guarantee. Shared caching remains explicitly opt-in.

### Software regression scheduling measurement

The same 792-test source at `77b4598` passed serial execution in 231.48 seconds
and two-process execution in 131.62 seconds with pytest-xdist 3.8.0 and execnet
2.1.1 (`-n 2 --dist=loadfile`). Both ran from immutable RAM source snapshots with
separate RAM temporary directories. All 224 tracked Python files in the core,
demos and tests matched the worktree; the available CPU affinity contained 12
CPUs. The observed reduction was 43.1%, without dropping tests or running Isaac.
This single comparison does not guarantee the same speedup on other machines
or shared-filesystem checkouts. Logs and the source manifest are retained at
`/cpfs02/user/yanming/physicalrsi-core-checks/core-parallel-check-001`.

The `test-parallel` extra makes this mode optional. Keep two workers as the
measured starting point, preserve per-worker temporary-directory isolation, and
inspect failures before changing concurrency or rerunning tests. Native device
leases, rollout quotas and physical acceptance scheduling remain separate.


### Interrupted matched native pilot, 2026-10-14

`/cpfs02/user/yanming/physicalrsi-core-checks/native-matched-bench-001`
compared adaptive and uniform search from the same initial harness, with two
rounds, population two, sampler seed 20261014 and a 22-Trial cap per search arm.
Development seed 2026101400, validation panels 2026101401–1402 and
2026101403–1404, and final test seeds 2026101410–1413 were disjoint. The
frozen source matches `5c5cbdb2edde21e4a82a03d2e03bb3ce020637dc` except for
local AGENTS.md instructions. Later sampling and elite-weighting changes were
not part of this study.

| Arm | Search Trials completed | Final test successes / planned |
| --- | ---: | --- |
| Fixed initial harness | 0 | 1 / 4 |
| Adaptive search | 10 | 4 / 4 |
| Uniform search | 12 | Unavailable: first test interrupted |

The adaptive final harness used lift height 0.418 and grasp width 0.250;
uniform selected 0.417 and 0.249. Both were promoted from global warmup.
Second-round fitted adaptive proposals did not earn further promotion.
The observed improvement over the initial harness is 75 percentage points on
four held-out cases; it does not establish adaptive superiority or generality.
The pilot's random stream included runtime identity, so arm-specific cache paths
also changed proposals despite the same configured sampler seed. Repeated
initial-harness checks varied between arms. These limitations prevent a clean
attribution to the search strategy even apart from the interrupted final panel.

At 2026-10-13 16:19:49 UTC, the driver terminated during uniform test seed
2026101410. The Python crash trace places the native failure in
`initial_sensor_frames -> world.render`, after the 90 settling steps and before
publishing the initial observation or dispatching the policy. Native frames
include `UsdGeomBBoxCache::SetTime`, USD stage loading and timeline callbacks.
This identifies the crash site, not its root cause. The receipt remains
`needs_reconciliation`; it is not a measured policy failure. The remaining
three uniform tests were not executed. The preregistered no-replacement rule
was preserved; the driver was not restarted.

There are 30 canonical completed Trials and one interrupted Trial. Archived
receipt copies must not be counted again. `interruption-audit.json` retains
state counts and hashes of the crash evidence. The two completed final reports
were joined back to candidate hashes, case hashes, receipt verdicts and all
seven physical checks by the read-only comparator. No complete three-arm result
or algorithm-superiority claim is available from this pilot.

### Native render diagnostic and failure reporting, 2026-10-14

`/cpfs02/user/yanming/physicalrsi-core-checks/native-render-diagnostic-001`
tested pausing the timeline during render-only sensor warmup in an isolated
source snapshot. It used a new development seed (2026101500), not a replacement
for the interrupted benchmark. The paused variant reached warmup but timed out
after 120 seconds with all three sensor arrays empty. Physics step 92 and time
1.5333334133028984 were unchanged across warmup. Initialization adds steps before
the explicit 90-step settling loop; absolute step 90 was an incorrect diagnostic
assertion, although execution failed before reaching that assertion. The original
variant was not dispatched because the diagnostic stopped on the first error.
This experiment rejects the paused variant for this path; it does not establish
the cause of the original native crash. No paused-render change was merged.

The diagnostic also exposed a reporting problem: the worker printed the Python
`TimeoutError`, but simulator shutdown returned status zero, leaving the parent
with only an exited-before-readiness error. The worker now atomically publishes
Python exception type and message in its owned IPC directory before shutdown.
The parent checks this failure before accepting readiness or successful exit;
unreported native failures include the process exit code. These failures remain
uncertain Trials requiring reconciliation, without automatic retries or policy
failure scores. Contract tests cover a real child process reporting an exception
then exiting zero, failure precedence over readiness, and an unreported signal
exit. They provide software evidence, not native crash-resolution evidence.

The follow-up `native-render-diagnostic-002` used Replicator's synchronous
`step(delta_time=0.0, pause_timeline=False, rt_subframes=4)` for sensor warmup.
On development seed 2026101503, two capture calls produced valid RGB, depth and
presentation frames in 65.95 seconds. Warmup retained physics step 92 and time
1.5333334133028984. The complete rollout used 1,576 physics steps and passed all
seven independently recomputed geometry checks; the resource lease was released.
This is one successful diagnostic, not evidence that intermittent USD crashes
are eliminated. The adapter now uses this static-capture path and rejects any
warmup capture that changes the physics step or simulation time before System 1
receives an observation. Camera evidence records the capture mode and settled
physics timestamp. The integrated implementation subsequently completed two fresh development
cases (2026101510 and 2026101511) in `native-static-capture-001`. Both passed
all seven independently recomputed geometry checks with 1,576 physics steps
and no change to settled physics during warmup. Warmup took 67.23 and 70.99
seconds, two static captures each. Both reservations were charged and the device
registry was empty afterward. The exact implementation also passed 890 software
tests in 169.18 seconds (`core-static-capture-001`). These acceptance results do
not replace the interrupted benchmark or prove intermittent-crash elimination.

### Second matched native pilot interrupted, 2026-10-14

`native-matched-bench-002` froze source `0abb6e162cedb218e877b34da1c0e69175d9d808`,
fractional elite weighting and new disjoint panels: development 2026101600,
validation 2026101601–1602 and 2026101603–1604, test 2026101610–1613.
Both first-round global candidates passed the development case and both first
validation cases; the initial harness failed all three. Core inherited
`(transit_height_m=0.383, grasp_depth_fraction=0.349)`, retaining the other
candidate and initial harness in the archive. An archive audit verified each
retained harness hash.

The second round fitted four completed development measurements across three
parameter vectors, including a repeated measurement of the inherited harness.
A receipt-level audit confirmed that no validation or test case entered fitting.
It proposed global `(0.292, 0.359)` and fitted `(0.428, 0.231)` candidates. The
global candidate passed development, but the fitted candidate never reached
policy execution: its native worker exited with SIGSEGV (-11) during the
90-step settling loop, before static sensor capture. The Python stack points to
`World.step -> SimulationContext.step`; native frames again include
`UsdGeomBBoxCache::SetTime`. Thus static capture did not eliminate the underlying
native crash, whose cause remains unresolved.

The frozen study stopped with 11 completed Trials (8 successes, 3 failures) and
one `needs_reconciliation` Trial. Uniform search and all final tests were not
executed, so no final benchmark score or strategy comparison is available. No
replacement was run. After confirming driver and worker-group exit and an empty
GPU compute-process list, the device lease was reconciled for resource reuse;
the uncertain Trial was not rewritten as a measured task failure. Evidence is
retained in `interruption-audit.json`, `device-reconciliation.json`,
`first-round-archive-audit.json` and `round-two-feedback-audit.json` under the
study root.

### Headless viewport diagnostic, 2026-10-14

`native-viewport-diagnostic-001` completed four fresh native Trials in ABBA
order (viewport off/on/on/off), with two paired development seeds
2026101700–1701, fixed memory and separate cold caches. All four passed seven
independently recomputed geometry checks and completed 1,576 physics steps.
Within each pair, layouts and policy memory were identical. Camera frames were
available and static sensor warmup preserved the settled physics timestamp.
`comparison.json` retains receipt hashes, checks and phase timings.

Viewport-off elapsed times were 231.75 and 229.18 seconds, versus 241.47 and
234.22 seconds with viewport updates enabled. The observed mean reduction was
7.38 seconds (about 3.1%); two pairs with runtime/cache variation do not establish
a general speedup or elimination of intermittent USD crashes. The native adapter
now disables viewport updates only in headless mode, retaining camera rendering
and GUI viewport updates. This follows Isaac Sim's documented headless
performance configuration. This diagnostic is not a replacement benchmark or
evidence of adaptive search outperforming uniform search.
Integration auditing confirmed the worker matches the native-tested viewport-off
variant apart from the diagnostic print and configuration key order. All four
sensor contract tests passed; the device registry was empty after the diagnostic.
