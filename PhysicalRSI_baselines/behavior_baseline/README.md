# BEHAVIOR baseline and RSI research

The target is a higher mean native benchmark score than the declared baselines
under comparable evaluation conditions, achieved through RSI iterations without
benchmark hacks. Installing the environment, completing a few tasks, or passing
software tests does not satisfy this target.

## Method

This implementation adopts [HarnessPAI](https://darwin-agent.github.io/HarnessPAI/)'s
separation between a fixed executable program within each rollout and program
revision between rollouts. Fixed programs may use observation feedback, perception,
learned action primitives, checks, and bounded recovery. No program rewriting occurs
during an episode. The public HarnessPAI repository currently contains its website
and method description, not the execution implementation. Its two BEHAVIOR pickup
tasks are methodological examples, not evidence of full benchmark performance.

The integration is being built on these existing physicalRSI components:

| Responsibility | Core component |
| --- | --- |
| Freeze reset identities and hold out report-only test cases | `CaseProtocol` |
| Execute a frozen controller and preserve native evidence | `ExperimentRuntime` |
| Evaluate paired parent/candidate trials under fixed budgets | `ExperimentEvaluator` |
| Admit, compare, select, and retain or inherit code | `SelfHarness`, `HarnessState` |
| Bound repeated experiments | `TrialQuota`, `ImprovementCampaign` |
| Store scoped failure/repair hypotheses and counterexamples | `ResearchMemory`, `ResearchCampaign` |

`protocol.py` currently implements the comparison binding and complete-cohort
aggregation using core storage and case protocols. It rejects reset overlap,
changed comparison conditions, missing trials, duplicates, unresolved outcomes,
and nonfinite scores. Its aggregation helper expects verified receipt references;
a hash alone does not establish evidence authenticity or policy compliance.
`verified_native_row` reads hash-bound native evidence against an independently
preregistered launch specification and verifies the candidate/task handshake and
native instance identity before constructing an aggregation row. Smoke policies
and shortened episodes are excluded. Callers still own asset verification,
policy admission, and binding each launch specification to the comparison.
`native.py` launches the unmodified native evaluator in a separate process with
a deadline, durable run specification, hash-bound logs/results/videos, and
explicit reconciliation after interruptions. Reusing a completed run verifies
its evidence instead of repeating simulator effects. This transport is not yet
the per-step `ExperimentRuntime`/`ExperimentEvaluator` adapter. Instead,
`evaluation.NativeEvaluator` supplies the core Self-Harness evaluator port using
complete native episodes. Its software integration is tested. Reviewed admission
and a prepared, development-derived proposal now drive an actual paired radio
round. Native selection retained the parent after zero measured gain; full-task
campaign performance remains unverified.

Native launches can now accept the core `System2Trial` attribution contract.
Their persisted case includes the task name, so identical instance numbers in
different tasks have distinct cohort identities. `bridge.py` verifies complete
native validation receipts and converts them into the existing Self-Harness
selector's result format, preserving raw evidence and equal task weights. A
software fixture exercised the actual core paired selector and evidence-tamper
rejection. The native evaluator port connects fresh cohort registration, quota
reservation and policy-service orchestration. The reference and first recovery
proposal passed reviewed admission, and all four paired validation trials
completed with final Q=0. The core selector retained the parent.
The bridge alone does not establish a completed RSI campaign or score improvement.

`program.py` implements the frozen System 1 program port using core harness
verification and embodiment contracts. Reviewed `control.py` implements
`initialize(task)` and `step(observation, state, primitive)`. The fixed action
backend is bound by `backend.json` in the foundation closure. Changing program
files or backend identity during execution fails the episode. Every episode
loads a fresh module namespace and resets the backend's episode state.
This is reviewed-code execution, not an untrusted-code sandbox.

`observations.py` projects the pinned R1Pro interface to RGB, optional metric
depth, and the official OpenPI baseline's 25 proprioceptive coordinates. Raw
proprio vectors can contain privileged fields; only the declared base velocity,
torso, arm, and gripper coordinates reach a program. Other native fields and
metrics are discarded. The public projection and policy-service connection have
passed one shortened real-policy native rollout; this is a reviewed interface,
not process isolation against malicious policy code.
The pinned evaluator's default `r1pro.yaml` names the robot `robot_r1`, while
the OpenPI wrapper uses the older `robot` observation keys. The projection
accepts either complete namespace and restores the wrapper's canonical keys
without changing image or proprioceptive values. Ambiguous or mixed namespaces
are rejected. A real native policy request exposed this mismatch; the failed
rollout is preserved, and regression tests cover both names. The corrected
engine has a new seed identity; its native policy smoke completed cleanly.

`openpi_backend.py` loads the official BEHAVIOR OpenPI wrapper from the pinned
checkout and verifies every checkpoint file against a declared manifest. It
restores only the baseline's consumed public proprio indices; all unused vector
coordinates remain zero. The original image processing, gripper summation,
normalization and action chunking stay in the upstream baseline implementation.
An external software check compared the upstream image preprocessing and exact
state-extraction function against this projection on four randomized batched
observations; the selected inputs matched exactly. This is not model-inference
or task-performance evidence.

`serve.py` exposes a `ProgramPolicy` through the official OpenPI BEHAVIOR
websocket server. The handshake includes the candidate and policy identities.
It supports episode resets and returns the native 23-dimensional control vector.
Loopback handshake, NumPy encoding, two episode resets, and action response were
checked with the actual upstream websocket server and a software backend fixture.
Real-checkpoint startup and inference passed in the local OpenPI environment:
the engine-bound seed returned finite `[1, 23]` actions for synthetic onboard
observations. The receipt is `runtime/pi05-inference-probe-002/result.json`.
The unmodified native `WebsocketPolicy` client subsequently completed two resets
and 18 actions per reset against that real checkpoint, using the default wrapper's
camera profile and synthetic Torch observations. All 36 actions were finite and
had shape `[1, 23]`, including requests across the 16-step action-horizon boundary
(`runtime/native-client-inference-001/result.json`). The service was stopped and
reaped after the check. A separate zero-action native scene smoke has since
completed with a clean shutdown. The corrected engine subsequently completed
a real-policy native radio smoke: six steps under the five-step smoke cap,
final Q-score 0, and no task success, with verified JSON/video and clean shutdown
(`runtime/native-policy-preflight-001.json`). The service was stopped and reaped.
This establishes actual observation-to-action transport, not competitive task
performance; shortened episodes are excluded from baseline means. A subsequent
full-budget radio train-instance-0 development trial completed with 3,225 native
steps, final Q-score 0 and no task success. The 107.5-second onboard video and
native result were verified, and the reference dependency admission completed
(`runtime/full-budget-radio-baseline-001.json`). This is one development case,
not the 100-task baseline mean or evidence of RSI improvement.
Run it from the OpenPI environment with this repository on `PYTHONPATH`:

```bash
python -m PhysicalRSI_baselines.behavior_baseline.serve /path/to/candidate.json --port 8000
```

The candidate's `backend.json` declares `kind: openpi-b1k/v1`, `source_root`,
`source_revision`, `adapter_sha256`, `checkpoint_root`, a complete
`checkpoint_files` SHA256 mapping, `training_config`, `repo_id`, `task`,
`control_mode`, and `action_horizon`. Those fields belong to the frozen foundation
closure. They must describe actual downloaded files, not placeholder identities.

The pinned GR00T wrapper has also passed a separate CPU software-fixture probe:
reconstructing only the public proprioceptive coordinates produces exactly the
same consumed state and RGB inputs as the original observation; two episode
resets and 18 temporal-ensemble actions per episode return finite 23-dimensional
commands (`runtime/groot-wrapper-probe-002.json`). This probe uses no model
weights and establishes neither GR00T inference nor native qualification.
The upstream `serve_b1k.py` does not pass a task prompt, leaving the wrapper's
generic pickup-and-place default. A future adapter must explicitly bind the
actual task prompt in its frozen configuration. Authorized Cosmos processor
access and real-checkpoint inference remain prerequisites. Separately, the
pinned OpenPI task registry contains only the radio prompt; a full-task backend
needs an explicit prompt catalog as well as model and normalization coverage.

After the checkpoint is fully extracted and the policy environment is installed,
`seed.py` hashes its files and creates the initial pass-through program. Capture
the installed policy environment with `uv pip freeze --python <policy-python>`
in an external file, then run:

```bash
python -m PhysicalRSI_baselines.behavior_baseline.seed /path/to/seed-001 \
  --source /path/to/pinned/openpi --checkpoint /path/to/extracted/checkpoint \
  --task turning_on_radio --environment-manifest /path/to/policy-environment.txt
```

Use the resulting `seed-001/candidate.json` with `serve.py`. Existing revisions
are never overwritten. Materialization records actual checkpoint hashes, the
upstream lockfile and the supplied installed-environment inventory; it does not
establish model compatibility, dependency admission, or benchmark performance.
New seeds also bind the shared System 1 engine (observation projection, program,
backend, websocket service and core contract/storage/artifact code) in
`engine.json`. Loading and acting reject an engine mismatch. Earlier candidates
without this dependency require a newly materialized revision; never rewrite an
old candidate's closure or reuse its identity after an implementation change.

## Evaluation contract

The user selected the official BEHAVIOR Challenge task set and baselines. The
current official release is the 2026 Challenge (100 tasks); external reference
methods are the official π0.5 and GR00T N1.7 pipelines. Freeze the complete task list from the official task
metadata before comparison. Use train instances for development and disjoint
validation; reserve the public test cohort for the final frozen comparison.
Never adapt against that final cohort. Do not use hidden evaluation assets.

`cases.py` verified the official task-instance archive against its Hugging Face
LFS SHA256 and registered all 100 tasks with core `CaseProtocol`. Per task, the
first 200 available training IDs form the development pool, the remaining 100
form the RSI validation pool, and public instances 301–320 form the public
inventory. Fifty tasks have training IDs 0–299 and fifty have IDs 1–300; the
inventory uses actual filenames. Validation is held out from RSI development,
not necessarily from the original baseline's training data. These are case
pools; each round must preregister its finite, paired validation cohort and
reserve its trial budget before execution.

```bash
python -m PhysicalRSI_baselines.behavior_baseline.cases \
  /cpfs02/user/yanming/behavior_runtime/archives/2026-challenge-task-instances.zip \
  /cpfs02/user/yanming/behavior_runtime/challenge-protocol-001
```

This inventories task and instance identities without reading test scene state.
It does not run 32,000 episodes or prove the asset installation works.

The official reported cohort is narrower than the public inventory: use indices
0–9 (instance IDs 301–310), one rollout each, for all 100 tasks: 1,000 episodes
per policy. `reporting_cases.py` freezes that prescribed cohort in a separate
core protocol and records the excluded indices 10–19. It preserves all training
splits and the original inventory, including any active engineering round that
already references it. Final full-task reporting must use the derived reporting
protocol, not average all 20 public instances. This registration precedes any
public-test execution and does not select cases from observed results.

```bash
python -m PhysicalRSI_baselines.behavior_baseline.reporting_cases \
  /cpfs02/user/yanming/behavior_runtime/challenge-protocol-001/cases/protocol.json \
  /cpfs02/user/yanming/behavior_runtime/challenge-reporting-001 \
  --expected-revision ab64d11e157d3a1dd55f3a9f99669f688e1f37cc2844d5faec9f5c35154b192b
```

The destination must be new. Registration validates complete coverage and exact
public identities without opening scene state. Official reporting requirements
are documented in the [submission rules](https://behavior.stanford.edu/challenge/submission.html).

Primary metric: native final Q-score, averaged across instances per task and then
equally across tasks. Also report task success, efficiency, runtime, and trial
budget. Incomplete experiments must be reconciled before reporting a complete
mean. Failed policy episodes remain in the denominator. Infrastructure errors
are not policy failures. No task cherry-picking or successful-only aggregation.

Baseline and evolved programs must share evaluator/assets, robot, camera profile,
action interface, episode limits, instances, and repeat counts. Freeze baseline
code and checkpoint identities. Compare against both the initial harness and
the declared external baselines; do not present zero-action smoke tests as a
competitive baseline. Use paired validation, task regression limits, and a final
held-out comparison with uncertainty estimates before claiming improvement.

Policy inputs follow the [official rules](https://behavior.stanford.edu/challenge/evaluation.html):
onboard RGB, depth, proprioception, and permitted task definitions. Ground-truth
segmentation, object states/poses, full-scene point clouds and robot global pose
are not evaluation policy inputs. The trusted native evaluator can access state
for scoring; that state must not enter the policy/proposal observation channel.
Do not mutate scoring code, task goals, resets, physics or scene state to obtain
success. Scripted teleportation and symbolic state setters are not action primitives
for this comparison. Any privileged training-data collection is tracked separately.

## Environment

Runtime dependencies, datasets, checkpoints and experiment artifacts live outside
this repository, currently under `/cpfs02/user/yanming/behavior_runtime`.
The root filesystem is full, so large caches also use that runtime.
Native decrypted assets need temporary storage that supports unlinking an open
file without leaving a directory entry. The local CPFS mount left `.nfs` files
and caused the upstream shutdown cleanup to fail after a completed smoke rollout.
An actual open/unlink/rmdir probe passed on `/dev/shm`; native launches can use
`--temporary-directory` to select an existing private directory there. This path
is frozen in the launch specification and paired comparison identity. Keep the
same setting across baselines and evolved programs. The default remains
`runtime/tmp` for filesystems without this cleanup problem.
The launcher explicitly sets `OPENPI_DATA_HOME=runtime/cache/openpi`: upstream
OpenPI does not derive this location from `XDG_CACHE_HOME`. Its public PaliGemma
tokenizer is pinned in the seed backend and checked before loading weights.
The cached file at `big_vision/paligemma_tokenizer.model` must match SHA256
`8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6`.

The current container also had empty NVIDIA Vulkan and EGL vendor manifests.
Runtime-local replacements under `runtime/config/nvidia_icd.json` and
`runtime/config/10_nvidia.json`, pointing at the installed NVIDIA libraries,
allowed headless `vulkaninfo --summary` to enumerate the RTX PRO 5000 successfully.
`native.py` uses these optional manifests together and binds their hashes to each
launch specification. A separate Isaac Sim check created an empty USD stage and
completed three headless application updates (`runtime/isaac-headless-preflight.json`).
Neither check establishes a working BEHAVIOR episode or task performance.

The local simulator installation needed two dependency repairs after the upstream
setup command: exposing Isaac Sim's bundled Warp 1.8.2 through a runtime-local
Python path file, and pinning PyArrow 21.0.0 for the installed NumPy 1.26.0.
OmniGibson then imported successfully. The environment freeze and repair evidence
are bound by `runtime/simulator-environment-receipt.json`. Package metadata
conflicts remain recorded in `runtime/logs/simulator-pip-check-001.log`; the
environment has not passed a clean dependency check. A subsequent native scene
smoke completed after the additional Warp and temporary-storage fixes below.
The first native scene smoke exposed an additional requirement: bundled Warp
1.8.2 cannot compile the pinned evaluator's `mesh_query_ray_anyhit` call. This
primitive was added in [Warp 1.11.0](https://raw.githubusercontent.com/NVIDIA/warp/v1.11.0/CHANGELOG.md).
Installing `warp-lang==1.11.0` into the simulator environment resolved the missing
primitive in an actual CUDA hit/miss probe. The updated environment inventory is
`runtime/environment-warp111.freeze.txt`, bound by
`runtime/simulator-environment-receipt-002.json`. Native integration with the
updated dependency passed a zero-action radio scene smoke with verified JSON and
video (`runtime/native-environment-preflight-001.json`). Subsequent full-budget
radio trials are recorded below; broader task coverage remains unverified. The first smoke's traceback and
unresolved receipt are preserved; no score was produced and no failed setup run
is treated as a policy result.

For a fresh installation with the same pinned upstream release, apply the two
observed dependency fixes after the upstream packages are installed and before
starting any simulator process:

```bash
runtime_root=/cpfs02/user/yanming/behavior_runtime
"$runtime_root/tools/bin/uv" pip install --python "$runtime_root/env/bin/python" \
  --no-deps 'pyarrow==21.0.0' 'warp-lang==1.11.0'
"$runtime_root/env/bin/python" -c \
  'import pyarrow, warp; assert pyarrow.__version__ == "21.0.0"; assert warp.__version__ == "1.11.0"; assert hasattr(warp, "mesh_query_ray_anyhit")'
```

The installed Warp package supplies the required module directly; a new install
does not need the earlier bundled-Warp `.pth` workaround. Capture a new environment
inventory and native smoke receipt for each repaired installation. Do not overwrite
existing environment receipts or change dependencies while an experiment is active.
The import check above verifies versions and API availability; the native smoke
is still required to establish scene loading, rendering, stepping and shutdown.

The installation documentation names `v3.9.3`, but that exact tag was absent from
the remote on inspection. We pin the available `v3.9.3-post2` release at
`a8247a8cc1633fe1ca0cc66aa07243d46c64f155`, which reports OmniGibson 3.9.3.
Upstream requests Python 3.11, PyTorch 2.7.0/CUDA 12.8 and Isaac Sim 5.1.0.
The isolated environment is created with uv; the installation delegates to the
unmodified upstream setup script. `setup_env.sh` uses the upstream automated
license acceptance flags and downloads the simulation assets outside Git.

```bash
bash PhysicalRSI_baselines/behavior_baseline/setup_env.sh /cpfs02/user/yanming/behavior_runtime
python -m PhysicalRSI_baselines.behavior_baseline.doctor \
  /cpfs02/user/yanming/behavior_runtime \
  --output /cpfs02/user/yanming/behavior_runtime/preflight.json
TMPDIR=/cpfs02/user/yanming/behavior_runtime/tmp \
  python -m pytest -q -p no:cacheprovider tests/test_behavior_protocol.py
```

The setup command requires the pinned upstream checkout and a Python 3.11 venv
at `runtime/env`. The doctor reports dependencies without launching Isaac Sim;
even a successful preflight does not establish working physics or task success.
After installation, run the upstream native evaluator with `--policy local` for
a short smoke episode, then real baseline policies with normal episode limits.

```bash
mkdir -p -m 700 /dev/shm/physicalrsi-behavior-yanming
python -m PhysicalRSI_baselines.behavior_baseline.native \
  /cpfs02/user/yanming/behavior_runtime \
  /cpfs02/user/yanming/behavior_runtime/runs/native-smoke-example-001 \
  --task turning_on_radio --mode train --instance-index 0 \
  --policy local --max-steps 5 --seconds 900 \
  --temporary-directory /dev/shm/physicalrsi-behavior-yanming
```

This command is an environment check only. A policy evaluation uses
`--policy websocket --policy-revision <pinned-code-and-checkpoint-identity>` with
an already running `serve.py` server. The launcher checks its handshake's
candidate identity and task before starting the evaluator, and preserves that
metadata with the evidence. This is a cooperative service declaration, not proof
of isolation from a malicious server. Competitive results also require the
frozen comparison protocol and verified source/checkpoint manifests. The default native episode
limit must be retained for official comparisons.

For local orchestration, `service.policy_service(...)` is a context manager that
starts the frozen OpenPI candidate in its own process group, waits for a matching
candidate/task/PID/launch-nonce handshake, and yields its endpoint. It records
startup, handshake, logs and termination in a fresh evidence folder and stops
the process group on exit, including exception paths. Existing launch folders
must be reconciled rather than reused. Its lifecycle tests use a real child
process with a fixture handshake; they do not establish model inference.

`round_cases.RoundCases` allocates a fixed schedule from the admitted official
case pool. Declare the pool revision, maximum rounds, and per-task development
and validation counts before the first round. Every round retains every task;
contiguous allocations use the actual reset IDs and never reuse validation
instances across rounds. Insufficient cases fail before allocation. Native case
identities include the task name, and the public test set remains report-only.
Register each returned `CaseProtocol.reset_identities()` with core research
lineage before execution. Allocation itself launches no trials and makes no
admission or performance claim.

`native.specification(...)` constructs a launch specification before execution;
`native.run(..., expected_specification=...)` rejects changed conditions before
contacting the policy or simulator. `trials.NativeTrials` freezes an entire list
of attributed native trials and reserves it atomically through core `TrialQuota`.
It preserves the reservation on interruption, rejects duplicate semantic cases
even when endpoints differ, and checks every existing receipt for unresolved
execution before starting another trial. The orchestrator still owns complete
candidate/cohort coverage, admission and policy-service lifetimes; this transport
does not provide those guarantees by itself.

`batch.execute` connects a reserved `NativeTrials` plan to supervised local
OpenPI policy services. It verifies the complete candidate registry and task
bindings before starting services, reconciles every existing trial, and shares
one service across each candidate/task/endpoint group. Completed trials are
verified and reused; unresolved trials stop the batch. Service cleanup occurs
on episode errors, and interrupted service folders require explicit reconciliation.
The batch does not create admission evidence or select a survivor. Software tests
cover completion reuse, unresolved-plan rejection before startup, and cleanup on
trial failure. One real full-budget radio development batch subsequently completed
with verified native feedback, followed by the completed paired RSI round below.

`feedback.export_development` checks the entire preregistered incumbent
development cohort against native receipts, then exports only final Q-score,
outcome, canonical case identity and onboard RGB video into `development-feedback`.
The pinned upstream evaluator builds that video from head and wrist cameras.
Raw metric dictionaries and evaluator logs are excluded from the proposal input
evidence list; validation and report-only episodes are rejected. Returned feedback
uses the core `split="evolve"` contract. This projection does not sandbox a
proposal process that has unrestricted filesystem access.

`evaluation.NativeEvaluator` implements `admit`, `development`, `validation`,
`evaluate`, validation/evaluation recovery, and research split registration for
the existing core loop. Supply a canonical `CaseProtocol` from `RoundCases`, an
external `TrialQuota`, the fixed native runtime and policy Python, a System 2
revision, and an explicit admission producer. That producer exposes `identity()`
and `check(candidate, directory)` and writes its reviewed evidence there, returning
`accepted` and relative evidence hashes (plus a reason on rejection). It must not
launch simulator effects. The adapter verifies this evidence before execution;
it does not invent an admission based on a candidate hash.

The evaluator reserves the complete paired pool before its first service starts,
then reuses verified results for subsequent core evaluation calls. Set the core
loop's protocol to `{"identity": evaluator.revision, "evaluation_kind":
"policy_evaluation"}` and keep its profile and scope consistent with the cases.
The proposer can delegate development to `evaluator.development(parent, output)`.
Software fixtures exercise the actual core iteration through immutable lineage
inheritance, recovery without repeated effects, atomic quota rejection, and
tampered admission evidence. The current OpenPI backend is task-bound: a larger
task scope is rejected before trial reservation, never reduced automatically.
This adapter does not establish full-task checkpoint coverage or native RSI gains.

`admission.ReviewedAdmission` consumes explicit code-review records named by the
candidate revision. It requires review decisions for onboard inputs, native
controls, absence of privileged-state access, absence of evaluation feedback
during the episode, and absence of runtime code rewriting. The reviewer and
reason must be recorded for the exact candidate and control hashes. This is a
trusted source-review attestation, not an automated proof of policy compliance.
The gate retains the reference foundation, verifies the System 1 engine, official
tracked source revision, tokenizer and every declared checkpoint file, and copies
the supplied preflight evidence into the admission record. Rejected reviews remain
rejected; changed prerequisite or checkpoint bytes fail the check. It performs no
simulator effects. Environment/preflight records retain their original scope, and
each new control revision requires its own review before paired native trials.

`recovery_control.py` is the first experimental control proposal derived from
the completed radio development video. It detects sustained low joint and wrist
image change after prior arm motion, then briefly explores around the primitive's
target for the previously active arm. Offsets are bounded by 0.03 radians, each
attempt lasts at most 96 steps, and there are at most three attempts per episode.
Base motion or a larger new arm command aborts recovery. Base, trunk and gripper
commands remain the primitive output. Synthetic controller tests cover bounds,
onboard-image gating, batched independence, resets and abort behavior.
The immutable `pi05-radio-recovery-001` candidate completed a core Self-Harness v2
engineering comparison against the reference on radio validation instances 200
and 201, with full native budgets and scoped research memory. Both candidates
scored Q=0 on both episodes. The core selector retained `pi05-radio-seed-003` for
insufficient improvement; the state remains the original bootstrap revision,
with the rejection evidence saved in the completed round. All four native
receipts and their preregistered specifications were verified. No native gain or
full-task qualification has been established.

`diagnostics.DevelopmentTrace` is an optional recorder in `ProgramPolicy` for the
next development run. It stores only the 25 projected proprioception values, primitive actions and
issued actions, with candidate/task/case attribution and a content digest. It
rejects validation and test splits, existing output files and malformed commands.
Software tests establish filtering, input preservation, record integrity,
candidate/task attribution and action equivalence with recording disabled.
The engine source closure now includes the recorder. `policy_service` accepts an
explicit `development_binding` checked against the frozen case protocol and passes
it to the serving process. The handshake binds the output path and case identity;
the native launcher rejects recording unless its System 2 development attribution
and instance match. After native completion, the trace's public-only field schema,
attribution and step count are checked before a stable copy enters the native
receipt's hashed evidence. Development feedback exports that verified copy.
A fresh native development episode remains outstanding; old candidates require
their archived source snapshot, and new runs require newly bound candidates. A trace
does not certify native episode completion or a task score. The current recovery
candidate's two validation episodes both completed with Q=0; without native action
telemetry, those results do not establish whether its recovery detector activated.

`revision.materialize` saves a reviewed control proposal as a new immutable
candidate. It binds the hypothesis to the parent's exported development feedback,
checks the feedback files, and copies the fixed foundation, dependencies and other
declared components unchanged. Only `control.py` changes. The helper checks syntax
and required entry points without executing proposal code, rejects unchanged
programs and existing destinations, and leaves the parent intact. The caller
still owns code review, feedback admission and independent evaluation; creating a
revision is not evidence that it is valid or better. Native BEHAVIOR feedback has
now rejected one candidate; no improved candidate has been promoted.
The materializer requires the proposal environment, recorded costs, expected
effect, and required evaluation checks. It returns the core Self-Harness
candidate provenance fields directly, including the parent revision and bound
feedback evidence. Optional `used_lessons` lists research-memory identifiers;
the core research round validates them against its admitted lesson context.
A software integration check passes the materialized candidate through the
actual core iteration to its admission port, where the fixture rejects it.
That test establishes interface compatibility, not native admission or selection.

The pinned official demonstration inventory contains 100 tasks, 20,000 episodes,
and 210,916,774 frames. Its full published payload is 3,255,272,082,693 bytes;
RGB videos, action/state tables and metadata account for 1,076,826,593,518 bytes.
The installed LeRobot reader enumerates every video feature in metadata, so
omitting depth downloads alone is insufficient. A separate runtime metadata view
removes only the three depth feature declarations and their unused statistics,
preserving the original metadata, all episode/task rows, frame counts and RGB
payload paths. The actual reader's path and timestamp queries were checked to
request only the three RGB streams (`runtime/rgb-metadata-reader-probe-001.json`).

A subsequent offline CPU check decoded the first, middle and last frames of
official training episode 0 through that view and applied the pinned OpenPI
prompt, repacking, state and delta-action transforms. All three samples produced
three uint8 RGB images, a finite 23-dimensional state and a finite 32-by-23 action
sequence (`runtime/rgb-sample-reader-probe-001.json`). The four original payload
files total 706,329,027 bytes and have recorded hashes. This verifies a real
training-input path; full data download, full-cohort normalization, optimizer
execution and full-task model coverage remain unverified. Do not reuse the
radio checkpoint's normalization statistics for a newly trained full-task model.

The official `pi05_base/params` checkpoint has since been downloaded separately:
20 generation-pinned public objects, 12,441,721,931 bytes, with publisher CRC32C
verification and local SHA256 records. The official CPU weight loader restored
51 arrays containing 3,353,433,872 parameters, verified their shapes and dtypes
against `pi05_b1k`, and checked all values are finite
(`runtime/pi05-base-restore-probe-001.json`). This is a base checkpoint, not a
trained full-task BEHAVIOR policy. A CPU shape trace gives about 50 GiB for its
full training state including Adam and EMA; activation and runtime costs remain
unmeasured at runtime. The batch64 GPU compiler probe completed: its buffer
estimate is 81.35 GiB, exceeding the device's 71.69 GiB physical capacity before
runtime overhead. This exact single-device compiled step cannot fit. The batch32
compiler probe completed with only about 1.47 GiB remaining before runtime
overhead; it does not establish equivalent global-batch64 gradient accumulation.

A user-authorized agent DSW now provides two RTX PRO 5000 72GB GPUs. The pinned
official global-batch64 step compiled with FSDP across both devices, estimating
46,572,890,256 buffer bytes per device. Actual base-weight Adam/EMA initialization
succeeded, but the first optimizer probe failed when the BFC allocator requested
an 18.32 GiB temporary buffer. A `platform` allocator diagnostic allocated the
buffers but failed during execution with a CUDA illegal-address error. Small-array
arithmetic, all-reduce and all-gather checks passed on both GPUs with both
allocators. A preallocated BFC diagnostic remains in progress. Compiler feasibility
is not runtime success. These probes use synthetic data and never publish a
trained checkpoint.

The full100 training-table download is separately pinned to the official dataset
revision: 955 files, 74,558,696,633 bytes, two workers, publisher SHA256 checks and
a 300 GiB shared-storage free-space floor. RGB video download, full-cohort
normalization and real demonstration training remain outstanding. Full training
must also budget Arrow caches and checkpoint retention beyond the published
payload size.

`training_tables.py` joins training rows across file boundaries and constructs
episode-clipped action horizons without decoding images. It checks global row
order, episode lengths, task identities, finite values and complete-cohort
consumption. Global normalization batches may span episodes and preserve the
official sequential batch/drop-tail behavior. Six software tests cover boundaries,
input preservation and malformed cohorts. A real 1,956-frame episode traversed
this implementation; seven sampled frames, including terminal padding boundaries,
matched the official RGB reader's state/action transforms exactly. This does not
establish full-cohort statistics. Runtime normalization preparation requires all
955 publisher-verified training tables before statistics can be published.

The pinned normalization CLI accepts `--config-name` and `--max-frames`; it does
not accept the dataset override flags shown in the pinned baseline documentation.
The runtime `scripts/full100_config_001.py` builder instead creates an explicit
OpenPI configuration for the full RGB dataset, a separate full-task normalization
asset ID and the verified local base checkpoint. CPU configuration construction
passed (`runtime/full100-config-probe-001.json`). This is configuration evidence
only: the builder does not download missing data or launch normalization/training.
Both future workloads must consume the same configuration. Batch size, step
count, workers and checkpoint retention in the probe are preview values, not a
launched or frozen training experiment.

Official references: [installation](https://behavior.stanford.edu/getting_started/installation.html),
[baselines](https://behavior.stanford.edu/challenge/baselines.html),
[upstream source](https://github.com/StanfordVL/BEHAVIOR-1K).
