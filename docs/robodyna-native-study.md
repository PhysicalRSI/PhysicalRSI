# Native RoboDyna skill study

PhysicalRSI runs `catch_ramp_ball` and `catch_valley_ball` through the official
RoboDyna environments at commit `6422e622d8025c3eaa4559bb4250c968fe4b2873`.
The native study uses real simulator rollouts, robot execution and native scoring.
The separate `study robodyna --fixture` command remains a CPU contract example.
See the [runtime setup](../PhysicalRSI_baselines/robodyna/README.md) for pinned
dependencies and a native lifecycle check.

## Actual integration boundaries

| Responsibility | Reused or implemented module |
|---|---|
| Native process isolation and Core trial ports | `PhysicalRSI_baselines/robodyna/adapter.py` |
| Reset, observation, step, native verdict and recording | `PhysicalRSI_baselines/robodyna/worker.py` |
| Cup grasp/placement and box push execution | `PhysicalRSI_baselines/robodyna/execution.py` |
| Shared primitive and composite skills | `src/physicalrsi/skills/dynamic_interception.py` |
| Campaign and final transfer orchestration | `examples/robodyna/native_study.py` |
| Selection and immutable experiment evidence | Existing Core `SelfHarness`, `ExperimentEvaluator`, `CaseProtocol`, `ResearchCampaign` |
| Trial scheduling and resource budget | Existing `ScheduledEvaluator`, `JobScheduler`, `TrialQuota` |
| Research hypothesis, dispatch, result and review | Existing Autoresearch `ResearchRecord` through `ResearchWorkflow` |

The framework runs in Python 3.11 or newer; each native episode runs in a separate
Python 3.10 process with an isolated output directory. Upstream assets and task
source stay in an external pinned checkout. The official asset downloader checks
its archive checksum, and its path expansion script generates machine-local
robot configuration. Runtime task/configuration files and adapter source hashes
are bound to experiment identity.

## Mode, observations and actions

The recorded mode is `native-interactive-robot`. The worker sets upstream's
`_interactive_robot_mode` before reset, enabling the Valley box's contact-driven
robot execution. This distinction matters because a frozen scene-interactive box
has different execution semantics. These runs are not executions of the
published VLA policy evaluator.

Observation qualification is `simulator-state-and-static-geometry`. The shared
skill receives position history, simulation timestamps, motion phase, static
guide geometry, gravity, capture height and a target reachability envelope. These
are privileged state inputs, not camera-derived perception. The policy receives
neither task names and seeds nor native scores and success flags. Future landing
coordinates, launch velocity and baked future ball lanes are excluded.

All shared vectors are world xyz in metres; timestamps and durations are seconds.
Motion estimation uses a fixed three-sample regression. Interception prediction
transports observed speed along static geometry and predicts a descending
ballistic crossing. Planning checks the declared workspace and speed envelope.
History is caller-owned and resets on episode and phase changes. Insufficient
observations return an explicit hold proposal. No task ID selects tuned targets.
A regression residual measures fit error; it is not calibrated uncertainty.

The two tasks share the same composition and prediction parameters. Their
execution adapters interpret the capture target as cup placement or box pushing,
using native robot motion and contacts. Native `check_success`, `get_score` and
`get_score_detail` supply outcomes; the adapter does not replace their formulas.
Success is latched only when observed at or before the 15,000-step simulation
budget. At that boundary the worker caches native progress, score details and
failure reason. A native action can finish after the boundary; later success
cannot turn that episode into a success. The boundary freezes scoring rather
than interrupting physics mid-action. Native success, the native deadline and
the budget determine the recorded termination reason. This budget is part of the declared
adapter protocol. The experiment runner additionally bounds each episode to
800 adapter actions and 600 seconds by default.

In `frozen` mode no additional simulation time is charged for inference. In
`async` mode the upstream mechanism advances dynamics for measured shared-skill
inference time. The recorded timing scope is **shared-skill inference only**:
process overhead, recording and low-level planning wall time are not an emulated
VLA inference delay. Keep frozen and async results separate and use separate
workspaces for their protocols.

## Research and transfer protocol

The CLI preregisters Ramp development seeds 101–102, paired Ramp validation seeds
201–203 and final seeds 10000–10002 for both tasks under the default scenario.
The study checks split separation and freezes its configuration before execution.
This small final subset is a pilot, not the full official benchmark suite.

The comparison groups are:

- **A0:** the same composition with fixed velocity gain 0.5 and energy factor 0.
- **A1:** the parent composition with velocity gain 1 and energy factor 0.
- **A2:** the selected survivor, which may be identical to A1.

System 1 runs the composition. System 2 reads retained Ramp development failures
and may propose one bounded change to the shared prediction primitive. Its
estimator derives a rolling-energy coefficient from observed tangential
acceleration and static guide slope, using the median of admissible samples.
It does not read upstream's future velocity. Motion estimation and execution
remain fixed. If development provides no failure or insufficient admissible
evidence, it retains the parent rather than manufacturing an improvement.

A proposed candidate freezes before fresh paired validation. The existing
Self-Harness compares parent and candidate and records selection with its
inherited revision and evidence. All final groups freeze before held-out access.
The Ramp-selected survivor is then evaluated on Valley without Valley tuning.
No final observation feeds proposal generation. Identical A1/A2 revisions reuse
one final report instead of consuming duplicate episodes. Joint evolution is implemented as a separate experiment after this transfer
comparison, started with `--joint-after PREDECESSOR_WORKSPACE`. It verifies the
completed predecessor's records and evidence, inherits the selected shared
parameters, and requires a new workspace. Its default seeds are 301–302 for
development, 401–403 for validation and 10003–10005 for final reporting, on both
tasks. Overlap with any predecessor seed split is rejected.

The joint protocol preregisters an observed-motion calibration trigger, so a
candidate can be proposed even when development episodes succeed. It still
requires admissible acceleration evidence and a changed bounded parameter.
Paired selection checks both tasks and permits no success regression on either.
New source pins belong to this new experiment; they do not reinterpret the
historical zero-tune transfer result.

The predictor is a physical hypothesis. Upstream Ramp uses slowed,
friction-dependent guide acceleration; Valley uses a kinematic guide and an
independent release rule. A coefficient fitted from Ramp observations therefore
need not improve Valley. Successful reuse of the interface, within-task
improvement and empirical transfer are distinct claims.

## Evidence and restart

The campaign retains frozen candidate manifests, parameter diffs, quota usage,
selection lineage and Autoresearch records. Native workers retain action and
event logs, state trajectories, score details, timing and rendered video.
`summary.json` and `report.md` aggregate completed final reports, including
per-task counts, success rates, progress, latency and success-rate uncertainty.
Setup errors remain separate from unsuccessful policy episodes.

Rerun the identical command and workspace to resume through the existing
persisted campaign. Frozen configuration and source identities are checked. Changed executable
source is rejected; use the archived source and matching runtime to resume a
historical campaign. The completed first pilot's restart check preserved its
14 native worker directories and 15 trial receipts without launching new
rollouts. That pilot retained its parent because both Ramp development episodes
succeeded; this is not evidence of a successful evolved revision.

`--worker-output` can place worker staging outside the durable campaign directory.
Core exports worker artifacts into retained trial evidence. Close requests
finalize video and event streams and release the simulator; the parent waits
for worker exit, then uses process-group termination if graceful shutdown fails.
It also stops resource monitoring and closes process pipes and logs. Worker
staging is not automatically deleted. Preserve unexported evidence and reconcile
an interrupted worker before considering cleanup.

On incomplete runs,
interrupted external execution requires reconciliation before retry. A completed
result must be supported by its retained native episodes, not by contract tests
or the presence of a dispatch record. The automated suite checks the shared
contracts, orchestration and persistence; measured benchmark claims come from
completed native reports.
