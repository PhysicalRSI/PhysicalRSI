# RoboDyna shared interception pilot

This study executes RoboDyna's `catch_ramp_ball` and `catch_valley_ball` through
native robot actions and upstream success/progress checks. It uses privileged
simulator state and static geometry in `native-interactive-robot` mode, with
frozen inference timing. These are code-policy pilot results, not scores from
the published visual-policy protocol.

## Ramp development and zero-tune Valley evaluation

The shared composition is `motion_estimation → intercept_prediction →
intercept_planning`. A0 and A1 use the same composition and execution adapters.
A0 is a fixed-parameter control with velocity gain 0.5; A1 uses gain 1.0. Both
start with rolling-energy factor 0 and a 30-second prediction horizon. Thus
this comparison measures parameter choices, not the causal advantage of
composition over a monolithic policy.

Development used Ramp seeds 101 and 102. Both succeeded. The preregistered
failure-triggered proposer therefore generated no candidate, and no paired
validation was dispatched. A2 retains A1 exactly; it shares A1's final receipts
instead of repeating identical trials. Reserved validation seeds were
201–203. Final evaluation used the upstream `default` scenario seeds
10000–10002 independently on both tasks.

| Group | Task | Success | Rate | 95% Wilson interval | Mean native progress | Mean skill inference |
|---|---|---:|---:|---|---:|---:|
| A0 fixed control | Ramp | 3/3 | 100% | 43.9–100% | 1.000 | 0.311 ms |
| A0 fixed control | Valley | 0/3 | 0% | 0–56.1% | 0.000 | 0.357 ms |
| A1 shared parent | Ramp | 2/3 | 66.7% | 20.8–93.9% | 0.667 | 0.312 ms |
| A1 shared parent | Valley | 0/3 | 0% | 0–56.1% | 0.000 | 0.359 ms |
| A2 retained survivor | Ramp | 2/3 | 66.7% | 20.8–93.9% | 0.667 | 0.312 ms |
| A2 retained survivor | Valley | 0/3 | 0% | 0–56.1% | 0.000 | 0.359 ms |

All planned episodes have native outcomes; uncertain and invalid counts are
zero. Intervals describe the small pilot sample and do not establish scenario
independence. Inference measures the shared CPU skill only. It excludes native
motion planning, communication, rendering and evidence storage.

This run demonstrates executable reuse across the two adapters. It does not
demonstrate evolution improvement or successful transfer: Valley failed in both
groups, and the shared parent underperformed the control descriptively on Ramp.
Successful development cases did not predict perfect heldout performance.

## Evidence and execution observations

The completed workspace is `/cpfs/PhysicalRSI_runs/robodyna-shared-001`.
Its `report.md`, `summary.json`, `research/`, `campaign/`, `lineage/` and
`final/` retain outcomes, immutable research records and trial receipts.
`source-snapshot.tar.gz` and its hash manifest preserve the initial executable
source. A completed restart was tested before source changes: all 14 worker
directories, 15 receipts and the summary hash remained unchanged. See
`validation/resume.json`.

There are 14 actual rendered episode videos: two development and twelve
distinct heldout rollouts. A2 reuses A1 evidence. One GPU worker ran at a time
on an NVIDIA RTX PRO 5000 72GB Blackwell. Measured peak GPU memory was 3312 MiB;
mean native initialization was 15.6 seconds and maximum process RSS was
3.57 GiB. Total worker wall time was 2077 seconds. Shared-filesystem evidence
writes contribute to this runtime; it is not a policy-compute benchmark.

Read-only analysis of the A0 Valley trajectories found actual box movement of
7.8–11.4 cm, with a 31.7–35.0 cm gap to the last commanded target. Plans
completed, but neither red-line clearance nor catching received native progress
credit. Predictions became infeasible around 5.6–6.1 simulation seconds, after
which the policy continued observing. This makes contact execution and target
timing plausible bottlenecks; it does not isolate them from prediction error.
The numeric observations and evidence hashes are retained in
`validation/a0-diagnostics.json`. They do not supply tuned parameters to the
next experiment.

## Joint calibration and paired selection

The subsequent experiment inherits the selected shared parameters and uses
fresh development seeds 301–302, paired validation seeds 401–403 and heldout
seeds 10003–10005 on both tasks. Its proposal calibrated one shared rolling-energy
coefficient from 925 valid observed development triples: `0 →
0.003489185063311683`. Valid per-triple estimates ranged from 0.001465 to
0.041866. The candidate changed no other skill parameter or execution adapter.
Each task has its own no-success-regression constraint. This is joint
development, not zero-tune transfer.

| Paired validation | Ramp | Valley | Mean normalized native progress |
|---|---:|---:|---:|
| Parent | 3/3 | 0/3 | 0.500 |
| Frozen calibration candidate | 3/3 | 0/3 | 0.500 |

Self-Harness retained the parent with `no_sufficient_improvement`: normalized
gain was zero. The parameter diff, candidate, development analysis, twelve
paired trial receipts and selection lineage remain available for inspection.
This is a completed feedback/proposal/comparison/selection loop, with a negative
improvement result. It does not justify deploying the calibrated candidate.
An independent treatment audit verified all twelve validation receipts and
their distinct parent/candidate policy revisions, and replayed every recorded
first feasible command. For Valley's identical paired first-command inputs,
the candidate changed the target by 0.514–0.722 mm. The negative result therefore
does not arise from accidentally dispatching the same policy twice. A small
parameter effect and execution error remain competing explanations; this
comparison does not isolate either cause.

The independent test on seeds 10003–10005 completed as follows:

| Group | Task | Success | Rate | 95% Wilson interval | Mean native progress | Mean skill inference |
|---|---|---:|---:|---|---:|---:|
| A0 fixed control | Ramp | 2/3 | 66.7% | 20.8–93.9% | 0.667 | 0.324 ms |
| A0 fixed control | Valley | 0/3 | 0% | 0–56.1% | 0.000 | 0.372 ms |
| A1 shared parent | Ramp | 2/3 | 66.7% | 20.8–93.9% | 0.667 | 0.319 ms |
| A1 shared parent | Valley | 0/3 | 0% | 0–56.1% | 0.000 | 0.380 ms |
| A2 retained survivor | Ramp | 2/3 | 66.7% | 20.8–93.9% | 0.667 | 0.319 ms |
| A2 retained survivor | Valley | 0/3 | 0% | 0–56.1% | 0.000 | 0.380 ms |

Uncertain and invalid counts are zero. A2 reuses A1's receipts because the
candidate was rejected. Mean episode wall times were 65.1/179.9 seconds for
A0 Ramp/Valley and 67.9/181.6 seconds for A1 Ramp/Valley. This round used local
scratch for frequent evidence writes; its wall times should not be interpreted
as a policy speed improvement over the first round.

The joint campaign contains 28 native episodes and videos: four development,
twelve paired validation and twelve distinct independent-test episodes. Its
completed restart preserved all 28 worker directories, 29 receipts and the
summary hash. Together, the two campaigns contain 42 distinct native episodes.
The active joint workspace was `/tmp/physicalrsi-robodyna-joint-001`; its durable
archive and readable reports are under
`/cpfs/PhysicalRSI_runs/robodyna-joint-001`. Restore archived workspace directories
to their recorded absolute paths before replaying persisted records.

## Validation and next research questions

The implementation passed 1007 CPU tests, compilation, and the existing demo
and skill-memory entry checks. Tests ran on a byte-identical Python source
snapshot; they establish software contracts separately from native outcomes.
Both completed native campaigns passed actual restart checks without new
workers or modified receipts. The evidence includes native scoring, rendered
video, trajectories, timing, resource measurements, parameter diffs and lineage.
A separate native async lifecycle check injected 100 ms of inference latency:
the environment advanced 25 latency steps plus 25 requested steps, totaling 50.
This checks the timing mechanism; it is not an async performance evaluation.

This experiment provides an executable shared skill and a functioning
feedback-driven selection loop. It provides no evidence that the tested
calibration improves either task. A further study would need to separate
prediction timing from contact execution using new development evidence.
Visual perception, additional embodiments, more tasks and a comparison against
task-specific tuning remain separate experiments; none is inferred from this
state-observation pilot.

See the [research protocol](robodyna-research-protocol.md) and
[native setup](../PhysicalRSI_baselines/robodyna/README.md) for reproduction.
