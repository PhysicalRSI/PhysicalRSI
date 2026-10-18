# RoboDyna dynamic interception research protocol

The research question is whether one composable interception capability can
improve on `catch_ramp_ball` and transfer to `catch_valley_ball` without tuning
on Valley outcomes. Native outcomes come from the upstream task implementation.
CPU orchestration tests establish persistence contracts only.

## Existing implementation reused

| Responsibility | Actual implementation | Use in this study |
|---|---|---|
| Primitive operations and contracts | `PhysicalRSI_core/contracts.py` | Shared skill input/output checks |
| Registered composition | `src/physicalrsi/skills/` | Versioned executable shared skill |
| Evolution loop | `PhysicalRSI_core/self_harness/__init__.py` | Development, candidate freezing, paired evaluation and selection |
| Trial evidence | `PhysicalRSI_core/experiments.py` | Reset/execute/verify receipts and trajectories |
| Evaluation | `PhysicalRSI_core/self_harness/evaluation.py` | Retained development, paired trials and report-only heldout |
| Research question and dispatch record | `PhysicalRSI_Autoresearch/records.py`, `src/physicalrsi/research/workflow.py` | Composition-bound question, frozen experiment, dispatch, immutable result and decision |
| Research persistence | `PhysicalRSI_core/self_harness/research_campaign.py` | Durable Ramp-only or joint evolution round |
| Selection lineage | `PhysicalRSI_core/lineage.py` | Parent/candidate/selected evidence identities |
| Seed separation | `PhysicalRSI_core/self_harness/protocol.py` | Frozen development/validation/test identities |
| Capacity and trial bounds | `PhysicalRSI_core/infra/scheduler.py`, `trial_quota.py` | One simulator allocation, durable paired candidate jobs, bounded trials |

`SkillLibrary` and `ExperienceCompiler` implement executable bundles compiled
from owned development experience and bound to research memory. This first
experiment changes a declared parameter artifact with a trusted execution
backend, so it does not claim compilation through those facilities.
`CandidateArchive` and `HarnessTree` support larger search histories; one
bounded candidate uses the existing campaign and lineage directly.

## Comparison order

A0 is a simple fixed-parameter control. A1 is the shared composition before
evolution. Development and fresh paired validation run on Ramp only. The
proposer records the retained failure evidence and bounded causal hypothesis;
the candidate and all implementation dependencies freeze before comparison.
Only the shared prediction parameter artifact is mutable.

The automatic proposer estimates the rolling-energy coefficient from retained
Ramp observations. Three-point finite differences estimate acceleration along
the nearest static guide segment; division by the local downhill gravity
component gives a coefficient. The median of at least three unique valid
triples sets the candidate coefficient. Overlapping windows are deduplicated;
uphill/flat segments, corners, non-rolling observations and estimates outside
[0, 1] are excluded. No simulator oracle speed or task-specific coefficient
enters the estimate. Insufficient evidence retains the parent. An explicit
candidate parameter dictionary can instead preregister a bounded hypothesis.
The proposal revalidates retained development receipts before using observations.

A2 is the selected survivor. A rejected candidate leaves A2 identical to A1.
The default pilot heldout seeds are 10000, 10001 and 10002, present in both tasks' upstream `default` scenario seed lists. Development and selection use distinct seeds.
All three report groups are committed before heldout access. Each is evaluated
on the same heldout seeds on both tasks. Identical revisions share the same
report and do not consume duplicate simulator trials. Valley outcomes never
enter this campaign's proposal or selection. Joint evolution uses `--joint-after` to start a new protocol in a separate
workspace after this comparison. The predecessor must have completed campaign
and research records; its evidence hashes and selected harness are verified.
The shared parameter artifact is inherited, while the new executable source
closure is pinned independently.

Joint development uses seeds 301–302, paired validation 401–403 and final
reporting 10003–10005 on both tasks. The implementation rejects overlap with
any predecessor seed split. Its preregistered prediction-calibration trigger
allows a bounded revision from admissible observed acceleration even if all
development episodes succeed. Selection uses both tasks and requires zero
success regression on either task. Joint results remain separate from the
preceding Valley zero-tune comparison.

Success rate uses every planned episode in its denominator, retaining failed,
uncertain and invalid outcomes. Reports include counts, a 95% Wilson interval,
native progress and episode latency when the adapter records them. Intervals
are descriptive and do not establish independence of simulator scenarios.

## Resume and frozen identities

`research/composition.json` binds the actual executable composition. The
Autoresearch record binds the parent, evaluator and Ramp protocol before a
durable local dispatch claim. A claim does not prove a running worker. The
result record binds the completed campaign, report-group identities, heldout
protocol and individual reports; the closing decision does not modify skills.

`config.json` fixes suite implementation, skill parameters, seeds, mode and
budgets. Candidate manifests hash the complete declared component closure.
`campaign/current.json` identifies the current immutable campaign revision;
`lineage/` binds selected revisions to comparison evidence. Trial receipts and
scheduler records are reconciled by existing Core code. Completed work is read
and verified on restart without rerunning simulator episodes. Resume checks
executable source identity as well as parameters and seeds; changed source is
rejected. Historical campaigns should be resumed from their archived source
with matching native runtime configuration, not silently migrated to new code.
The first completed native pilot's resume preserved 14 workers and 15 receipts.
Its two successful Ramp development episodes triggered no candidate, leaving
A2 identical to A1. An interrupted
external effect requires explicit reconciliation; a missing receipt is never
permission to reset or repeat a trial automatically.

Native evidence stays in the chosen external experiment workspace, including
adapter logs and any exported recordings. `--worker-output` selects a separate
staging directory; exported trial evidence remains under Core's durable
experiment records. Shutdown finalizes native recording and closes the scene;
the parent waits for exit and terminates the process group on failed graceful
shutdown. Resource monitoring, pipes and log handles are closed. Staging files
are preserved rather than automatically removed. Generated workspaces do not belong
in the repository. Frozen and async evaluation are separate protocols and
must not be pooled as one experiment. Actual native performance is reported
only after the environment has completed its reset/step/score lifecycle.


## Native scoring cutoff

The worker caches native score details and failure reason at 15,000 simulation
steps and accepts native success only up to that boundary. A low-level native
action may finish beyond it; subsequent physics cannot improve the reported
outcome. This is a scoring cutoff, not a claim that physics stops at exactly
15,000 steps. The experiment also bounds action count and wall time. Reports
retain the observed step count, configured budget and termination reason.
