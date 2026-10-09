# Dexterous manipulation research record — 2026-10-09

This is a progress record of real online-agent experiments, including failures.
It is not a task-success announcement. Raw workspaces and videos are retained
outside this source contribution. Artifact identifiers and hashes below allow
comparison with the retained evidence; the raw evidence is not bundled here.

Research scope: bimanual acquisition, object retention, contact manipulation
and eventual transfer to Tianji arms and BrainCo hands. The results below were
obtained with the native DexJoCo simulation backend. They are not hardware trials.

## Completed comparisons

| Campaign | Scope | Completed policy episodes | Native successes | Selection |
| --- | --- | ---: | ---: | --- |
| `dexjoco-rsi-upstream-round-02` | One Assembly/Photograph round | 6 | 0 | Parent retained |
| `assembly-reviewed-loop-01` | Two Assembly rounds | 6 | 0 | Parent retained in both rounds |
| `learning-feedback-loop-01`, round 1 | Development plus paired Assembly/Photograph validation | 6 | 0 | Parent retained |

The learning campaign's second round was still running when this source export
was prepared. Its incomplete outcomes are not a success-rate estimate. The
completed two-round Assembly campaign verified that the selected parent's
actual artifact bytes were loaded in the next round.

## What the new feedback path actually did

The formal first-round reflection received two fresh development episodes,
one captured scratch Python file, 15 relevant public command records, native
post-episode diagnostics and six lessons from completed historical validation.
Its pinned ResearchMemory revision was:

`2ae624176011a5be51bac19e892c9be70f5e73f5b35d3ae6fa512dd9096d95c7`

Astra submitted a 11,579-character memory and the 7,908-character
`visual_geometry.py`. Its 12 constructed numerical/CLI tests passed in the
public reflection tool log. The frozen helper matched the tested source.
The candidate Assembly actor subsequently imported and invoked it on current
RGB pixel selections. This establishes real feedback, proposal, loading and
tool use; it does not establish improved task success.

| Artifact | SHA-256 |
| --- | --- |
| Frozen candidate | `538bb8bb24a9c48e9b0d25e0bbe330f517c0c20472ba4cf0dfdc91200ef8b2bb` |
| Helper source | `c17975d373b18eaec78123baa9fa3fbc6187fef19eba77dc95ca302ced6d5778` |
| Paired Assembly initial condition | `13825f0946dd15549e938cb553af1f34eaf3430ab5f733ccecaa5c632a2ba2cb` |
| Candidate Assembly continuous video | `2bb2493bb48b6969b0291ea06d2a4acf75c142fb7e7a7b50cbebd0280c9d7825` |

A separate archived-reflection integration check produced a different helper
with 14 constructed tests. It did not execute candidate policy episodes or
promote a candidate. It must not be confused with the formal 12-test candidate.

## Earliest failure: left-tray acquisition

The user identified an important omission in the initial interpretation:
**the left hand had not picked up the tray**. Reporting only smaller peg-to-hole
errors obscured this earlier task-stage failure.

On the first-round paired Assembly case (seed 7203), the candidate's closest-tip
lateral error was approximately 33 mm versus the parent's 74 mm, and directed
axis error was approximately 11 degrees versus 25 degrees. Both failed and
neither contacted the socket bottom. This single comparison is not evidence of
stable capability improvement or a causal attribution to the helper alone.

The independent grasp review verified all 56 candidate public checkpoints with
zero replay error and found:

- First left-hand contact at control 292; 943 controls with active left contact.
- After first contact, table support was present for 1197 of 1209 controls.
- Maximum clearance of the tray's lowest collision surface was **0.40 mm**.
- There were **zero controls** with left-hand contact, no table support and
  at least 1 mm of tray clearance; the 5/10/20 mm checks were also zero.
- During one attempt the left wrist rose approximately 40 mm while the tray
  remained supported by the table. Operational notes overstated the hold.

This is a left-pickup failure. Contact, wrist movement, closed fingers or a
smaller insertion error cannot stand in for verified acquisition and retention.
The original native Assembly success predicate checks peg-bottom contact for
30 controls; it does **not** independently require left-tray pickup. A native
score therefore cannot alone certify the requested bimanual workflow.

The next hypothesis is to make acquisition/retention evidence precede alignment
in reflection and acting memory, using fresh RGB to reject false pickup claims
and trigger bounded regrasp. The independent review is implemented and was run;
its integration into future feedback is not an achieved manipulation result.

Photograph also remained unsuccessful: parent/candidate validation had zero
viewing-angle pass controls. The candidate reached the capture region for 59
controls and triggered the shutter for 54, but never satisfied all conditions.

## Validation and limitations

The source snapshot before export passed 54 adapter checks. Six imported native
validation episodes were replayed with 327 public checkpoints, all with zero
error. Source-export checks are recorded separately in the local PR review.

The campaign uses one validation case per task per round. It cannot establish
a reliable success rate, generalization or a hardware-ready control policy.
Tianji model, SDK/ROS actions, joint mapping, hand calibration, robot geometry
and synchronized sensing still require device-side integration. No physical
robot was moved by these experiments.
