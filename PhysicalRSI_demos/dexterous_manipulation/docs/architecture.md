# Architecture and evidence boundaries

## Relationship to existing workflows

| Entry | Purpose | Implementation |
| --- | --- | --- |
| Existing `/demo dexjoco`, `/layouts`, `/collect`, `/train`, `/cycle` | Demonstration playback and the configured dataset/policy-training workbench | `PhysicalRSI_demos/` and the main CLI |
| `physicalrsi-dexterous-loop` | Direct GPT-as-Policy execution with Core memory/program reflection, paired trials and lineage | `PhysicalRSI_demos/dexterous_manipulation/`, installed with the repository's `dexterous` extra |

The research code exposes its own console command and the
`PhysicalRSI_demos.dexterous_manipulation` Python subpackage. Existing `/demo`
routes keep their current behavior. It imports the repository Core directly.
Each research campaign requires its own new
workspace. Use the separate environment in the README so its optional simulator
dependencies do not alter an existing demo environment.

Backend-specific module names are retained to preserve correspondence with
the executed source snapshot. `physicalrsi-dexterous-loop` currently runs the
simulation backend; its broader name does not imply a completed hardware loop.

## Components

| Component | Responsibility |
| --- | --- |
| GPT-as-Policy `CodexPolicy` | Persistent app-server thread, image tools, shell, episode notes and action-tool calls |
| `dexjoco_sim.py` | Native task reset/step, bounded target interface, measured observations and continuous video |
| `dexjoco_robot_feedback.py` | Robot-only forward kinematics from measured joints and fixed geometry |
| `NativeVerifier` | Recorded native `info.succeed` outcome and image/trace evidence |
| `dexjoco_campaign.py` | Astra reflection tools, candidate creation, source admission and one Self-Harness round |
| `dexjoco_loop.py` | Preregistered case schedule, bounded multi-round campaign and actual survivor-load audit |
| `working_programs.py` | Bounded copies of episode scratch Python and public command outcomes |
| `task_diagnostics.py` | Verified, post-episode native replay for System 2 failure analysis |
| `research_feedback.py` | Completed rejected-candidate history and pinned Core ResearchMemory retrieval |
| PhysicalRSI Core | Experiments, independent evidence, Self-Harness, paired comparison, selection and lineage |

## Preserved behavior

The upstream policy is patched to accept a runtime profile. Its persistent
Codex app-server transport, image handling, native shell tools and working
workspace remain in use. Same-thread network continuation is allowed for the
simulation profile under the reviewed upstream recovery contract. An uncertain
action or interrupted experiment is not silently converted to a failure or
replayed as if nothing happened.

The simulation actor sees three RGB views, 46 robot-state values, camera
calibration, robot fingertip/palm geometry and fixed task rules. It receives no
live scene-object pose arrays, contact oracle or privileged planner. Generated
helpers can analyze supplied observations through the shell. Motion still goes
through the bounded native action interface.

System 2 receives additional simulator-assisted diagnostics only after a
completed development/validation episode. The recorded controls are replayed;
all public robot-state checkpoints must agree within 1e-7 and native success /
termination flags must match. These diagnostics do not alter scores and should
be described as additional simulation feedback, not a deployable robot sensor.

## Candidate lifecycle

1. Run development episodes under the frozen parent.
2. Collect outcomes, bounded working code, public tool outputs and diagnostics.
3. Retrieve applicable prior failures from an immutable ResearchMemory revision.
4. Let Astra submit a complete memory and optional helper inventory.
5. Validate helper restrictions, bind source/content hashes and freeze the candidate.
6. Run parent and candidate on fresh, matching initial conditions.
7. Select using native task success and the frozen comparison protocol.
8. Commit the selected revision and verify its actual memory/program bytes in
   the next round's policy workspace.

An unsuccessful tie retains the parent. Rejected proposals remain available as
historical development evidence. ResearchMemory retrieval is active;
ResearchMemoryGate and ShadowGate are not enabled. There is no demonstrated
promotion in the real runs documented here. Retained-parent inheritance was
checked in a completed campaign; promotion paths have software-fixture coverage.

## Packaging changes from the running snapshot

The original working directory contained physical copies of upstream packages.
This export resolves the executing Core and GPT-as-Policy packages at import
time and hashes their source under explicit logical roots. Candidate admission
checks the complete current manifest, including additions and removals. No Core
implementation is modified by this contribution.

Native source paths are explicit CLI arguments, simulator tests use
`DEXJOCO_SOURCE`, and dependency preparation verifies the pinned patched runtime.
The former standalone `tianji_agent` package is integrated as
`PhysicalRSI_demos.dexterous_manipulation`; tests and analysis scripts use that
namespace. The root project supplies optional dependencies, its console entry
point and packaged skills, provenance and runtime patch.

The export retains the simulation execution path and extracts shared numeric
validation and media verification from the original hardware adapter. The
base policy/profile now require concrete environment hooks instead of supplying
Tianji defaults; the API provider label is `physicalrsi_astra`. Only Assembly
and Photograph are exposed. Hardware drivers, gateways, configuration drafts
and their tests are excluded. The app-server/Core integration tests use a
software environment with the simulation observation/action shape. Historical
results refer to the original snapshot, not a new rollout of this export.

The recently prepared acquisition-feedback changes live in a separate local
working version and are not represented here as code used by the active
campaign. The executed standalone grasp review and its findings are included.
