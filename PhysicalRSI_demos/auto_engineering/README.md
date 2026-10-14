# Auto-engineering lab demo

See the [integration TODO](TODO.md) for the planned skill condensation, memory
retrieval and controlled transfer experiments through the physicalRSI pipeline.

A self-built Isaac Sim scene: a Franka arm transfers a blue sample tube from a
source stand into a green receiving rack without disturbing an amber beaker
or a red distractor. The bench, racks, room and sample vials are authored for this demo. Optional
LabUtopia meshes supply the glass beaker and background drying oven, while a
Poly Haven HDRI supplies studio illumination. No RoboDojo scene, task registry
or expert trajectory is used. The Franka robot is an external NVIDIA asset.

The workflow follows the engineering loop described in
[General Robotics' Auto-Engineering article](https://www.generalrobotics.company/post/introducing-auto-engineering-for-robotics):

1. **Robot ingestion:** record joint names, sensor contracts, simulator and assets.
2. **World experience:** build a lab and generate reproducible randomized layouts.
3. **Skill creation:** compose RGB-D detection with physical pick/lift/carry/release.
4. **Evaluation and repair:** diagnose a development failure, propose a higher
   carrying path, compare both versions on new paired layouts, and retain only
   an improvement through the existing `SelfHarness` and `HarnessState`.

## Presentation assets

Download assets outside the source tree using a separate Python environment
with `usd-core` installed:

```bash
python scripts/prepare_lab_assets.py --assets-dir /path/to/lab-assets
export PHYSICALRSI_LAB_ASSETS=/path/to/lab-assets
```

The script checks the LabUtopia source digest and records the derived mesh and
HDRI hashes. LabUtopia assets are **CC BY-NC 4.0**; Poly Haven lighting is **CC0**.
See the generated `lab-assets.json` for source links and modifications. Assets
are downloaded separately and are not redistributed in this repository. The
lab uses authored fallback glassware if the optional assets are absent.

## Run

Use Python 3.11 with Isaac Sim 5.1 and its extension cache installed, following
[NVIDIA's instructions](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html).
Install the simulator dependencies with
`/path/to/isaac-env/bin/python -m pip install -r PhysicalRSI_demos/auto_engineering/requirements-isaac.txt`.
Install the lightweight physicalRSI dependencies in the launcher environment.
Run from the repository root:

```bash
./physicalrsi auto-engineering \
  --workspace /path/to/fresh/lab-run \
  --isaac-python /path/to/isaac-env/bin/python \
  --episodes 3
```

The command executes one native development episode, then (if a repair is
proposed) three paired validation layouts for each policy. Outputs include:

- `index.html`: local video report, diagnosis, scores and selected revision.
- `round-001/development/`: RGB, depth, USD scene, video and native evidence.
- `round-001/episodes/`: parent/child validation recordings on identical layouts.
- `round-001/selection.json`: verified paired selection, when a comparison exists.
- `state/history/`: immutable lineage and source-bound evidence.

A tie retains the parent. A failed process aborts the round; it is not counted
as an ordinary policy failure. Existing workspaces are not silently rerun.
Inspect their evidence and use a new workspace for a new run. `--gui` requests
a visible Isaac window. To rebuild the report from completed evidence:

```bash
./physicalrsi auto-engineering --workspace /path/to/lab-run --report-only
```

For an individual native episode, write a memory JSON containing
`{"transit_height_m": 0.36, "sample_height_m": 0.12}` and run:

```bash
/path/to/isaac-env/bin/python -m PhysicalRSI_demos.auto_engineering.isaac_scene \
  --seed 17 --memory /path/to/memory.json --output /path/to/new-episode
```

## What the result means

This is a **simulation development demo**, not a reproduction of GRID's full
platform, physical deployment, or a benchmark-wide score. Its System 2 is one
explicit, bounded engineering rule authored by Codex: use a detected obstacle's
height and sample dimensions to propose a higher transit path. It does not
claim unrestricted autonomous code synthesis or model training.

System 1 receives color detections from RGB, rendered metric depth, camera
calibration and robot proprioception. Color labels intentionally make this
first demo inspectable. It does not use a learned general-purpose detector or
RGB-only depth estimation. Simulator object poses are restricted to evaluation and presentation recording;
they are never supplied to the policy. The
policy runs as reviewed in-process code; this boundary is not an OS sandbox
attestation.

The evaluator checks the correct sample's destination, upright orientation,
release, settling, and disturbance of the beaker and distractor. Beaker
translation is a limited obstacle-disturbance check, not exhaustive contact
certification. Detailed render meshes use declared cylindrical collision proxies. The beaker
has a glass render mesh; no liquid is simulated.
There are no teleportation actions or synthetic attachment joints.

Scene generation, policy, perception and evaluator code are frozen by source
hashes for a round. Native receipts retain layout identities and file digests.
The external Isaac/NVIDIA dependency and asset closure is not fully attested.
`qualification` remains `null` even if all declared simulation checks pass.

## Software checks

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/test_auto_engineering.py
```

These check sensor filtering, layout generation, evaluator failure cases and
bounded repair behavior. They do not prove that Isaac Sim initializes or that
a robot completes the task; only native receipts and recordings establish that.

## High-quality success video

A successful native episode records rigid-body poses alongside its joint trace.
Render those measured states with the Isaac environment:

```bash
OMNI_KIT_ACCEPT_EULA=YES /path/to/isaac-env/bin/python \
  -m PhysicalRSI_demos.auto_engineering.high_quality \
  --episode /path/to/successful/episode --output /path/to/new-render \
  --width 3840 --height 2160 --samples 512
```

This uses path tracing, 16 ray bounces, denoising and a high-quality H.264 export.
`--frame 300` renders one recorded frame for a quality/time check; add `--count 3`
to measure consecutive-frame throughput. Lossless PNG frames are retained beside
the video. After interruption, repeat the identical command with `--resume` to
verify and reuse completed frames. A complete NVIDIA driver installation, including the OptiX denoiser
weights, is required for denoising. On a headless container with an unusable
GLX Vulkan ICD, set `VK_ICD_FILENAMES` to the EGL ICD written by the native
harness in its runtime directory (or an equivalent system EGL ICD). The source
receipt and trajectory digests bind the render to the native episode. Recorded body positions and rotations are checked after every captured frame. Physics
is disabled during presentation replay: this is a re-render of measured states,
not a second evaluation or additional success evidence. The original native
video and receipts remain available for comparison.
# Common core research path

The optional `--core-research` path runs this native task through the shared
Trial/Evaluation, owned evidence, recursive research and versioned lineage
contracts. It uses the existing bounded public-clearance repair as its default
control evolver. See [core development](../../docs/core-development.md#isaac-native-acceptance-path)
for evidence semantics, recovery boundaries and provider extension points.

```bash
./physicalrsi auto-engineering --core-research \
  --workspace /path/to/new-isaac-research \
  --isaac-python /path/to/isaac51/bin/python \
  --episodes 3 --rounds 2 \
  --device-registry /path/to/shared-local-device-registry
```

The command produces JSON research receipts and a held-out final report. It
requires native Isaac Sim 5.1; no software physics substitute is used. The
existing command without `--core-research` retains its single-round visual
report workflow. Neither workflow qualifies physical robot execution.

Add `--staged` to run native development screening before paired validation.
The declared pool contains one clearance repair inferred from public development
observations and two authored low-clearance negative controls (0.10 m and 0.20 m).
They test whether screening rejects ineffective candidates; they are not learned
proposals. Each child receives one screening rollout. The incumbent is marked
unmeasured during screening and receives full paired validation alongside
surviving children. Add `--screen-incumbent` to also collect its diagnostic
screening rollout. A local queue permits one native worker at a time; all clients must
still share the device registry. Rejected branches and screening evidence remain
in the candidate archive. See the [staged acceptance protocol](../../docs/core-development.md#native-development-screening)
for split identities, trial limits and the meaning of the result.

Add `--feedback-search` with `--core-research` to search a bounded carrying-height
parameter using actual development outcomes, including previously rejected
screening probes. This mode implies staged evaluation with one candidate per
round. It uses known failure seed 1000 for development and separate preregistered
validation/test seeds. A grasp-like joint aperture and measured obstacle
disturbance support the clearance hypothesis; insufficient evidence or
contradictory outcomes stop that search. The field bounds and monotonicity
assumption are authored. See [parameter search](../../docs/core-development.md#development-feedback-parameter-search)
for the exact split protocol and limitations.

```bash
./physicalrsi auto-engineering --core-research --feedback-search \
  --workspace /path/to/new-isaac-feedback-study \
  --isaac-python /path/to/isaac51/bin/python --episodes 3 --rounds 2 \
  --device-registry /path/to/shared-local-device-registry
```

## Declared two-parameter empirical search

The empirical path explores carrying height and grasp depth using archived
native development measurements. It supports adaptive or uniform proposals,
explicit per-round validation panels, repeated execution, and preregistered
screening thresholds. Create a JSON plan such as:

```json
{
  "strategy": "adaptive",
  "seed": 20261028,
  "population": 2,
  "repetitions": 2,
  "screening_minimum_success_rate": 0.5,
  "shared_cache": true,
  "panels": {
    "development": [1000, {"seed": 1001, "scenario": "tall-obstacle"}],
    "validation": [[2000, {"seed": 2001, "scenario": "tall-obstacle"}],
                   [2010, {"seed": 2011, "scenario": "tall-obstacle"}]],
    "test": [3000, {"seed": 3001, "scenario": "tall-obstacle"}]
  }
}
```

Prepare a new workspace and inspect its Trial allowance without launching
Isaac or reserving any executions:

```bash
python -m PhysicalRSI_demos.auto_engineering.empirical \
  --workspace /path/to/empirical-search \
  --python /path/to/isaac-env/bin/python \
  --plan /path/to/plan.json --prepare-only
```

The example declares a maximum of 52 Trials. Remove `--prepare-only` to run
search followed by the held-out final test. Repeating the same command reuses
completed evidence; uncertain executions require reconciliation through the
core APIs. The command freezes the exact JSON options, launcher digest and
demo source digests in `empirical-plan.json`; changed declarations require a
new workspace. Use a fixed source checkout for reproducible core behavior.
Preparation inspects configuration and budget, not native runtime readiness.
On an existing workspace, the returned quota includes previous reservations.

A threshold of 0.5 only admits partial development results to independent
validation; full-success promotion and the seven final task checks remain
unchanged. Repeated executions count against the allowance and do not create
additional independent cases. To share a GPU with other cooperating runs,
include the same `device_registry` path and `resource` name in every plan.
This command runs one search arm. For a multi-arm study, freeze matching plans
and use `--search-only` for every arm first. After all searches complete, use
`--test-only` with each arm's unchanged plan and workspace. These modes are
mutually exclusive with `--prepare-only`. `--test-only` rejects an unfinished
search before dispatching a provider or Trial; it does not resume that search.
The core equivalent is `ResearchCampaign.report_test(initial,
require_completed=True)`. It still audits completed evidence before testing.
The caller remains responsible for waiting for every arm before testing any arm.
