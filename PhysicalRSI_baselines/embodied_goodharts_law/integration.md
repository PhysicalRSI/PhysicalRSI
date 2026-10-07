# ASPIRE reference and benchmark integration

## Confirmed scope

The requested pipeline is System 2 → GPT System 1 → Code-as-Policies (CAP),
with ASPIRE as the implementation reference. The three target benchmark families
are LIBERO, LIBERO-Plus and RoboTwin. References below were reviewed on
2026-10-05. Source-review pins are provenance, not tested installation locks:
execution also requires dependency, asset, task and environment identities.

| Project | Reviewed commit | Role |
| --- | --- | --- |
| [ASPIRE](https://github.com/NVlabs/ASPIRE/tree/f4c8939aab0af9b97690c561bd80e282940f7886) | `f4c8939aab0af9b97690c561bd80e282940f7886` | Code-policy and skill-refinement reference |
| [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO/tree/8f1084e3132a39270c3a13ebe37270a43ece2a01) | `8f1084e3132a39270c3a13ebe37270a43ece2a01` | Original task family |
| [LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus/tree/4976dc30028e805ff8094b55501d532c48fec182) | `4976dc30028e805ff8094b55501d532c48fec182` | Separate perturbation-robustness evaluation |
| [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin/tree/ea8b21121ebb3cd201ff5b3fe361944ac94eda3f) | `ea8b21121ebb3cd201ff5b3fe361944ac94eda3f` | Separate bimanual benchmark adapter |

## What to reuse from ASPIRE

ASPIRE's [public API abstraction](https://github.com/NVlabs/ASPIRE/blob/f4c8939aab0af9b97690c561bd80e282940f7886/aspire/sim/cap/integrations/base_api.py)
exposes named callable functions and their documentation. Its
[Franka LIBERO code environment](https://github.com/NVlabs/ASPIRE/blob/f4c8939aab0af9b97690c561bd80e282940f7886/aspire/sim/cap/envs/tasks/franka/franka_libero_env.py)
is a concrete place to inspect code execution and environment integration.
Reuse these concepts through an explicitly versioned adapter; do not present a
raw generated-code string as a validated action or an inherited skill.

The proposed EGL boundary is:

```text
System 2: development evidence → proposed prompt / skill / memory revision
                                  ↓ freeze and identify
GPT System 1: task + admitted observations → code / skill invocation
                                  ↓ bounded CAP execution
Benchmark adapter: admitted actions → simulator → recorded observations
                                  ↓
Official proxy evaluator + separate independent task audit
                                  ↓
Comparison and lineage through physicalRSI Self-Harness
```

Define whether GPT generates code once per episode or revises it during the
rollout. Both are System 1 behavior if used to choose actions inside that
rollout; cross-trial artifact refinement and candidate selection are System 2.
Record each generated program and its digest even when the enclosing prompt
and skill library are unchanged. Pin model settings and provider identifiers;
stochastic inference does not become deterministic just because it is logged.

ASPIRE's published workflows include LIBERO-Pro development and LIBERO transfer.
Its [suite guide](https://github.com/NVlabs/ASPIRE/blob/f4c8939aab0af9b97690c561bd80e282940f7886/aspire/sim/.claude/libero/CLAUDE.md)
distinguishes those task families. Do not relabel LIBERO-Pro as LIBERO-Plus or
claim that an ASPIRE quick start covers the three requested families.

The [ASPIRE README](https://github.com/NVlabs/ASPIRE/blob/f4c8939aab0af9b97690c561bd80e282940f7886/README.md)
describes generated Python execution with full import access; its worker
processes are not a security sandbox. physicalRSI's existing stdlib-only
isolated policy cannot be claimed as a drop-in host for the full ASPIRE stack.
A concrete integration must separate the candidate execution surface from
simulator internals, verifier state and result storage, and record the actual
isolation profile. Source review alone does not establish ASPIRE execution;
the native preflight status is recorded below.

## Work per benchmark

| Family | Adapter work | Evaluation and audit work |
| --- | --- | --- |
| LIBERO | Bind concrete suites/tasks, initial states, camera observations, action semantics and CAP calls to the original pinned evaluator. Check ASPIRE wrapper compatibility rather than assuming it. | Preserve the official success predicate. Define a separate task-specific audit; inspect candidate behavior when the two disagree. |
| LIBERO-Plus | Use a separate environment/configuration from original LIBERO. Bind each task to its perturbation category and assets. | Follow its own official protocol and report category-wise coverage. Keep additional repeated-seed research results separately labeled. |
| RoboTwin | Implement and verify a bimanual CAP bridge, embodiment mapping, task configuration, reset and official evaluator integration. Existing RoboDojo integration is a reference, not proof of RoboTwin support. | Pin tasks and conditions. Audit the intended object relations and task completion independently of the official result. |

The [LIBERO-Plus README](https://github.com/sylvestf/LIBERO-plus/blob/4976dc30028e805ff8094b55501d532c48fec182/README.md)
describes seven perturbation dimensions and an evaluation protocol with one
trial per task. It installs the `libero` package, so the original and Plus
variants must not silently replace one another in a shared environment.
Different language instructions and perturbed states are real test conditions,
not interchangeable seed labels.

The [RoboTwin README](https://github.com/RoboTwin-Platform/RoboTwin/blob/ea8b21121ebb3cd201ff5b3fe361944ac94eda3f/README.md)
describes RoboTwin 2.0 and XPolicyLab deployment. This makes the existing
XPolicyLab integration worth inspecting, but does not establish CAP compatibility
or eliminate benchmark-specific contract checks.

## First milestone and claims

Start with one declared task per family to validate reset, generated-program
provenance, action traces and both outcome channels. Then freeze the wider task
population, budgets, development cases and held-out audit before searching for
exploits. None of those execution milestones has been completed in this directory.

Distinguish three findings:

- **Genuine task success:** CAP completes the intended task and receives the
  official score; this establishes capability, not Goodhart failure.
- **Restricted generalization:** a policy scores well but fails under new
  conditions; report robustness limits without automatically calling it hacking.
- **Confirmed proxy exploitation:** behavior earns official credit while failing
  the independently specified task, with an investigated evaluator loophole and
  a targeted confirmation experiment.

An independent criterion must follow the task specification. Additional human
preferences, such as motion style or energy efficiency, cannot retroactively
turn valid benchmark success into an exploit unless those requirements were
part of the intended task being tested.

## LIBERO reset correction

The reviewed ASPIRE source has two reset inconsistencies: `LiberoHandle.reset`
returns observations captured before applying its initial state, and
`FrankaLiberoEnv.reset` calls `reset()` after selecting an initial state. The
latter can discard the selected layout. The local
[reset patch](patches/aspire-libero-reset.patch) retains observations returned by
`set_init_state` and removes that subsequent reset. Upstream settling behavior
is unchanged. This is a local adaptation, not an upstream fix or a claimed
benchmark improvement.

Apply to the pinned ASPIRE checkout, using absolute paths:

```bash
git -C /path/to/ASPIRE apply --check \
  /path/to/PhysicalRSI/PhysicalRSI_baselines/embodied_goodharts_law/patches/aspire-libero-reset.patch
git -C /path/to/ASPIRE apply \
  /path/to/PhysicalRSI/PhysicalRSI_baselines/embodied_goodharts_law/patches/aspire-libero-reset.patch
python PhysicalRSI_baselines/embodied_goodharts_law/check_aspire_reset.py \
  --aspire-root /path/to/ASPIRE
```

The check executes the two actual reset method bodies from the trusted checkout
against a small stateful fixture. It first reproduces the original source's
failure, then verifies that both selected initial states and their observations
survive the patched reset. It does not import or run a simulator. A separate native check has now passed using the actual ASPIRE
`FrankaLiberoEnv` wrapper and its pinned LIBERO-PRO fork: seeds 1 and 2
selected state indices 0 and 1, with exact state and observations before the
unchanged 10 settling steps. This is not original-LIBERO or LIBERO-Plus
qualification. Record the patch digest alongside the upstream commit in every
subsequent experiment.

## Native development-layout capture

`sample_libero_layout.py` is the initial trusted layout-generation entry point
for original LIBERO and LIBERO-Plus. It requires a dedicated interpreter with
that benchmark's native dependencies. It checks the source revision and import
location, uses a private path configuration, resets the real environment and
stores model XML, simulator state and public camera/proprioception observations
with the core artifact store. A parent process imposes a deadline on reset,
which upstream otherwise retries without a bound.

From the physicalRSI repository root:

```bash
/path/to/libero/python -m PhysicalRSI_baselines.embodied_goodharts_law.sample_libero_layout \
  --benchmark libero --source /path/to/LIBERO --suite libero_spatial \
  --task-id 0 --seed 100000 --seconds 180 \
  --output /tmp/egl-libero-development-layout-100000
```

Use a fresh output directory per attempt. `generation.json` and `worker.log`
retain completion, failure or timeout; `layout.json` is published only after
capture and simulator cleanup. For Plus use `--benchmark libero-plus` and its
separate source/environment/assets. The official initial-state evaluation
protocol is not replaced by these generated development conditions.

This command passed a native original-LIBERO run on DSW worker 01 using
`libero_10`, task 0 and seed 1000. Both camera images, depth, proprioception,
model XML and simulator state were captured; initial official success was false.
The tested platform adaptation uses Python 3.12, MuJoCo 3.3.0, Robosuite 1.4.0,
NumPy 1.26.4, `future==0.18.2` and PyOpenGL 3.1.10. PyOpenGL 3.1.6 failed at
the EGL ctypes boundary on this interpreter. This verifies layout capture only;
ASPIRE's separate LIBERO-PRO reset check is described above. Plus capture subsequently
passed on worker 08 for `libero_10`, task 0, seed 1400, with initial official
success false. Its task was
`KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_table_1`.
Plus additionally required the system ImageMagick library (`libmagickwand-6.q16-6`).
Its verified assets were expanded to local disk, and the external checkout's
`libero/libero/assets` symlink points to that per-node location. When using
`--asset-root`, this source-relative path must resolve to the same directory:
native arena code reads it directly, independently of the path configuration.
It does not implement
RoboTwin sampling, novelty against a complete campaign history, or candidate selection.
The manifest explicitly records `novelty: not-yet-compared`; different seeds
alone do not prove different layouts. Private reset state/XML/object placements are infrastructure
artifacts, not inputs to GPT or generated CAP skills. Full environment locking
must be completed before campaign integration. Replay verification is available
as the separate preflight below.

A native follow-up on this original-LIBERO case rebuilt the pinned task,
restored the captured state and reproduced all stored observations exactly,
then executed ten zero actions. Rebuilding from `mj_saveLastXML` output instead
preserved the state vector but changed the front RGB image by up to 18 intensity
levels and the end-effector quaternion by about 1.1e-7. Consequently the XML
artifact is not yet a qualified exact-replay format. A future replay adapter
must verify that its rebuilt model matches the captured layout, or restore a
lossless model representation; state-vector equality alone is insufficient.

`verify_libero_layout.py` implements the first of those checks:

```bash
/path/to/libero/python -m PhysicalRSI_baselines.embodied_goodharts_law.verify_libero_layout \
  --layout /tmp/egl-libero-development-layout-100000/layout.json \
  --task-id 0 --seconds 180 --output /tmp/egl-libero-replay-100000
```

It verifies artifact hashes, the pinned source and task definition, rebuilds
the captured model using the capture seed, requires identical compiled XML,
runs ten zero-action physics steps and restores the saved state. The restored
state and all nine public camera/proprioception channels must match exactly;
the initial official success predicate must also agree. It never reloads the
exported XML. A timeout or mismatch leaves a failure journal and worker log,
without a verified result. Private model/state files stay in the trusted
simulator process.

This check passed on the LIBERO and Plus cases above. An earlier probe found
that a subsequent Plus reset with another seed changed the compiled model;
state-only restoration across arbitrary resets is therefore inadmissible.
Reconstruct and verify the model for each candidate. These checks cover the
tested environment only: full simulator dependency closure still needs separate
locking, and exact replay is not guaranteed across GPU or
simulator versions. They do not qualify ASPIRE's wrapper or count as task trials.

New captures also include content-addressed private object-root poses and a
geometry digest. The replay preflight verifies these poses when present. To
compare two captures from the same pinned task:

```bash
python -m PhysicalRSI_baselines.embodied_goodharts_law.compare_layouts \
  --left /tmp/egl-layout-a/layout.json --right /tmp/egl-layout-b/layout.json \
  --output /tmp/egl-layout-comparison
```

This requires NumPy but does not import a simulator. It checks the artifact and
geometry digests, then compares object positions and orientations with 1 mm and
0.5 degree thresholds. It excludes robot motion, elapsed simulation time and
image noise from the placement criterion. A changed texture, articulated joint,
language instruction or object population requires a separately declared
condition; this conservative comparator does not call those changes new object
placements. Geometry and comparison details stay with the trusted layout
generator and evaluator, outside candidate observations and memory.

Native checks on both task-0 cases captured seeds 2000, 2000 and 2001. Repeated
seed 2000 produced identical placements; seed 2001 moved multiple objects, with
a largest displacement of approximately 0.0465 m in each family. The two
comparison reports preserve their input digests. These pairwise checks do not
establish novelty against every previous campaign layout, replace held-out
evaluation, or demonstrate improved policy performance.
The seed-2000 captures were also replayed on different assigned nodes with the
same GPU family: worker 05 to 01 for LIBERO and worker 12 to 08 for Plus. Compiled
models, state, object-root poses and all nine public observation channels matched
exactly in those two checks. This is evidence for those node pairs, not a
qualification of the entire fleet.

New captures also inventory every external mesh/texture file referenced by the
compiled model. `model_assets.py` requires absolute references contained within
the prepared benchmark or Robosuite asset directories, records sizes and SHA256
hashes, and binds the inventory digest to the layout identity. Replay checks
those file contents as well as XML, state, object placements and observations.
Legacy captures without the inventory explicitly report that asset verification
was unavailable. Relative references and symlinks escaping the declared asset
directories are rejected rather than resolved by guessing a working directory.

Native capture and replay with this inventory passed on both task-0 cases at
seed 2300, covering 87 unique files for LIBERO and 88 for Plus. Separate checks
confirmed that changed file bytes change the inventory and that relative paths
and escaping symlinks are rejected. The inventory covers referenced model
assets; it does not freeze all installed libraries or all assets in the dataset.

## Native primitive bridge

`libero_primitives.NativeLiberoPrimitives` keeps the native simulator in the
trusted host and supplies five explicit callbacks to the existing isolated
code-policy bridge: `get_observation`, `move_to_joints`, `open_gripper`,
`close_gripper` and `goto_home_joint_position`. The admitted controller is the
native seven-joint fixed-impedance delta controller with a 0.05-radian scaling
range. Targets must be finite and inside native joint limits. Every physics
step consumes a shared allowance; motion also has a convergence limit and
callbacks check the caller's deadline between steps. An enclosing process
supervisor must still handle a native simulator call that hangs.

Observations contain two calibrated RGB-D cameras using ASPIRE's camera
convention and robot proprioception. `robot_cartesian_pos` is the native
`robot0_right_hand` body pose relative to `robot0_base`, encoded as position,
quaternion in wxyz order, and gripper fraction. It is not a tool-center pose.
They exclude object poses, simulator state,
segmentation labels and evaluator outcomes. This RGB-D/calibration access must
be declared explicitly before benchmark comparisons; it is not automatically
equivalent to another policy's observation contract. The trusted object must
never be passed to generated Python. Its `handlers` mapping is the only
candidate-callable surface.

On the LIBERO and Plus task-0 cases, a native probe moved joint 0 by 0.02 radians,
closed and opened the gripper and returned home, with 68 recorded physics steps.
A handwritten isolated Python policy then called all five capabilities and
completed its sequence through the existing Linux chroot/seccomp bridge. These
are interface preflights, not GPT-generated solutions or task success scores.
The implementation's camera schema has been exercised, but equivalence to
ASPIRE's complete perception/IK stack remains unverified. SAM3, Contact-GraspNet,
inverse kinematics, candidate generation and Self-Harness selection still need
end-to-end integration.

### Hand-frame and IK preflight

The initial adapter inherited a fixed -0.107 m offset from the gripper EEF.
That produced a pose about 10.5 mm from the standard Panda URDF hand frame.
The adapter now reads the native hand body directly. Native checks passed on
one task in each LIBERO family at three joint configurations, including
joint-0 movements of ±0.15 radians. Base-to-world pose round trips matched
the native hand; rotation matrix differences were below 1e-6. Earlier
primitive and paired-selection receipts retain their original adapter hashes
and must not be interpreted as checks of this revised pose contract.

ASPIRE's PyRoKi solver also passed two reachable-target checks with its default
Panda model from `Gepetto/example-robot-data`, pinned to
`a0d3281d19fc5239a255ef69a6afc40aae55963c`, file
`robots/panda_description/urdf/panda.urdf`, SHA-256
`63792d2679f22c4e41cda2c4d9d903644e628970ad0656b9eec8485113941555`.
Those targets and their verification used the same URDF. A different Panda
URDF bundled in Robosuite's Bullet assets was rejected after its native-frame
comparison showed a roughly 0.36 m discrepancy.

The default model still requires an explicit hand-frame conversion: native
link-7-to-hand translation is 0.1065 m and its quaternion is rounded, whereas
the URDF uses 0.107 m and an exact -π/4 rotation. Applying the fixed transform
derived from those model definitions yielded a maximum position discrepancy
of 0.024 mm and orientation discrepancy of 0.000052 radians across the six
native configurations. This is a limited kinematic preflight, not validation
over the workspace, collision planning, or IK-driven task execution. The
solver ran on CPU; it is not evidence of GPU utilization. ASPIRE's separate
TCP offset must not be applied implicitly to this native-hand observation.

A subsequent host diagnostic executed ASPIRE `solve_ik_rest` solutions in both
native environments. It used the current joints as the solver seed, zero rest
cost, and the explicit model-hand conversion to request a 5 mm upward hand
translation. The initial 0.0005-radian joint convergence requirement failed
within 120 physics steps. Those failures are retained. A LIBERO diagnostic
measured 0.00191 radians of residual joint error and 0.275 mm Cartesian error.

The follow-up declared a 0.002-radian joint tolerance and kept the original
1 mm position and 0.005-radian orientation thresholds. On fresh seed-3400
layouts, both task-0 cases converged in 101 steps: actual upward displacement
was 4.708 mm, position error 0.292 mm and orientation error 0.000821 radians.
Initial public robot state was checked against the solver input, and targets
were recorded before movement. This verifies one small IK-driven native motion
per family. It does not establish collision-aware planning, grasping, task
success, or a generated policy's use of an IK service.

### Optional isolated-policy pose capability

[AspireHandIK](aspire_ik.py) runs the ASPIRE solver in a separately configured
interpreter, verifies the pinned URDF bytes, converts the declared native hand
frame, and checks the solved pose and joint limits. The trusted environment
factory supplies its interpreter, model path and native link-7-to-hand transform.
Record `solver.identity()` with the environment identity; interpreter paths
alone do not attest the installed dependency closure.

Pass this solver as `pose_solver` when constructing `NativeLiberoPrimitives`.
Only that configured instance exposes `move_to_pose` to isolated code. The
candidate-facing call accepts seven numbers: hand xyz in metres followed by a
unit wxyz quaternion, all relative to the robot base. It does not change the
gripper command. Example inside a candidate primitive:

```python
state = robot.get_observation()
target = state["robot_cartesian_pos"][:7]
target[2] += memory["height_delta"]
result = robot.move_to_pose(target)
```

The host owns the 0.002-radian joint convergence tolerance, 1 mm Cartesian
position tolerance and 0.005-radian orientation tolerance. Invalid targets,
expired deadlines and exhausted action budgets fail before native movement.
The solver subprocess is bounded by the callback deadline and a configured
timeout. Joint convergence does not override a failed Cartesian check.
`pose_trace` records the target, solver output, native endpoint and completion
or failure; retain it in trusted experiment measurements alongside the physics
action trace. The existing Recorded CAP adapter records the policy request
before invoking this callback. This interface provides no collision planning
and requires the surrounding experiment supervisor for blocking native calls.

A reviewed candidate now exercises this capability through actual isolation
and Recorded CAP on original LIBERO task 0, seed 3500. Its memory contains a
5 mm height increment; the primitive reads the public observation and calls
`move_to_pose`. ExperimentRuntime recorded observation, pose request and finish,
including 101 native physics steps and 0.292 mm final position error. The
official task predicate remained false, and the trial correctly records
**failure** despite successful movement. Policy shutdown and GPU lease
quiescence passed. This is one original-LIBERO diagnostic, not a GPT-generated
policy, Plus capability qualification, or an admitted Self-Harness round.

The first attempt failed before movement because resolving the virtual
environment's Python symlink selected the base interpreter without JAX. The
solver now preserves the virtual-environment executable path. Its interrupted
lease was reconciled only after checking owner inactivity, worker/solver process
exit and an empty GPU compute-process list. Both failure and reconciliation
records remain separate from the successful execution diagnostic.

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  tests/test_egl_pose_boundary.py tests/test_egl_recorded_cap.py \
  tests/test_egl_candidate_program.py
```

These 13 software checks include invalid pose rejection before solving,
deadline expiry before movement, rejecting Cartesian failure after joint
convergence, and preserving the virtual-environment interpreter path. Native
execution is established by the separate diagnostic described above.

## DSW execution transport

### Installed-provider byte inventory

[environment_inventory.py](environment_inventory.py) captures installed
distribution versions, actual file hashes, interpreter identity and editable
source references using the provider's own Python. It reads wheel `RECORD`
directly because some Python versions silently omit missing files from the
distribution file API. It reports missing files and RECORD hash/size differences
without treating the installed environment as admitted.

```bash
/path/to/provider/bin/python \
  PhysicalRSI_baselines/embodied_goodharts_law/environment_inventory.py \
  --output /tmp/egl-provider-inventory.json
```

Use a fresh output path and freeze installations during capture. The command
hashes potentially tens of gigabytes. It excludes regenerated bytecode and
does not cover stdlib, external libraries, editable source trees, model assets
or runtime configuration. `complete_execution_closure_attested` remains false;
an inventory is evidence for admission, not admission itself.

Native inventories now cover four provider interpreters:

| Environment | Distribution entries | Distinct installed file paths |
| --- | ---: | ---: |
| LIBERO | 518 | 111,683 |
| LIBERO-Plus | 524 | 112,449 |
| RoboTwin | 159 | 38,381 |
| ASPIRE adapted environment | 711 | 154,627 |

Distribution counts include overlapping system and environment installations.
The initial byte inventories were supplemented with a direct RECORD presence
check; they were not retroactively relabelled as the corrected inventory tool's
output. RoboTwin's listed files had no hash or size differences. LIBERO and Plus
each had 106 discrepancy entries, and ASPIRE had 108; hash and size findings
may refer to the same file. In each affected environment, 102 entries matched
another installed distribution's RECORD. This establishes overlapping byte
ownership, not which version Python or the native loader selects. Remaining
differences include `decord` and `ipykernel`. Editable source trees and inherited
system packages also require explicit admission review. Existing native runtime
checks remain valid for their recorded scope, but these inventories do not
upgrade the diagnostic candidate closure to campaign admission.

An actual import/load probe subsequently passed in all four environments.
Each imported PyTorch 2.10.0+cu128 and executed a CUDA tensor computation on
the assigned Blackwell GPU. LIBERO and Plus loaded OpenCV 4.11.0; the adapted
ASPIRE interpreter loaded OpenCV 4.12.0, JAX 0.4.29 and its local PyRoKi.
RoboTwin imported SAPIEN 3.0.0b1 and the patched CuRobo source checkout.
The probes recorded module paths and hashes plus 180, 180, 106 and 255 mapped
file identities for LIBERO, Plus, RoboTwin and ASPIRE respectively. These
mappings cover this probe, not everything a full simulator episode can load.

The remaining unmatched installed-file differences were `decord`'s
`top_level.txt` package metadata and Jupyter's `kernel.json`, rather than code
loaded by this probe. No inherited system package was overwritten to conceal
those differences. The installed inventories, actual runtime selection and
local adaptations must remain distinguishable in the final environment lock.

[source_inventory.py](source_inventory.py) complements installed-file capture
with actual tracked source bytes and working-tree status:

```bash
python PhysicalRSI_baselines/embodied_goodharts_law/source_inventory.py \
  --root /path/to/pinned/provider-checkout \
  --output /tmp/egl-provider-source.json
```

It retains commit and index identities while hashing local modifications.
Submodule contents, untracked/ignored build outputs and symlink targets are
explicitly excluded and need separate evidence. A source hash inventory alone
does not admit the provider. Regression checks exercise modified tracked bytes,
missing files, untracked files and a symlink whose target is unavailable:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  tests/test_egl_source_inventory.py tests/test_egl_environment_inventory.py
```

The source check also distinguishes uninitialized submodules from initialized
checkouts: Git can otherwise return a parent repository's HEAD when invoked in
an empty submodule directory. A regression test covers that case. ASPIRE's
uninitialized `b1k` submodule is recorded as absent, not as an installed provider
or a benchmark qualification.

Contact-GraspNet's three modified tracked files were verified byte-for-byte
against ASPIRE's own bundled compatibility patch and verification script at the
reviewed ASPIRE revision. The changes explicitly select legacy checkpoint
loading, use `torch.linalg.cross` and qualify the mesh utility import. They are
ASPIRE-supplied adaptations, not unexplained EGL edits. Their patch, verification
script and actual source-file hashes are retained separately from the pinned
Contact-GraspNet commit. The checkpoint still has its separately verified file
identity; this source check does not establish semantic perception or grasping.

The official [PAI CLI quick start](https://www.alibabacloud.com/help/en/pai/developer-reference/quick-start)
provides `pai dsw instance exec`. This avoids requiring SSH key installation on
workers when the caller already has the required DSW account permissions.
[Configuration documentation](https://www.alibabacloud.com/help/en/pai/developer-reference/configuration-management)
describes the instance credential chain; do not copy temporary tokens into the
study manifest or generated-policy workers.

```bash
PAI_REGION=cn-shanghai PAI_WORKSPACE_ID=your-workspace \
  /path/to/pai dsw instance exec your-instance --timeout 30 -- \
  'printenv DSW_INSTANCE_ID; nvidia-smi'
```

Check returned instance identity, GPU UUID, current processes and shared paths
before assigning a worker. The current setup has used PAI CLI v1.0.1 to verify
18 distinct single-GPU instances and execute a small CUDA kernel on each. This
establishes transport/basic CUDA readiness only, not simulator or CAP readiness.
Runtime inventories and assignment files remain outside the repository; refresh
them before dispatch because GPU availability can change.

Install environments and package caches on each DSW's local disk. The first
RoboTwin worker installed its simulator dependencies in about 15 seconds there;
the shared-filesystem installation spent minutes unpacking the same packages.
Keep pinned sources and durable evidence on the shared filesystem.

`provision_node.py` checks the expected instance and single GPU UUID before
installing the frozen native dependency lists from the shared runtime's
`config/libero-native-constraints.txt` or `config/robotwin-native-constraints.txt`.
Those lists were captured from the verified node environments. For example:

```bash
python PhysicalRSI_baselines/embodied_goodharts_law/provision_node.py \
  --family libero --instance your-instance --gpu-uuid your-gpu-uuid \
  --local-root /mnt/workspace/egl --shared-root /path/to/shared/egl_runtime
```

Invoke on the assigned worker through PAI. A per-family lock prevents concurrent
environment writes. `preflight/nodes/<instance>.json` records dependency
installation and its input hashes; `installed` does not mean benchmark-ready.

An optional `--wheelhouse /path/to/verified/wheels` reuses local wheel files
without changing the frozen requirements. The directory must contain only
`manifest.json` and its listed wheels. The manifest declares `state: verified`
and a `wheels` list with `file`, `bytes` and `sha256` for every wheel. Provisioning
checks those sizes and hashes before handing the directory to uv, then records
the manifest digest in the installation receipt. The trusted operator owns this
bundle; a candidate policy cannot supply it.

For this fleet, Robosuite 1.4.0 and OpenCV 4.11.0.86 were repacked from complete
cache entries on a verified node. Every payload file was checked against its
original wheel `RECORD` before packing and against the resulting archive after
packing. Archive hashes therefore describe the repacked files, not the original
registry archives. A fresh offline environment installed the frozen LIBERO
requirements and captured a real layout in approximately 19 seconds. Six
unfinished dependency installations then switched to this verified bundle;
their original receipts, explicit installer termination and replacement results
were preserved. No simulator or candidate-policy worker was replaced.

All 18 assigned nodes subsequently passed their family-specific native
preflight: six original-LIBERO captures, six Plus captures, and six SAPIEN
physics/rendering probes. Only the first two checks execute benchmark task
resets. The SAPIEN probes do not establish complete RoboTwin task readiness,
and none of these checks establishes a running policy campaign.

Worker 13 passed a SAPIEN 3.0.0b1 physics-step and 64x64 RGBA rendering smoke
check. Its Python 3.10 environment needs `setuptools==80.9.0` for SAPIEN's
`pkg_resources` import. The container's default Vulkan path failed; a
process-local `VK_ICD_FILENAMES` manifest pointing to its installed
`libEGL_nvidia.so.580.126.09` enabled rendering. NVIDIA documents the EGL ICD
alternative in its [driver component reference](https://download.nvidia.com/XFree86/Linux-x86_64/580.126.09/README/installedcomponents.html).
Discover and verify the installed driver version on other workers before
configuring them. This smoke check does not execute a RoboTwin benchmark task.

ASPIRE's installed upstream CUDA 12.6 environment failed a real kernel check on
these Blackwell GPUs: Torch 2.12.1+cu126 had no `sm_120` kernel. A separate platform
environment now passes a real CUDA kernel and the native LIBERO-PRO reset
check using Torch 2.10.0+cu128 and Torchvision 0.25.0+cu128. The original lock
file is retained. This does not validate perception, IK, or a benchmark policy. The upstream native pytest attempt lost its gateway response and no
test result was recovered, so it is not counted as a pass.

Contact-GraspNet also passed a separate CUDA inference preflight with the
verified checkpoint (`39fc3439d5814043ba64e0715c127c2e8aca6ea376c22614e11af1bc89317762`).
A seeded synthetic 4,096-point cloud produced 200 finite grasp candidates on
Torch 2.10.0+cu128, using approximately 2.55 GB of peak allocated CUDA memory.
This checks model execution only; it does not validate semantic segmentation,
real scene grasp quality, grasp execution, or benchmark success.

## Codex generation and local SAM3 checkpoint

The user selected the current Codex session to author CAP candidates, replacing
the earlier requirement for a separate GPT API configuration. Each generated
candidate retains its source, memory, binding hash, development evidence and
generation provenance. This is Codex-assisted candidate generation; it does not
claim an external API response, known provider revision, or a language-model
call inside every episode. System 2 remains responsible for evaluating changes
and recording selection before inheritance.

A local checkpoint was found through the sibling kai-aleph0 asset manifest at
`../rdj_master/policy_submission/frozen/rdj_rgb/assets/sam3.pt`. Its size is
3,450,062,241 bytes and SHA-256 is
`9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`.
The local copy is verified before loading. No new gated download is needed.
The [local checkpoint patch](patches/aspire-sam3-local-checkpoint.patch) adds
`--checkpoint-path` to ASPIRE's SAM3 service while preserving its default
download behavior when the option is absent. With an explicit local checkpoint,
the provider disables Hugging Face loading and uses the requested device.

```bash
git -C /path/to/ASPIRE apply --check \
  /path/to/PhysicalRSI/PhysicalRSI_baselines/embodied_goodharts_law/patches/aspire-sam3-local-checkpoint.patch
git -C /path/to/ASPIRE apply \
  /path/to/PhysicalRSI/PhysicalRSI_baselines/embodied_goodharts_law/patches/aspire-sam3-local-checkpoint.patch
/path/to/aspire/python -m aspire.sim.cap.serving.launch_sam3_server \
  --device cuda --host 127.0.0.1 --port 8114 \
  --checkpoint-path /path/to/verified/sam3.pt
```

These commands start perception inference, not a benchmark campaign. The local
checkpoint passed GPU inference on an archived public LIBERO RGB image at about
6.08 GB peak allocated memory. The three prompts produced 1, 4 and 1 detections
for alphabet soup, tomato sauce and basket respectively; detections are fallible
and are not independently verified object identities.

[Sam3Client](sam3_client.py) provides a loopback-only, deadline-bounded image
client with image, response-size, mask-shape and score checks. A configured
`NativeLiberoPrimitives` instance exposes `segment_objects(prompt, camera)`;
it reads only a public camera image. Candidate programs receive no simulator
object poses or evaluator state. The first Codex candidate combined masks with
public depth and calibration to estimate visible surface positions. On fresh
seed 3600 it executed through Recorded CAP with five recorded requests and no
robot motion. Both can labels selected the same position and the basket was
not detected. Official task outcome remained failure, with policy shutdown and
lease quiescence verified. These development findings require further candidate
refinement; they do not establish task success or a completed Self-Harness round.

## RoboTwin planner platform adaptation

The pinned CuRobo revision (`d64c4b005459db10c5dd867d8b30a87d5bda9bdb`)
compiles with Python 3.10 development headers, CUDA 12.8 and
`TORCH_CUDA_ARCH_LIST=12.0`. Compilation alone does not establish task readiness.
Native task initialization exposed a removed Warp namespace with both Warp
1.18.0 and ASPIRE's locked 1.14.0: CuRobo called
`wp.torch.device_from_torch`, while 1.14.0 exposes `wp.device_from_torch`.
The one-line [compatibility patch](patches/curobo-warp-device.patch) updates
that call on the separately pinned CuRobo checkout. It does not change
RoboTwin's tasks or success predicates. Complete `blocks_ranking_rgb` task
initialization now passes on worker 13 with `demo_clean`, seed 1000 and
Aloha-AgileX, including both native planners and four nonconstant RGB-D camera
streams. Initial official success was false. This single-task reset is not a
RoboTwin score or fleet-wide task qualification. Record the patch digest with every
subsequent environment receipt.

```bash
git -C /path/to/curobo apply --check \
  /path/to/PhysicalRSI/PhysicalRSI_baselines/embodied_goodharts_law/patches/curobo-warp-device.patch
git -C /path/to/curobo apply \
  /path/to/PhysicalRSI/PhysicalRSI_baselines/embodied_goodharts_law/patches/curobo-warp-device.patch
```

These commands check applicability and apply the compatibility change; they
do not compile CuRobo or run a benchmark.

## RoboTwin CAP primitive boundary

[NativeRoboTwinPrimitives](robotwin_primitives.py) keeps the task in the trusted
host and exposes explicit observation, proprioception, joint-target, Cartesian
pose, gripper and home callbacks. Commands use the unmodified official
`take_action` interface. The adapter validates target dimensions, finite values,
joint limits, normalized pose quaternions and gripper ranges. It bounds action
attempts and records each command plus the native action counter. This is an
action budget, not a count or bound on individual physics steps. A surrounding
process deadline must bound a native planner or action call that stalls.

Public camera observations retain RoboTwin names and conventions: `depth` is
in millimetres, `intrinsic_cv` and `extrinsic_cv` use the upstream CV convention,
and `cam2world_gl` is the upstream GL transform. Object poses, segmentation and
evaluator outcomes are excluded. Unlike upstream `joint_action`, which contains
drive targets, `measured_joint_positions` reads articulation qpos. Gripper
fractions are explicitly labelled commanded values, not measured aperture.

The native check passed seven official joint/gripper actions on
`blocks_ranking_rgb`: three host calls followed by four actions requested
through five isolated RPCs. Both arms were commanded 0.02 radians from their
initial joint-0 positions; measured joint error was about 0.00361 radians.
Invalid arm names, non-finite and out-of-limit joints, and an expired callback
deadline were rejected before action. A separate native Cartesian check requested a 5 mm left end-effector
translation: actual displacement was 4.72 mm, position error 0.284 mm and
orientation error 0.00143 radians. Its exhausted one-action budget and an
isolated unknown-RPC request were rejected without advancing the native action
counter. These local checks do not establish whole-workspace reachability or
grasp success.

The generated-policy side uses a separately built Python 3.12 stdlib runtime;
the simulator remains in Python 3.10. The initial Python 3.10 isolated policy
failed when `time.sleep` was denied by the existing sandbox. No sandbox rules
were relaxed. The passing policy is a handwritten infrastructure diagnostic,
not a GPT policy, task solution, or Self-Harness round.

Three fresh native resets also passed a private placement comparison for this
RoboTwin task: seeds 3000, 3000 and 3001. Repeating seed 3000 yielded exactly
the same three block root poses. Seed 3001 changed all three placements;
translations ranged from 0.0556 to 0.1928 metres. The gate uses 1 mm translation
or 0.5 degree orientation thresholds. These private poses are infrastructure
evidence only. This checks pairwise seed-generated placement novelty, not
arbitrary-state replay, novelty against the full history, or all RoboTwin tasks.

## Editable candidate programs

[candidate_program.py](candidate_program.py) binds three JSON files to the
existing isolated CAP executor and supplies a `JsonEditContract` for the
existing Self-Harness `StructuredProposer`:

| Artifact | Self-Harness component | Runtime use |
| --- | --- | --- |
| `egl/primitive_skills.json` | `skills` | Python functions exported through the `skills` dictionary inside the isolated program. |
| `egl/skill_combinations.json` | `skill_selection` | Defines `policy(robot, memory)` and composes those functions. |
| `egl/memory.json` | `memory_rules` | JSON memory snapshot copied into each episode. |

The primitive and combination files contain `schema` and `source`; their schema
identifiers are `physicalrsi.egl-primitives/v1` and
`physicalrsi.egl-combinations/v1`. Memory contains `schema` set to
`physicalrsi.egl-memory/v1` and a `memory` object. Example combination source:

```python
def policy(robot, memory):
    return skills["reach"](robot, memory)
```

Binding checks the complete declared harness hashes, exclusive component
ownership, JSON schemas and Python syntax. It never executes or imports the
candidate on the host. The trusted caller passes `bound.source` and
`bound.memory` to `PhysicalRSI_baselines.robodojo.code_execution.execute_policy`
with the admitted native callbacks and isolation runtime. Record
`bound.identity()` before execution. Host-native controls, scoring, model
credentials and sandbox limits are outside these three editable files.

Changing any of these artifacts produces a different harness revision.
Runtime memory reads return independent copies; transient updates cannot change
an old snapshot or become inherited memory without a new proposal. The binder
checks declared bytes, not completeness of simulator dependencies or eligibility
for a benchmark. Admission and independent evaluation remain required before
Self-Harness selection; successful materialization is not selection.

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  tests/test_egl_candidate_program.py tests/test_structured_proposals.py
```

These checks cover data-only binding, memory copy isolation, changed-file and
ownership rejection, the three edit domains, stale-parent rejection and the
existing structured proposal machinery. They are software checks, not a
benchmark campaign or GPT evaluation.

A native diagnostic now covers all three edit types on RoboTwin's
`blocks_ranking_rgb`. After a parent program ran, the existing StructuredProposer
materialized three reviewed, enumerated single-file edits from its development
evidence. Each candidate then ran in a fresh native task through isolation:

| Version | Change | Commanded joint displacement | Native actions |
| --- | --- | --- | --- |
| Parent | Initial program | 0.01 rad | 1 |
| Primitive child | Double the primitive's displacement factor | 0.02 rad | 1 |
| Memory child | Change the memory displacement parameter | 0.02 rad | 1 |
| Combination child | Compose movement with return-home | 0.01 rad, then home | 2 |

The parent moved about 0.00736 rad and the primitive/memory children about
0.01639 rad. The combination returned within 0.00126 rad of home. All four
programs had distinct verified harness revisions; each child changed only its
declared artifact, and the parent remained unchanged. Isolated process receipts
and native command traces were retained as content-addressed evidence.

This is proposal materialization plus native execution, not a completed
Self-Harness selection round. Its diagnostic closure explicitly lacks full
dependency admission, the candidates were enumerated rather than GPT-generated,
and no benchmark improvement or survivor was reported. Full campaign evaluator
integration, task coverage, provider configuration and independent audit remain
outstanding.

## Recorded CAP experiments

[recorded_cap.py](recorded_cap.py) connects bound candidate programs to the core
`ExperimentRuntime`. `RecordedCapPolicy` runs the isolated executor in an owned
worker. Its callbacks enqueue requests; they never call the simulator. Each
`policy.act` returns the next request as data. Core writes `pending-action.json`
before `RecordedPrimitiveEnvironment.step` invokes the admitted native callback
on the experiment thread. Public callback replies go back to the candidate;
the separate `evaluation` field stays in the trusted trajectory and verifier.

A distinct finish request records program completion without interpreting the
program's return value as success. `OfficialProxyVerifier` reads the native
Boolean predicate recorded by the environment. It explicitly marks the
independent intended-task audit as absent. Candidate exceptions, missing native
results and cleanup failures remain interrupted/reconciliation cases; they are
not silently scored as task failures. The environment owns native cleanup and
supports lease quiescence. Its factory must clean up any partially initialized
simulator if reset construction fails. A surrounding process supervisor is
still needed to stop a blocking native call.

Steps in this adapter count CAP requests, including observation and finish
requests, rather than physics steps. Include the native primitive trace and
native action count in the trusted measurement callback to preserve the
underlying commands. Keep the factory, calibration, simulator dependencies,
capability set, budgets and official measurement implementation frozen in the
experiment identity. The candidate receives no raw task handle or scoring
callback.

One actual worker-13 RoboTwin run now passes through this interface with a
System 2 development attribution, a candidate-bound System 1 identity and a
named GPU lease. The recorded requests were `get_proprioception`,
`move_to_joints`, `goto_home_joint_position`, and finish. Two native actions
executed, the native official predicate remained false, and the recorded
outcome was **failure**. Core re-read and verified the full receipt, including
trajectory, policy shutdown and lease quiescence. This diagnostic is not a
benchmark score, admission of its fixture dependency closure, or completion of
a Self-Harness selection round. The same runtime binding subsequently passed on original LIBERO and
LIBERO-Plus workers: each ran `get_observation`, joint movement, return-home
and finish, recording eight actual physics steps and an official failure.
Policy shutdown, named GPU lease quiescence and isolated process evidence were
verified for both. These checks cover one task per family. Full campaign
evaluator integration still requires verification.

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  tests/test_egl_recorded_cap.py tests/test_egl_candidate_program.py
```

These nine software checks include recording before effects, simulator-thread
ownership, withholding private measurement fields, ignoring self-reported
success, rejection before actuation, and stopping a waiting policy after native
failure or official termination. Protocol tests use a controlled executor stub;
they do not substitute for the separate real-isolation/native check above.

## Native paired selection diagnostic

A separate paired diagnostic now connects actual LIBERO and LIBERO-Plus
ExperimentRuntime receipts to `experiment_result` and `select_survivor`.
The pool and binary official-proxy scoring profile were frozen before registering
fresh seed-3200 cases, one task per family. The parent used a 0.02-radian memory
parameter; a reviewed StructuredProposer edit produced a separate 0.04-radian
child. Both used identical native execution budgets and environments.

Before selection, the diagnostic checked that each parent/child pair had
identical reset object geometry, compiled model identity and full initial public
camera/proprioception observation. Each validation layout's geometry also
differed from the seed-3100 development reset. Core then validated candidate,
comparison, cohort, execution-protocol and evidence attribution for all four
native receipts. Both programs failed both official predicates, so the selector
returned `retain_parent`. Additional movement was not counted as improvement.

This establishes a native paired-evidence/selection path on two declared tasks.
It does not establish an admitted full Self-Harness campaign: fixture dependency
closure is still unattested, the candidates are reviewed diagnostics rather than
GPT outputs, and no lineage promotion occurred. RoboTwin was not part of this
paired comparison. The four failing diagnostic trials are not a benchmark-wide
success-rate estimate or evidence of an exploit.

## Public surface location for placement

`LocatedLiberoPrimitives` adds a read-only `locate_object` callback to the
existing native action boundary. It passes public camera observations to
`PublicObjectLocator`, which combines SAM3 masks with calibrated depth. The
result contains visible surface quantiles in robot-base coordinates, mask
confidence and image-border contact. It does not expose simulator object poses,
assert semantic identity, or claim to know the complete object extent.

The current Codex-authored placement hypothesis uses the visible basket opening
to choose a release location, then composes grasp, lift, transport, release and
retreat for both cans. A nearly closed gripper after lifting causes an early
return. Neither gripper width nor completion of that sequence proves task
success: the original native predicate remains the reported proxy outcome.
Development trials and candidates retain `qualification: null`; they do not
constitute an admitted campaign or a benchmark-wide score.

Native tracking now checks the requested hand position and orientation during
movement, permitting early completion after two consecutive measurements within
1 mm and 0.005 rad. It retains the 120-step primitive limit and the native
1,000-step episode horizon. This addresses wasted motion steps observed when
the hand reached its target but the stricter joint criterion did not converge.
The IK adapter separately projects finite measured gripper fractions onto the
model's `[0, 1]` seed domain. Native contact trials exceeded both initial 0.04 mm
and subsequent 0.4 mm excursion thresholds, showing that an arbitrary cutoff on
this optimizer initial value was inappropriate. Non-finite inputs still fail.
Raw public measurements, the projected seed and the clipping flag remain
visible; Cartesian convergence is measured independently after motion. This
projection does not qualify or validate physical gripper measurements.
