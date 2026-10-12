# Physical L1/L2/L3 method

This package extracts the core method from the Tianji dual-arm host into
physicalRSI. It includes execution, rescue learning, immutable code/memory
candidates, and independent Self-Harness comparisons. It does not include a robot
launcher, SDK, ROS nodes, local configuration, credentials, recordings or a
physically qualified policy. Importing the package starts no hardware or model.

## Method

System 1 executes a hierarchy:

1. **L1 — reusable code policy.** `L1Controller` executes `policy(robot, memory)`
   through the original code driver, a read-only Python runtime, bubblewrap,
   seccomp and a host-owned RPC bridge. `look`, `align`, `relate` and semantic
   `gripper` requests are checked by the host. There is no unrestricted execution
   fallback. A capability limitation hands control to L2.
2. **L2 — direct-tool policy.** `L2Controller` uses fresh three-view RGB and
   measured state, keeps explicit conversation history, and requests bounded
   EEF actions. Image inspection, calculation and notes support exploration.
   Public L1 bottlenecks, verified policy lessons and historical L3 lessons are
   delivered separately from current geometry. Unknown actuation stops the
   episode; it is never an invitation to retry a command.
3. **L3 — assistance.** `guidance.submit` records language interventions.
   `VideoReferenceL3` lets the analyst choose historical clips, time windows,
   sampling density and crops, then returns a qualified lesson to development.
   `TerminalTeleoperation` is the optional measured-HDF5 import/handoff adapter:
   it releases robot ownership and stays in human control until judged complete.
   It does not automatically reacquire a robot or resume L1. A deployment that
   resumes after teleoperation must implement an explicit ownership handoff.

System 2 uses `DevelopingL1` to propose new source and memory after L2 or video
assistance, and `learn_from_episodes` to recover learning from completed
**development** episodes. Every proposal receives declared runtime capabilities.
The learning projection retains RGB, visible purposes, lessons and result labels;
it removes raw joint/EEF actions and private assessment text. Online candidates
can return to L1 exploration but are never automatically promoted.

`DevelopmentProposer` and `PairedEpisodeEvaluator` connect to the original
`SelfHarness`: freeze candidates, restore held-out scenes, run independent
L1-only parent/child trials, select and record lineage. Validation evidence is
excluded from development. Uncertain outcomes cannot become scored failures;
assisted successes cannot become independent L1 successes. Repeated rounds bind
new development input without changing the stable proposer identity.

## Core files

| Responsibility | Files |
| --- | --- |
| Hierarchy and execution | `coordinator.py`, `controllers.py`, `contracts.py`, `capabilities.py` |
| L1 admission and isolation | `policy_runtime.py`, `_jail.py`, `library.py` |
| Independent assessment | `monitor.py`, `dialogue.py` |
| Execution-side vision | `visual_perception.py`, `visual_servo.py`, `analysis_tools.py` |
| L3 assistance | `guidance.py`, `video_learning.py`, `video_reference.py`, `demonstrations.py` |
| Learning and selection | `experiment.py`, `learning.py`, `learning_projection.py`, `evolution.py` |
| Evidence and accounting | `evidence.py`, `usage.py`, `reporting.py` |
| Host/model boundaries | `physical_port.py`, `host_environment.py`, `models.py` |

## Host integration

Implement `contracts.RobotPort` and supply independent model roles implementing
`contracts.Model`. `models.make_model` supports the existing HTTP transport and
an offline mailbox. `require_independent_models` admits the bundled HTTP
transport; a custom transport needs its own host isolation check. There is no
implicit local-login transport or model-side terminal/filesystem capability.

The observation/action contract intentionally retains the original three cameras
(`front`, `left`, `right`), 26 measured values `[left arm 7, left hand 6, right arm 7,
right hand 6]`, `wxyz` orientations, semantic gripper requests, and 40 ms policy
steps (two 20 ms host servo updates). This is a method extraction, not an
embodiment-independent replacement for these contracts. Changing them requires
refreezing candidates and revalidation. Legacy `tianji.*` schema names remain
for evidence compatibility; old qualification does not transfer to this package.

The deployment owns SDKs, IK/FK, timestamp pairing, collision/contact handling,
shared device leases, successful-send tracking, stop confirmation, manual rearm,
and any scene restoration. `physical_port.PhysicalRobot` retains the optional
Unix-socket client protocol; its worker is external. No collision planner or
palm/TCP calibration is supplied here. Image alignment and motor completion do
not prove grasp retention or task success.

A host composes these objects (this fragment is not a robot launcher):

```python
from PhysicalRSI_demos.real_robot.contracts import Level
from PhysicalRSI_demos.real_robot.controllers import L2Controller
from PhysicalRSI_demos.real_robot.coordinator import Coordinator
from PhysicalRSI_demos.real_robot.experiment import DevelopingL1
from PhysicalRSI_demos.real_robot.library import PolicyLibrary
from PhysicalRSI_demos.real_robot.video_learning import VideoReferenceL3

l1 = DevelopingL1(library=library, model=development_model, runtime=runtime,
                  perception=perception, parent=parent)
def policy_context():
    if l1.parent is None:
        return None
    program = PolicyLibrary.load(l1.parent)
    return dict(policy_id=l1.parent["id"], memory=program["memory"])

controllers = {
    Level.L1: l1,
    Level.L2: L2Controller(execution_model, control_geometry=host_geometry,
                          policy_context=policy_context),
    Level.L3: VideoReferenceL3(reference_directory, video_model),
}
result = Coordinator(task=task, robot=robot, monitor=monitor, judge=judge,
                     episode=episode, controllers=controllers,
                     rescue_learning=l1.improve,
                     before_success_stop=host_confirm_supported_object).run()
```

Supply the independent monitor and judge through their own contexts. The success
handoff callback must confirm that any held object is supported before normal
shutdown. Errors and stop requests still follow the host stop path. For
validation use a frozen `L1Controller`, `allowed_levels=(Level.L1,)`, no rescue
learning callback, and no L2/L3 execution.

Reusable skill sources are passed explicitly to
`PolicyLibrary(..., reusable_skills=[...])`; each entry includes `name`, `source`
and `qualification="unvalidated-development-candidate"`, plus optional entry and
parameter descriptions. Sources and hashes are included in the immutable
candidate closure. No bottle-specific or side-grasp file is silently read from
another checkout. Source provision, call references, actual execution and
physical validation remain separate facts.

For physical comparisons supply both `environment` and `environment_capture`
to `PairedEpisodeEvaluator`. The capture callback inventories the actual host's
configuration, loaded SDK implementation, model assets and package versions.
`host_environment.capture_environment(files=..., package_versions=...)` provides
explicit file fingerprinting; the host is responsible for inventory completeness.
Save matching receipts at `evaluation/environment-before.json`,
`hardware-owner/environment.json`, and `evaluation/environment-after.json`.
`selected_policy(..., qualification="physical", environment_capture=...)`
checks the committed receipts again. These hashes do not attest firmware,
mechanical calibration or physical scene restoration.

## Installation and runtime

From the repository root, on Python 3.11 or newer:

```bash
python -m pip install -e '.[real-robot]'
```

L1 isolation requires Linux, bubblewrap, libseccomp and user namespaces. Prepare
a new runtime outside the repository:

```bash
python -c 'from PhysicalRSI_demos.real_robot.policy_runtime import prepare_runtime; prepare_runtime("/tmp/physicalrsi-real-robot-runtime")'
```

Historical video extraction also needs `ffmpeg` and `ffprobe`.

The contribution contains the method and integration documentation. Its regression
tests remain in the development workspace and are not included here. Local
validation exercised isolation, synthetic RGB control, rescue learning and
Self-Harness selection with fake robots and deterministic models. It does not
establish GPT learning ability or physical task success.

`provenance.json` records SHA-256 values of the source modules at extraction.
