# Skill design principles

These rules define what a PhysicalRSI skill is and how one is reviewed. They
apply to every primitive and composed skill in the mainline, in benchmark
adapters, and in experiment code. A skill that breaks a rule is either fixed
or documented as an exception with its reason before it enters an experiment.

## 1. A skill is a typed transformation

A skill maps one typed value to another: an observation becomes an estimate,
an estimate becomes a plan, a plan becomes a command. Each skill declares its
input and output as `Contract`s with name, unit, frame, embodiment and, for
arrays, shape and dtype. If two skills are chained, `compose_sequence` must
accept the chain without `strict=False`. A conversion between contracts (units,
frames, image formats) is an explicit `adapt(...)` skill, never hidden inside
another skill.

## 2. Small primitives, explicit composition

A primitive does one thing that can be tested alone: perceive objects, fit a
support plane, choose an arm, plan a grasp, verify a grasp. A behaviour such as
"stack three blocks" is a composition of primitives, recorded as a
`SkillComposition`, not one class that does everything. The test of the split:
System 2 must be able to replace, insert, remove or re-parameterize one
primitive without editing the others, and a result must be attributable to the
primitive that changed (`SkillMutation` names it).

## 3. Public inputs only

A policy skill reads only what the target embodiment could sense: camera
images and calibration, depth, proprioception, force and contact, and the task
description. It never reads simulator object poses, hidden task state, seeds,
layout identifiers or success oracles. Privileged state is allowed only in the
verifier and in tests, and every skill suite has a test that makes privileged
accessors raise for a whole episode.

## 4. No task-geometry shortcuts

Positions, heights and targets come from observation or from the task
description, not from constants that happen to match a simulator layout. A
missing observation is reported as missing; the skill re-observes, retries or
fails cleanly. It never substitutes a fabricated value. Embodiment facts (arm
bases, joint limits, camera mounting) come from the embodiment description,
not from literals in skill code.

## 5. Closed loop and self-verification

Every action that can fail is followed by a check from public observations:
after a grasp, verify the object moved with the gripper; after a place, verify
it rests on its support. The skill decides when it is done from its own
verification. The success oracle is the verifier's judgement after the episode
and never terminates or steers the episode.

## 6. Parameters are declared, bounded and owned

Every tunable value lives in the harness, scoped to the primitive that uses it
(`control.json` section per primitive), with bounds declared as `SearchField`s.
No tunable constant is buried in code. A starting configuration and its
rationale are recorded with the experiment.

## 7. Execution goes through declared tools

Skills emit declarative commands (move to pose, grasp, place) that a declared
tool executes: a motion planner, a controller, a device driver. The tool is a
versioned dependency of the skill's System 1 declaration. Skills do not call
simulator internals directly.

## 8. Sensor data travels as blobs

Images, depth, point clouds and other arrays are stored in the trial's blob
store and referenced from observations by digest. Observations stay small JSON;
evidence stays byte-verifiable.

## 9. Evidence, not claims

Each skill exposes what it used and why: detected objects with confidence,
chosen arm, planned poses, verification outcomes. These go into trial evidence
so a failure can be traced to a primitive. A skill's documentation states its
scope: simulator, observation mode, embodiment, and what has not been tested.

## Skill architecture: primitives evolve, task skills are written

### The theory

Skills are structured. Manipulation reduces to a small vocabulary of motor
primitives (approach, grasp, lift, move, place, release, retreat, push; later
insert, turn and pull) together with the perception and verification
primitives they need. The vocabulary can be exhausted. The parameters cannot:
they are continuous and depend on the scene and the robot.

A specific task is solved by a task skill: code written at test time that
composes primitives with task logic. Evolution should make the agent better at
two things: using primitives, which means choosing their parameters and methods
from observation, and combining them. Durable improvement accumulates in the
primitives. Task skills are cheap to write and disposable.

### Layer 1: the primitive library (evolved and shared)

One implementation of each primitive lives in `physicalrsi.skills` and serves
every task, benchmark and embodiment. A primitive never contains benchmark,
robot or task names, or layout values.

Every motor primitive has the same four parts:

1. **Preconditions**: when it applies, checked from public observation. For
   example, grasp needs the target localized and the gripper open at the
   approach pose.
2. **Parameter choice**: from observation and the embodiment description. For
   example, the grasp pose comes from the lifted object estimate, the support
   plane and the tool offset. Tunable values are bounded `SearchField`s.
3. **Command**: canonical declarative commands to the execution tool.
4. **Verification**: a success check from public observation that returns the
   outcome with its evidence. A nudge is not a grasp.

The first motor primitives are approach, grasp, lift, move, place, release,
retreat and push. Parameter choice is implemented by planning primitives
(`select_arm`, `plan_grasp`, `plan_place`, `plan_push`) so each can be tested and
mutated alone. Perception primitives are `detect_objects` (with a replaceable
method: colour prototype, appearance memory or open-vocabulary detector),
`lift_detections` and `estimate_support_plane`. Existing core skills
(`depth_point`, `tool_frame`, `motion_arrival`, `effect_guard`,
`vertical_retreat`, `verify_region`) are reused rather than duplicated.

**Composites** are named compositions that recur across tasks: pick is
approach, grasp and lift; pick-and-place is pick, move, place, release and
retreat. A composition is data, a graph of primitives whose transitions depend
on verification outcomes (success continues, failure takes a recovery branch or
fails cleanly, retries are bounded). One shared interpreter runs any
composition, so there are no hand-written task controllers. A composite that
recurs in several task skills may be promoted into the library.

### The embodiment boundary

A benchmark adapter supplies only three things.

- **Embodiment description** (data): for each arm its name, base pose, tool
  offset relative to the commanded frame, approach axis in the tool frame,
  gripper open and closed widths and how width is sensed; for each camera its
  name and calibration source. RoboTwin's gripper bias and LIBERO's Panda
  gripper are two instances of one description.
- **Observation adapter**: converts simulator or robot output into the
  canonical observation: upright RGB, metric depth in metres, calibration
  (intrinsics, camera to world), proprioception (tool pose and gripper width per
  arm) and the task text. Conversions such as z-buffer to metres happen here,
  once.
- **Execution tool**: executes `move_tool(arm, pose)`, `set_gripper(arm,
  width)` and `wait(steps)` and reports status and the reached pose. RoboTwin's
  motion planner and LIBERO's operational-space servo loop implement this one
  interface.

### Layer 2: task skills (written at test time)

A task skill is written for one task by the agent itself, through the system's
language-model proposer. It composes primitives and composites with task
logic: which objects, in what order, onto which support, when the task is done,
and task-level recovery.

- It acts only through a restricted interface: invoke a primitive or composite
  (parameter overrides only within declared bounds, and recorded), read
  primitive outcomes and evidence, read public observation summaries
  (detections, support plane, proprioception) and the task text. It has no
  simulator access, no privileged state and no direct execution-tool commands.
- It contains no geometry corrections. A literal offset or target that
  compensates for a primitive's error hides a weak primitive and stops it from
  improving. Integer literals for counts and indices are task logic; any other
  number must be a bounded parameter declared in the task skill's control
  section, and a static check rejects the rest.
- It is versioned per task and records the primitive revisions it used.

### Where a fix belongs

The first primitive whose verification failed in the trial evidence locates
the failure. The fix goes to one layer:

| Finding | Layer |
|---|---|
| A primitive fails its verification although its preconditions held | the primitive: parameters, method or verifier |
| A primitive's preconditions were not met because of task order or choice | the task skill |
| A robot fact is wrong (tool offset, approach axis, gripper widths) | the embodiment description |
| An observation has the wrong unit, orientation or format | the observation adapter |
| The same workaround appears in several task skills | a primitive or a composite |

Examples from the RoboTwin stacking work (E3.5): the grasp check that rejects
nudges belongs to grasp verification; averaging block estimates over several
frames belongs to perception; the gripper offset and top-down approach belong to
the embodiment description and grasp parameter choice; keeping the last placed
block as the next support is stacking logic and belongs to the task skill.

### Two evolution loops

- **Inner loop, per task, at test time**: write or repair the task skill
  against a frozen primitive library, validated on the task's development and
  validation cases with the paired gate.
- **Outer loop, across tasks**: propose changes to primitives and to
  compositions. Every `SkillMutation` carries a kind: `use` (a primitive's
  parameters, method or verifier) or `combine` (insert, delete or reorder a
  step, add a recovery branch, extract a composite). A primitive change is
  accepted only through a multi-task paired gate: every registered task that
  uses the primitive is re-run with its task skill unchanged, on every
  embodiment, so the comparison isolates the primitive. It is accepted when the
  pooled improvement is significant under the paired permutation test, with
  alpha split across comparisons, and no task shows a significant regression.

### How the theory is tested

- **Primitive competence over rounds**: the verified success rate of each
  primitive under its preconditions, across tasks and both embodiments,
  confirmed afterwards by the benchmark oracle.
- **Held-out first-attempt success**: freeze the library at rounds 0, k and 2k;
  for held-out tasks, write fresh task skills with the same coder and budget.
  First-attempt success should rise with the round. This is the main test.
- **Task-skill size**: lines and recovery branches per task should fall as
  primitives become more robust.
- **Vocabulary closure**: the share of new tasks solved without adding a
  primitive should approach one. A new primitive is added only by design review,
  and the addition is recorded.

The test of the shared library itself: the same primitive revision runs in
RoboTwin and in LIBERO without edits, and a `SkillMutation` to one primitive is
evaluated in both.

## Review checklist

Use this list when reviewing a new or changed skill.

- [ ] Every skill has a `SkillDescriptor` with typed input and output contracts.
- [ ] Compositions pass `compose_sequence` in strict mode; conversions are `adapt` skills.
- [ ] Each primitive does one job and has its own unit test.
- [ ] The privacy test makes privileged accessors raise for a full episode and passes.
- [ ] No literal positions, heights or targets tied to a simulator layout; no fabricated fallbacks.
- [ ] Actions that can fail are verified from public observations; the oracle never ends an episode.
- [ ] All tunable values are bounded `SearchField`s in a per-primitive control section.
- [ ] Execution uses declared tools listed as System 1 dependencies.
- [ ] Arrays travel as blob references.
- [ ] Mutations proposed by System 2 carry a `SkillMutation` naming the changed primitive.
- [ ] Documentation states scope and untested conditions.
- [ ] Each motor primitive has preconditions, parameter choice, command and verification.
- [ ] Primitives contain no benchmark, robot or task names; robot facts come from the embodiment description.
- [ ] Task skills act only through the restricted interface: no simulator access, no direct tool commands.
- [ ] Task skills contain no geometry corrections; non-integer numbers are declared bounded parameters.
- [ ] Each fix sits in the layer the attribution table gives; mutations are labelled `use` or `combine`.
- [ ] A primitive change passes the multi-task gate with no per-task regression.

## Known violations (updated 2026-11-02)

| Skill | Violations | Status |
|---|---|---|
| E3 RoboTwin `CameraBlockSkill` | 2 (monolithic), 4 (fixed stack target, table height, fabricated fallback), 5 (open loop, oracle ends episode), 8 (frames as JSON) | Being rebuilt as composed primitives (E3.4); blobs in core (C7) |
| E3 RoboTwin and E3-L LIBERO skill suites | 2 at library level: each suite carries its own copy of the same primitives (table plane, grasp and place planning and verification); LIBERO also copies the skill protocol and composition | Unification into the shared library (this document, 2026-11-02) |
| E3 RoboTwin `ClosedLoopController` and E3-L LIBERO controller | Layers mixed: hand-written task controllers hold task logic and primitive fixes together (frame averaging, retained support target, retries); composition is code, not data, so it cannot evolve | Split into primitives, composites and a task skill during migration (unify phase 2) |
| RSI-Bench-0 parametric and neural policies | 1 and 2 (plain parameter files, no descriptors or composition) | Acceptable for a statistics benchmark; not a skill reference |
