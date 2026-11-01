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

## Known violations (2026-11-01)

| Skill | Violations | Status |
|---|---|---|
| E3 RoboTwin `CameraBlockSkill` | 2 (monolithic), 4 (fixed stack target, table height, fabricated fallback), 5 (open loop, oracle ends episode), 8 (frames as JSON) | Being rebuilt as composed primitives (E3.4); blobs in core (C7) |
| RSI-Bench-0 parametric and neural policies | 1 and 2 (plain parameter files, no descriptors or composition) | Acceptable for a statistics benchmark; not a skill reference |
