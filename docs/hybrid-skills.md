# Hybrid code and learned skills

`PhysicalRSI_core.skill_execution.SkillExecutor` coordinates a fixed, versioned
sequence of subgoals. Each stage has an inference provider, an independent
observation monitor, a release operation, and action/observation/time budgets.
The full original instruction travels alongside every stage goal. The frozen
plan includes the executor source digest and caller-declared action-validator
revision; its returned identity is a copy.
A provider may be reviewed code or a learned action model. System 1 executes
the frozen plan; System 2 may propose and compare different plans through
Self-Harness. This addition does not itself select a survivor or load weights.

## Execution boundary

All three stage operations are explicit, trusted `Operation` instances:

- The provider receives the complete subgoal and a controller observation. It
  proposes a bounded `physicalrsi.action-chunk/v1`, bound to that observation.
- The monitor receives the same public input, independently of model predictions.
  It returns `running`, `succeeded`, `failed`, or `unknown`, with evidence.
- Release resets inference state and queued model actions. It must positively
  acknowledge completion before another provider can take ownership.
  It also runs before first acquisition, clearing any previous episode state.

Only the caller's admitted controller actuates. The executor rejects mismatched
action contracts, stale observations, episode/clock changes, malformed chunks,
concurrent calls, and missing controller acknowledgements. The caller must
acknowledge an entire completed chunk before supplying the next observation.
No mid-chunk handoff is supported: choose an appropriately short chunk horizon.
The native controller remains responsible for actual execution, resource
ownership, observation provenance, and timely cancellation.

`unknown` stops inference ownership and requests another observation without an
action. `failed` stops the plan. `succeeded` releases the current provider and
advances one stage; a fresh observation is required before the next stage can
act. Subgoal completion is not the official benchmark outcome. A model's
predicted value or self-reported success cannot complete a stage.

Callbacks must cooperate with deadlines and have an external process/service
timeout. This coordinator is not an OS sandbox for a model or monitor. Use the
existing isolated code-policy executor for generated code. Cleanup uses a
separate bounded deadline so cancellation can still release inference state.
Uncertain cleanup or controller execution requires reconciliation, not retry.

The fresh workspace records the plan and ordered events, including checks,
returned actions, acknowledgements and handoffs. The executor refuses an existing
workspace; it does not claim crash recovery or exactly-once native execution.
Inspect the journal and native controller before starting a replacement episode.

## LIBERO integration

`PhysicalRSI_baselines.embodied_goodharts_law.hybrid_libero` supplies explicit
native action contracts and a wrapper for an externally loaded learned backend.
The wrapper requires checkpoint, implementation, preprocessing and normalization
digests. Those are caller attestations; the backend must verify actual bytes.
Inference returns normalized native action lists, and reset must return `True`.
The full goal remains in the request. Perception and model adapters must expose
only admitted public observations; the generic executor cannot detect encoded GT.

The current EGL `NativeLiberoPrimitives` requires **JOINT_POSITION, 8 controls**.
The official [OpenVLA-OFT evaluator](https://github.com/moojink/openvla-oft/blob/main/experiments/robot/libero/run_libero_eval.py)
uses the default LIBERO **OSC_POSE, 7 controls** interface. These contracts are
intentionally different. Padding, truncating, or relabeling actions is invalid.
Either adapt and validate code skills under OSC control or explicitly qualify a
controller conversion. Never switch native controllers implicitly mid-episode.

Before inference, preserve the checkpoint's camera orientation/crop, proprio
layout, action normalization, gripper conversion and control period. A checkpoint
trained on whole tasks is not automatically qualified for arbitrary subgoals or
handoff states. Start with coherent manipulation stages and test entry states.
An image-only world model is not an action provider; a world-action model needs
an admitted action interface. Predictions remain separate from observed outcomes.

## First experiment

Use cabinet-top and upper-drawer Spatial tasks as development targets. Compare:

1. A frozen code-policy reference.
2. A frozen released learned policy under its documented evaluation protocol.
3. A hybrid plan using the same frozen learned weights, with public grounding,
   bounded manipulation stages, and independent post-release observation.

Freeze initial cases, episode budgets, cameras, source/model identities and
controller conventions. Controller differences between arms must be disclosed;
an isolated composition-effect comparison requires the same controller. Track
official success, public subgoal evidence, handoff failures and wall time. Reserve
separate layouts for final reporting. Model training data are part of the hybrid
baseline; its results cannot be attributed to a zero-shot pure-code policy.

The current implementation provides orchestration and admission checks, not a
native hybrid LIBERO result. No released VLA is loaded by these tests.

An additional native interface smoke check on 2026-10-08 initialized the original
LIBERO Spatial drawer and cabinet-top scenes under OSC_POSE. Both produced
nonconstant 256-by-256 RGB views from the external and wrist cameras, finite
robot proprioception, and accepted three neutral 7-dimensional controls at 20 Hz.
This checks initialization and the observation/action interface, not manipulation
success or a learned policy. The existing environment initially resolved the
ASPIRE LIBERO-Pro dependency; the successful probe explicitly selected the
original LIBERO source and used its own temporary configuration. Native adapters
must preserve this source boundary instead of relying on ambient imports.

## Check the implementation

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  tests/test_skill_execution.py tests/test_hybrid_libero.py \
  --basetemp=/tmp/physicalrsi-hybrid-check
```

The isolation case executes an actual generated code provider in chroot/seccomp
and hands control to a host inference fixture. It proves software handoff only.
Other cases cover missing acknowledgements, unknown evidence, stale observations,
uncertain release, cancellation, budgets, and incompatible action spaces.
