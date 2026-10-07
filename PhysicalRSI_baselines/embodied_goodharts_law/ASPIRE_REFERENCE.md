# Public-sensor ASPIRE reference work

The current development policies are not a verified reproduction of ASPIRE's
benchmark results. In Spatial cohort 034, the cabinet-bowl candidate succeeded
in 1/24 trials versus 0/24 for its parents. These are paired development trials,
not the full LIBERO benchmark or an independent holdout.

## Source comparison

The checked-out ASPIRE example at
`aspire/sim/cap/envs/tasks/franka/franka_libero_env.py` uses language-conditioned
point clouds, an oriented bounding box and a top-down grasp pose. Its grasp-network
alternative is commented out. Thus, absence of a grasp-network call alone does
not establish a departure from ASPIRE's method.

The example also calls `plan_grasp_trajectory` with world collision enabled,
executes a joint trajectory, requests visual grasp/lift confirmation, and calls
`plan_with_grasped_object` for transport. These are material gaps in the current
LIBERO development path, whose pose primitive explicitly performs IK without
scene collision planning. The example is implementation evidence, not proof of
benchmark performance.

`aspire/sim/cap/integrations/motion/curobo_api.py` provides the reusable public
depth-to-world and grasp-trajectory functions. Its grasp planner defaults to
fingertip poses and applies a hand offset. Existing native primitives use hand
poses. This convention must be declared and checked against native robot forward
kinematics before execution. Do not import privileged object-pose helpers to
resolve this discrepancy.

## Implemented boundary

`public_scene_grasp.py` accepts only metric depth, camera calibration, a boolean
perception mask, measured arm joints and estimated grasp poses. It calls the
ASPIRE depth-world builder and collision-aware grasp planner. It requires both
target and scene meshes, preserves scene collision checks, excludes only the
target during approach, and does not relocate the ungrasped target to the hand.
Hand/fingertip convention is explicit. Returned trajectories are numerical plans,
not executed or verified grasps. A reconstructed scene is only partially observed.

The adapter is not yet wired into native Self-Harness trials. Unit tests use an
injected backend and prove contract behavior only. Separate GPU diagnostics have
verified loading, scene reconstruction and one native approach execution on a
development reset. Robot-model parity was checked on 95 recorded public states
after calibration on one other state. This does not qualify a grasp or a task.
Host deadlines require an externally bounded worker because a Python deadline
check cannot interrupt CUDA.

In the native approach diagnostic, a 15-degree orientation change enabled a
strictly planned trajectory that the original fixed orientation did not produce.
Using ASPIRE's .01-radian intermediate joint tolerance, 50 waypoints executed in
a run totaling 576 physics steps. The measured endpoint error was .000114 metres
and .00428 radians, below the independent .001-metre/.005-radian endpoint limits.
The earlier .002-radian waypoint tolerance stopped near the beginning of the
path. These are single-scene control diagnostics, not benchmark improvements;
grasp confirmation, held-object transport and independent layout validation
remain outstanding. Do not promote the particular orientation as a universal
grasp rule from this observation.

Near-field wrist observations require an explicit grasp-inference depth range.
The legacy client default of .2--2 metres excluded 13,282 of 26,725 masked bowl
pixels in one diagnostic image. `ContactGraspClient(depth_range_m=(.015, 2.))`
admits those measured near points; the legacy default remains available for
existing callers. This is a sensor-processing parameter, not an object-pose hint.

On that retained observation, the near-depth request returned 32 candidates.
The top five by network score and their half-turn variants did not plan. A
lower-scoring candidate passed strict scene planning when near-side candidates
were considered. XY distance to the base was only the search-order heuristic;
the planner still checked feasibility. Neither network confidence, geometric
ranking nor an offline plan proves grasp success.

The reviewed ASPIRE raw-CGN conversion adds .12 metres along local Z before
CuRobo subtracts .1168 metres to obtain its Panda hand pose. The combined .0032
metre offset is implemented explicitly in `aspire_grasp_frame.py`. A local-Z
half-turn preserves ideal parallel-finger contacts, but its robot/palm collision
and reachability must be checked again. Native hand calibration is separate.

## Post-close depth and association diagnosis

The retained native-010 post-close wrist capture exposed a second near-depth
cutoff in `masked_surface_geometry`, independent of the CGN client cutoff.
SAM3 detected the bowl, but its masked depths were about .077–.141 metres.
The default .2–2 metre geometry range rejected every point. The helper now
accepts an explicit validated `depth_range_m`; its default remains unchanged
for existing callers. Replaying the stored patterned-bowl mask with .015–2
metres restored 18,740 points without new simulator execution or object state.

This does not resolve association: the restored surface touches the image
boundary, and its quantile-bound midpoint is .06305 metres from the previous
reference, exceeding the existing .04 metre gate. A clipped visible-surface
midpoint is not an object centre. The original failure had no valid wrist
surface; after repairing depth admission, the association gate still fails.
Do not relax that gate solely to pass this example. Multi-view or temporally
consistent public evidence must establish identity and lift before claiming a
grasp. No new lift, placement or official success follows from this replay.

The native-011 replay retained the same public start and source-hashed paths,
recording both cameras after each grasp waypoint. Fixed-agentview texture tracks
were effectively stationary through waypoint 11, then shifted approximately
4.5, 5.7 and 7.4 pixels across the following three transitions, before closure.
This supports target displacement during approach rather than a purely
post-close association error. The manually selected image region was used for
offline diagnosis only; it is not a runtime policy input or object-pose label.

`PublicSceneGraspPlanner` now exposes the existing ASPIRE `use_grasp_approach`
option, disabled by default. A fixed-snapshot diagnostic enabling this option
for the previously selected grasp produced a 31-waypoint plan with independent
FK endpoint errors below .001 metres and .005 radians. This metric is a planning
bias, not proof of a contact-free approach: the target still remains excluded
from collision checks, and the upstream metric uses partial pose weights.
Native replay must establish whether it reduces displacement before promotion.

## Target reconstruction and contact checks

Further fixed-observation diagnostics separated target-only and scene-only
world collision. Eight poses that passed robot-only IK all passed scene-only
IK and all failed target-only IK with a 1 cm target mesh. Rebuilding the same
masked public target points at 3 mm, while retaining the 1 cm scene mesh and
the same robot model and thresholds, admitted four poses: the grasp endpoint
and a 3 cm pregrasp in each parallel-finger orientation. Intermediate 1–2 cm
retreats did not pass; endpoint feasibility therefore cannot justify a straight
Cartesian insertion.

A two-stage offline plan subsequently passed with target collision enabled in
both stages and no observed non-finger surface overlap along the sampled path.
It used near-view geometry with an earlier robot start. A native policy must
actually acquire that near observation, retreat and revalidate the target;
this offline composition does not authorize access to future observations.

`PublicSceneGraspPlanner` exposes explicit `scene_mesh_pitch_m`, optional
`target_mesh_pitch_m`, and `allow_target_contact` for controlled candidates.
Defaults preserve the earlier reference. A fine target is rebuilt only from
the provided public depth, mask and calibration; the scene remains separate.
Partial visible geometry, approximate robot spheres and a successful plan do
not qualify native contact, grasping or a benchmark score.

## Required experiment sequence

1. Verify native/CuRobo FK and hand/fingertip conventions using measured robot
   state. Check backend dependencies and acquire the device lease before GPU work.
2. Replay fixed public sensor captures through segmentation and scene planning.
   Report empty reconstructions and planning failures separately; do not treat
   offline planning as a successful native episode.
3. Run the reference and current policy on identical fresh layout seeds, simulator
   settings, observation channels and execution budgets. Record commands, measured
   poses, masks/depth provenance and official outcomes.
4. Measure approach, grasp, visible lift and placement independently. Keep visual
   object association distinct from grasp confidence and official success.
5. Make one-factor Self-Harness proposals to primitives, memory or combinations.
   Use a separate holdout before promoting a reusable revision. Retain failed
   candidates and their evidence.
6. Expand across the full LIBERO, LIBERO-Plus, RoboTwin and RoboEvolve catalogs.
   Local development scores cannot satisfy the requested approximately 99% goal.

Runtime sensing currently uses SAM3 RGB/text masks and simulator-rendered metric
depth, not learned depth estimation. Historical development used privileged
feedback; this boundary does not retrospectively make development GT-free.
