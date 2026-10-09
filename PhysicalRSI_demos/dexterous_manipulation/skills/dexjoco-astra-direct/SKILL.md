---
name: dexjoco-astra-direct
description: Run a Galbot Astra Direct online agent in a prepared DexJoCo development episode using RGB, robot proprioception, native bounded action tools, and measured simulation feedback. This is a separate online-agent track before Tianji and BrainCo hardware migration.
---

# DexJoCo Astra Direct

Read workspace.json. This simulation uses Panda arms and Allegro hands. It does
not use the Tianji/BrainCo hardware joint layout. Call dexjoco_start once with
the exact task. Inspect all three images and current_eef. The simulator is
paused between tool calls; use image viewing, shell/Python geometry and notes
as useful. Do not read object ground truth or change the simulator.

The 46D proprioception layout is right TCP xyz+wxyz, left TCP xyz+wxyz, right
hand 16 joints, left hand 16 joints. Targets use the world frame, meters, and
unit quaternions in wxyz order. Allegro targets are radians, NOT [0,1] closure.
Read hand_limits_rad and hand_joint_names. Right finger order is index, middle,
ring, thumb; the left is ring, middle, index, thumb; each has four joints.

Call dexjoco_act with the latest observation_path and request_id. response:
mode=eef, steps (1..30), reason, target.right and target.left, each containing
position[3], quaternion_wxyz[4], and hand_joints_rad[16]. Explicitly hold the
other side when moving one arm. Each Cartesian target must be within 5 cm and
0.35 rad of the latest measured pose. The host holds the same target throughout
the requested native control steps and returns fresh RGB/proprioception. Read
the timing fields: one control step is 0.02 seconds, containing ten MuJoCo
integration steps. Thinking time advances no simulated time. Use 20-30 steps
for clear-space approach and settling, shorter chunks where contact needs
close observation. The torque controller often requires several chunks to
reach a target: inspect measured error and keep pursuing a still-valid fixed
goal until it is reached. One-step commands are only 20 ms of motion.

Read robot_geometry: it supplies measured palm frames and fingertip collision
capsules from robot-only FK. A TCP is the wrist/flange, not a fingertip. With
finger shape and wrist orientation held fixed, translate the TCP by the
difference between the desired fingertip point and its measured point; observe
again after any change of finger shape. Capsule centers are not surface points:
account for the reported radius and axis when planning shallow contact. These
fields do not provide object positions or hypothetical physics rollouts.

Camera calibration describes robot-mounted cameras in OpenGL convention:
camera -Z is the viewing direction and +Y is up. Use all views for geometry;
calibration is not object pose information. Do not infer task completion from
intended actions. Native success is the sole completion criterion.

For Photograph, follow the task's assignment: left hand supports the camera,
right hand operates the shutter after alignment with the visible logo. For
Assembly, left hand supports the tray and right hand grasps and inserts the
peg. These describe goals, not a prevalidated sequence of coordinates.

Read task_context for fixed task requirements. Photograph has a capture region
relative to the visible logo and a viewing-direction requirement, in addition
to physical shutter contact. Estimate the logo and camera from RGB; the task
context never supplies their current poses. Preserve a stable supporting grasp
before bringing the other hand into contact. Use small contact motions and
verify that the object follows the support hand before proceeding.

Maintain short grounded notes about current progress, tracking errors and the
next correction. workspace.json and each observation link to history.json, the
host's structured same-episode history. execution.tracking_error compares the
commanded targets with measured poses and hand joints; it does not prove grasp
attachment. Read native_control_steps_remaining to budget the actual task
horizon. Read context/candidate_memory.md when present. Candidate
memory is a hypothesis from previous
development episodes, not an observation of this episode. Never replay stored
actions without using current feedback. Continue until rollout_finished is true;
low expected success is not permission to end early. If recovery is no longer
possible, maintain a non-contact hold and observe through the remaining native
horizon. There is no model finish/stop tool. The runner handles native terminal
feedback and interruptions; never reset or claim an unobserved success.
