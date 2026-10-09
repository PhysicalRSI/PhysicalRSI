# From simulation to Tianji / BrainCo hardware

The research objective is reusable dexterous manipulation with two arms and
multi-finger hands. Simulation is the current development environment. Tianji
arms and BrainCo hands are the intended physical platform; the exact device
models, control interfaces and calibrations must be confirmed on that machine.

## What transfers and what needs new evidence

| Layer | Current evidence | Hardware work |
| --- | --- | --- |
| Acting agent | GPT-as-Policy sessions, images, shell, notes and bounded tools run in simulation | Supply synchronized real images and measured robot state through the device adapter |
| PhysicalRSI Core | Development, reflection, candidate freezing, paired evaluation, selection and lineage have run | Bind real reset conditions, trial budgets and execution evidence to the same contracts |
| Memory and helpers | A frozen geometry helper was loaded and invoked; task success remains absent | Revalidate image geometry and grasp procedures with actual cameras, hands and objects |
| Arm/hand control | Native Panda/Allegro simulation | Confirm Tianji SDK or ROS actions, joint order, frames, TCP and BrainCo command units |
| Outcome verification | Native task outcome plus post-episode simulator-assisted diagnostics | Establish independent observable acquisition, retention and task-completion checks |

The current simulation commands 16 Allegro joint angles per hand. BrainCo's
command channels and units must be confirmed on the actual device. Camera
calibration, hand geometry, grasp shapes, tracking limits
and timing require device-specific validation; action arrays cannot be reused
as hardware commands.

## Migration sequence

1. **Establish acquisition and retention in simulation.** For each hand, verify
   that the object leaves its support and follows the hand, then maintain the
   grasp during transport. The observed left-tray pickup failure is the current
   priority. Alignment errors alone do not establish this prerequisite.
2. **Bind the device and observations.** Establish the configuration from the
   actual robot. Confirm SDK/ROS action endpoints, joint names and units, robot
   geometry, TCP, camera calibration, timestamps and measured state. Device
   implementation and commissioning configuration are future work.
3. **Validate bounded motion and grasp behavior.** Check measured tracking and
   stop behavior, then separately verify left/right acquisition, small lifts
   and sustained retention. This contribution includes the simulation adapter;
   a physical device adapter must be implemented and validated separately.
4. **Connect the real experiment loop.** Record reproducible physical reset
   conditions, independent outcomes, images and execution receipts. Run parent
   and candidate under comparable conditions before selecting a new revision.
5. **Extend to coordinated contact tasks.** Only after the required grasps are
   established, evaluate camera handling and insertion with device-appropriate
   timing and contact control.

These are planned stages, not completed milestones. The current native
insertion predicate checks bottom contact but does not require left-tray pickup.
The requested bimanual behavior therefore needs its own independent stage
evidence alongside the environment's original outcome.

## Physical feedback and timing

The simulation replay diagnostics use privileged state after an episode. A
physical robot cannot reproduce that oracle. Real feedback must use available
measurements, calibrated visual checks and, initially where needed, independent
human labels. A force or tactile channel should enter the observation contract
only when it actually exists and has been integrated.

Simulation pauses while the language model reasons. Hardware does not share
that assumption. The device controller must execute bounded motions and
maintain an appropriate hold between agent decisions, with fresh observations
and device-side stopping behavior. Existing simulation timing does not establish
continuous real-time hardware control.
