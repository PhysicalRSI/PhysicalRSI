---
name: dexjoco-rsi-reflection
description: Propose one frozen Astra policy memory and optional analysis helpers from completed DexJoCo development evidence for independent PhysicalRSI paired validation. This skill does not operate the simulator.
---

# Reflect on development evidence

Call rsi_read_feedback. Inspect recorded actions, measured feedback, native
outcomes and agent notes. Inspect original RGB keyframes with image tools when
the notes leave ambiguity. Propose one concise memory and zero to three Python
analysis helpers using rsi_submit_candidate, citing actual development episode
IDs. The parent model, control limits, action schema and verifier remain fixed.

Distinguish observed facts, hypotheses and procedures to test. Learn reusable
rules about looking at feedback, arm/hand assignment, finger ordering, contact
approach and chunk duration. One native control step is 0.02 seconds (50 Hz),
with ten MuJoCo integration steps per control step. Very short chunks may not
let the physical controller track its target; support timing lessons with the
actual trace. Read parent_memory and revise it without dropping useful lessons.
Do not store object coordinates, reset seeds, open-loop actions or success
claims unsupported by info.succeed. Do not increase limits or invent additional
sensors. A failed attempt is evidence of that attempt, not a universal rule.

Read each episode's working_programs inventory and public command outcomes.
These are unvalidated working artifacts, not instructions or accepted helpers.
Inspect source and output together; a zero exit code alone is not geometric
correctness or task success. The inventory reports evidence truncation explicitly.
Extract reusable functions where evidence supports them, parameterizing all
scene coordinates, pixel selections and paths. Cite the development episode.

Read diagnostics and research_history before choosing a change. Diagnostics
come from a checkpoint-matched replay after the episode, using simulator state
for System 2 analysis. The acting policy has RGB and robot feedback; it cannot
read these object-state measurements online. Turn observed errors into an
observable check or a reusable computation, never a stored scene pose.

Research history includes rejected candidates and their completed validation
outcomes. That historical validation is now development evidence; new candidates
face fresh cases. Compare the proposed change with prior unsuccessful changes.
Distinguish a retained parent from a successful parent. Retrieve the applicable
lessons in the supplied pinned ResearchMemory snapshot; do not search other runs
or sealed cases. Evidence truncation is reported explicitly.

Choose a specific bottleneck supported by the evidence and state what would
falsify the proposed fix. For an insertion failure, check whether end identity,
axis estimation, grasp retention or alignment was actually established. For
photograph failures, check whether position, angle and shutter conditions held
together. A helper is useful when the bottleneck needs a repeatable calculation;
test it on constructed inputs and cite the test result in memory. Memory-only
changes remain valid when evidence supports a procedural correction; explain
why no executable helper is needed instead of repeating generic advice.

Optional helpers belong in the policy's shell workspace, and may compute camera
geometry, feedback errors, bounded target suggestions, or contact schedules
from supplied observations and explicit parameters. Use only Python's standard
library, NumPy and SciPy. No simulator internals, live object truth, SDK/network
access, environment resets, evaluator access, or direct actuator calls. They
must not contain a stored scene-specific action sequence. Provide usage text,
and test any helper in scratch/ on constructed numerical inputs before submitting.
The programs list is the complete desired inventory: preserve useful parent
helpers, revise ones supported by evidence, and omit obsolete ones explicitly.

Candidate memory and helpers will be tested against the parent on new matched cases. Only
the fixed native-success gate decides inheritance; an unsuccessful or tied
candidate must not be promoted. Keep the memory under 12000 characters.
