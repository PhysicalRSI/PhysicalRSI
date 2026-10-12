"""PhysicalRSI primitive interface, with explicit Tianji 7+6 motor adaptation."""
import math

from .contracts import vector
from .timing import direct_servo_ticks

LOOK_FEATURE_LIMIT = 16
LOOK_NAME_LIMIT = 160

API = dict(
    schema="tianji.reusable-code-policy/v5",
    policy_signature="def policy(robot, memory)",
    methods={
        "relate": dict(signature="robot.relate(arm, relations, motion)", description=(
            "Satisfy multi-view visual relations by translation or pose (translation+rotation). "
            "Each relation has camera, moving landmark, target landmark, and optional offset "
            "{from: landmark, to: landmark, scale: dimensionless number in [-2,2]}. Desired point "
            "is target + scale*(to-from) in that same current image. This can express object-sized "
            "clearance from a support without a fixed cm/pixel goal. Require at least two cameras; "
            "pose mode needs at least three distinct moving landmarks and a full-rank measured response. "
            "Hand commands are preserved, so lifting a held object requires actual grasp evidence first. "
            "Returns the same residual/image-only status as align; no contact, collision or task-success claim.")),
        "align": dict(signature="robot.align(arm, moving, target, views)", description=(
            "Translate one arm to align named visible moving and target landmarks in two or three "
            "camera views. No object coordinates required. Host code identifies the local visual "
            "response from measured motion and repeatedly corrects using new RGB. Keeps orientation "
            "and hand commands unchanged. Returns image_aligned, residual, unobservable, stale_observation "
            "or insufficient_motion_evidence, never task success. Projection alignment does not prove "
            "grasp/contact; no collision or contact planner is provided. Software tested offline only.")),
        "look": dict(signature="robot.look(features)",
            feature_constraints=dict(min_count=1, max_count=LOOK_FEATURE_LIMIT, unique=True,
                                     max_name_characters=LOOK_NAME_LIMIT), description=(
            "Read three current RGB images through execution-side perception. features must be a list of "
            f"1..{LOOK_FEATURE_LIMIT} unique, nonempty visible landmark names, each at most {LOOK_NAME_LIMIT} characters. "
            "Deduplicate shared object landmarks. Split task phases into separate calls when needed; "
            "do not merge geometry from different observation IDs into one pose/clearance calculation. "
            "Return observation-bound normalized image uv, visibility, confidence "
            "and pairwise delta_uv (target minus source) per camera. Occluded/absent/uncertain points "
            "have uv=null. No metric object coordinates or calibration required. Image overlap does "
            "not prove 3D alignment, contact or success. This estimates features; code selects actions.")),
        "scene": dict(signature="robot.scene(names)", description=(
            "Ground named task objects/features from current camera evidence in the calibrated robot "
            "base frame. Return observation-bound measured/estimated poses and uncertainty, never "
            "hidden object geometry or success. Missing calibration/geometry raises a capability error.")),
        "reobserve": dict(signature="robot.reobserve()", description="Return current three RGB views, measured 26D state and robot EEF poses."),
        "approach": dict(signature="robot.approach(arm, position_m, quaternion_wxyz)", description=(
            "Execute a scene-derived absolute EEF goal through host planning/control. This is a reusable "
            "goal, not a list of centimetre moves. The monitor runs at control checkpoints. Returns "
            "measured state and execution evidence, not task success. Preserve the other arm/hand.")),
        "gripper": dict(signature="robot.gripper(arm, action, mode=None)", description=(
            "Semantic Revo2 gripper: arm left/right; action open/close/hold/release. "
            "Modes wrap (five-finger) and tripod (three-finger). Open/close default to wrap. "
            "Release without mode uses the current known grasp; otherwise specify a mode. "
            "Hold takes no mode and preserves existing commands. Host executes configured "
            "finger coordination as a complete measured-feedback job, independent of L2 steps. "
            "Initial manufacturer-derived profiles are available without human teaching; they remain unvalidated. "
            "Choose a configured grasp mode, not six motor numbers. Demonstration is optional. "
            "Returns position_reached/held/stable_residual/timeout_residual. Contact remains unknown; "
            "tool completion never proves a grasp. Preserve the other hand.")),
        "yield_control": dict(signature="robot.yield_control(reason)", description=(
            "Report a public capability/task limitation so the coordinator can try L2. "
            "Cannot bypass a hardware fault, unknown action effect or manual stop.")),
    },
    rules=["Task code derives decisions from current visual features/relations or supported metric scene and reusable memory.",
           "Reusable does not mean number-free: current perception/control may produce numbers; do not bind code to one object's fixed coordinates.",
           "Do not inspect files, import SDKs, send raw joint commands or access judge data.",
           "Never retry an action with unknown effect.",
           "The independent monitoring system alone adjudicates task outcome."],
    dimensions=dict(left_arm=7, left_hand=6, right_arm=7, right_hand=6),
)


def validate_direct_action(arguments, observation):
    """GPT-as-Policy gpt_only_client/validation semantics, adapted to Revo2.

    Keep fresh observation IDs and 1..5 policy steps of 40 ms each; return new RGB.
    Position/orientation bounds follow the reference. Direct arm steps preserve
    the hands; semantic gripper jobs execute separately to measured completion.
    """
    required = {"observation_id", "targets", "steps", "reason"}
    if set(arguments) != required or arguments["observation_id"] != observation.id:
        raise ValueError("Direct action must bind to the current observation")
    direct_servo_ticks(arguments["steps"])
    if not isinstance(arguments["reason"], str) or not arguments["reason"].strip():
        raise ValueError("Visible evidence and action purpose required")
    targets = arguments["targets"]
    if not isinstance(targets, dict) or set(targets) != {"left", "right"}:
        raise ValueError("Explicit dual-arm targets required")
    for arm, target in targets.items():
        if set(target) != {"position", "quaternion_wxyz", "hand"}:
            raise ValueError("EEF pose and explicit hand target/null required")
        position = vector(target["position"], 3)
        q = vector(target["quaternion_wxyz"], 4)
        if abs(math.sqrt(sum(v*v for v in q)) - 1) > 1e-4:
            raise ValueError("Unit wxyz quaternion required")
        old = observation.eef[arm]
        if math.dist(position, old["position"]) > .05 + 1e-9:
            raise ValueError("Direct EEF target exceeds reference 5 cm envelope")
        dot = abs(sum(a*b for a, b in zip(q, old["quaternion_wxyz"])))
        angle = 2 * math.acos(min(1.0, dot / math.sqrt(sum(v*v for v in q))))
        if angle > .35 + 1e-9:
            raise ValueError("Direct EEF rotation exceeds reference 0.35 rad envelope")
        if target["hand"] is not None:
            raise ValueError("Direct arm actions preserve hands (hand=null); use the semantic gripper tool")
