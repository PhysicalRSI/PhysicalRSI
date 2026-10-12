"""Host-side translation/pose control of visible image relationships.

No object pose or hand-eye calibration is consumed. The local response is learned
from changes in RGB features and *measured* robot FK after confirmed actions.
Alignment is an image criterion, never grasp/contact/task success. This adapter
has no collision planner and is not physically qualified by offline tests.
"""
from dataclasses import dataclass
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .capabilities import validate_direct_action
from .contracts import CAMERAS, CapabilityUnavailable, digest, vector
from .visual_perception import validate_features


@dataclass(frozen=True)
class ServoSettings:
    probe_m: float = .01
    max_translation_m: float = .03
    image_tolerance: float = .025
    min_confidence: float = .65
    gain: float = .7
    max_updates: int = 64
    max_image_age_s: float = 30
    probe_rotation_rad: float = .08
    max_rotation_rad: float = .12

    def __post_init__(self):
        values = vector([self.probe_m, self.max_translation_m, self.image_tolerance,
                         self.min_confidence, self.gain, self.max_image_age_s,
                         self.probe_rotation_rad, self.max_rotation_rad], 8)
        if (not 0 < values[0] <= values[1] <= .05 or not 0 < values[2] < 1
                or not 0 < values[3] <= 1 or not 0 < values[4] <= 1
                or not 0 < values[5] <= 30 or type(self.max_updates) is not int
                or not 3 <= self.max_updates <= 1000 or not 0 < values[6] <= values[7] <= .35):
            raise ValueError("Invalid host visual servo settings")


def validate_goal(arm, moving, target, views):
    validate_features([moving, target])
    if (arm not in ("left", "right") or not isinstance(views, list)
            or not 2 <= len(views) <= 3 or any(v not in CAMERAS for v in views)
            or len(set(views)) != len(views)):
        raise ValueError("Alignment needs one arm and two or three distinct camera views")
    return dict(arm=arm, moving=moving, target=target, views=list(views))


def image_error(measurement, goal, min_confidence):
    """Ordered moving-minus-target error; never subtract different cameras."""
    errors = []
    for camera in goal["views"]:
        features = measurement["views"][camera]["features"]
        a, b = features[goal["moving"]], features[goal["target"]]
        if (a["visibility"] != "visible" or b["visibility"] != "visible"
                or min(a["confidence"], b["confidence"]) < min_confidence):
            return None
        errors.extend(x-y for x, y in zip(vector(a["uv"], 2), vector(b["uv"], 2)))
    return np.asarray(errors, dtype=float)


def validate_relations(arm, relations, motion):
    if arm not in ("left", "right") or motion not in ("translation", "pose"):
        raise ValueError("Relations need an arm and translation/pose control mode")
    if not isinstance(relations, list) or not 2 <= len(relations) <= 12:
        raise ValueError("Declare 2..12 visible feature relations")
    cameras, features, clean = set(), [], []
    for row in relations:
        if (not isinstance(row, dict) or set(row) not in
                ({"camera", "moving", "target"}, {"camera", "moving", "target", "offset"})):
            raise ValueError("A relation needs camera, moving, target and optional feature-vector offset")
        validate_features([row["moving"], row["target"]])
        if row["camera"] not in CAMERAS:
            raise ValueError("Unknown relation camera")
        cameras.add(row["camera"])
        names = [row["moving"], row["target"]]
        offset = row.get("offset")
        if offset is not None:
            if not isinstance(offset, dict) or set(offset) != {"from", "to", "scale"}:
                raise ValueError("Offset must scale a currently observed feature vector, not fixed pixels")
            validate_features([offset["from"], offset["to"]])
            scale = vector([offset["scale"]], 1)[0]
            if not -2 <= scale <= 2:
                raise ValueError("Feature-vector scale must be within [-2, 2]")
            names += [offset["from"], offset["to"]]
            offset = dict(offset)
        features += [name for name in names if name not in features]
        clean.append(dict(camera=row["camera"], moving=row["moving"], target=row["target"], offset=offset))
    # Deduplicate after each row too (offset endpoints may name its landmarks).
    features = list(dict.fromkeys(features))
    validate_features(features)
    if len(cameras) < 2 or len({digest(r) for r in clean}) != len(clean):
        raise ValueError("Relations require distinct constraints in at least two cameras")
    if motion == "pose" and (len(clean) < 3 or len({r["moving"] for r in clean}) < 3):
        raise ValueError("Pose control needs at least three moving landmarks; response must also be full rank")
    return dict(arm=arm, motion=motion, relations=clean), features


def relation_error(measurement, goal, min_confidence):
    errors = []
    for row in goal["relations"]:
        points = measurement["views"][row["camera"]]["features"]
        names = [row["moving"], row["target"]]
        if row["offset"] is not None:
            names += [row["offset"]["from"], row["offset"]["to"]]
        if any(points[n]["visibility"] != "visible" or points[n]["confidence"] < min_confidence for n in names):
            return None
        uv = {n:np.asarray(vector(points[n]["uv"], 2)) for n in names}
        desired = uv[row["target"]]
        if row["offset"] is not None:
            offset = row["offset"]
            desired = desired + offset["scale"]*(uv[offset["to"]]-uv[offset["from"]])
        if np.any(desired < 0) or np.any(desired > 1):
            return None
        errors.extend(uv[row["moving"]]-desired)
    return np.asarray(errors)


def rotation(q):
    return Rotation.from_quat(np.asarray(vector(q, 4))[[1, 2, 3, 0]])


def fit_response(transitions, dimensions, dofs=3):
    """Fit d(image error) / d(measured pose increment), not commanded motion.

    Rank-deficient evidence cannot establish the requested three/six-axis control.
    Only recent local transitions are used; there is no fixed task trajectory.
    """
    if dofs not in (3, 6) or dimensions < dofs or len(transitions) < dofs:
        return None
    recent = transitions[-24:]
    x = np.asarray([vector(t["measured_translation"], 3) +
                    (vector(t["measured_rotation"], 3) if dofs == 6 else []) for t in recent])
    y = np.asarray([vector(t["image_error_change"], dimensions) for t in recent])
    singular = np.linalg.svd(x, compute_uv=False)
    if singular[-1] < 1e-5 or singular[0]/singular[-1] > 100:
        return None
    solution, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    response = solution.T
    spectrum = np.linalg.svd(response, compute_uv=False)
    if spectrum[-1] < .05 or spectrum[0]/spectrum[-1] > 100:
        return None
    residual = float(np.linalg.norm(x @ solution-y))
    if residual > max(.01, .4*float(np.linalg.norm(y))):
        return None
    return response


class VisualServo:
    def __init__(self, perception, *, settings=None):
        self.perception = perception
        self.settings = settings or ServoSettings()

    def align(self, context, arm, moving, target, views, *, deadline):
        validate_goal(arm, moving, target, views)
        return self.relate(context, arm, [dict(camera=c, moving=moving, target=target) for c in views],
                           "translation", deadline=deadline)

    def relate(self, context, arm, relations, motion, *, deadline):
        goal, features = validate_relations(arm, relations, motion)
        dofs = 3 if motion == "translation" else 6
        if self.perception is None:
            raise CapabilityUnavailable("RGB feature perception is not configured")
        settings = self.settings
        transitions, observations = [], []
        updates, stalled, regressed = 0, 0, 0

        def finish(status, reason, error=None):
            result = dict(status=status, reason=reason, goal=goal, updates=updates,
                          observation_ids=observations,
                          image_error=None if error is None else error.tolist(),
                          independent_success=False)
            context.episode.event("visual_alignment_finished", level="L1", result=result)
            return result

        def measure():
            context.robot.hold()
            obs = context.checkpoint(force=True)
            measured = self.perception.look(obs, features, deadline=deadline)
            if (measured["observation_id"] != obs.id or measured["captured_ns"] != obs.captured_ns
                    or measured["frames"] != [dict(camera=f.camera, captured_ns=f.captured_ns,
                                                    sha256=f.sha256) for f in obs.frames]):
                raise CapabilityUnavailable("Feature evidence does not match the current observation")
            context.episode.event("visual_features", level="L1", observation_id=obs.id,
                                  features=features, result=measured)
            observations.append(obs.id)
            return obs, relation_error(measured, goal, settings.min_confidence)

        obs, error = measure()
        while True:
            if error is None:
                return finish("unobservable", "Required landmarks are occluded, ambiguous or low-confidence")
            try:
                obs.validate(live=True, max_age_s=settings.max_image_age_s)
            except ValueError:
                return finish("stale_observation", "Perception finished after the action image expired", error)
            if np.max(np.abs(error)) <= settings.image_tolerance:
                return finish("image_aligned", "Image criterion only; contact and task success are not inferred", error)
            if time.monotonic() >= deadline or updates >= settings.max_updates:
                return finish("residual", "Visual control budget ended with known action receipts", error)
            response = fit_response(transitions, len(error), dofs)
            if response is None:
                if updates >= dofs:
                    return finish("insufficient_motion_evidence", "Measured changes do not establish a local visual response", error)
                delta = np.eye(dofs)[updates] * (settings.probe_m if updates < 3 else settings.probe_rotation_rad)
                phase = "identify_response"
            else:
                delta = -settings.gain * np.linalg.lstsq(response, error, rcond=None)[0]
                phase = "reduce_image_error"
            length = float(np.linalg.norm(delta))
            if not np.isfinite(length) or length < 1e-6:
                return finish("residual", "No supported motion reduces the measured relation", error)
            # Bound translation and orientation independently without changing
            # the direction of the joint solution in the fitted task space.
            ratio = max(1., np.linalg.norm(delta[:3])/settings.max_translation_m)
            if dofs == 6:
                ratio = max(ratio, np.linalg.norm(delta[3:])/settings.max_rotation_rad)
            delta /= ratio
            # No refresh may disguise old features as a newly observed action.
            # Every command is bound to the exact image used to compute it.
            try:
                obs.validate(live=True, max_age_s=settings.max_image_age_s)
            except ValueError:
                return finish("stale_observation", "Perception finished after the action image expired", error)
            targets = {side:dict(position=list(pose["position"]),
                                 quaternion_wxyz=list(pose["quaternion_wxyz"]), hand=None)
                       for side, pose in obs.eef.items()}
            targets[arm]["position"] = (np.asarray(obs.eef[arm]["position"])+delta[:3]).tolist()
            if dofs == 6:
                # FK quaternions map tool axes into base axes. Left multiply a
                # base-frame incremental rotation, matching measured log below.
                q = (Rotation.from_rotvec(delta[3:])*rotation(obs.eef[arm]["quaternion_wxyz"])).as_quat()
                targets[arm]["quaternion_wxyz"] = q[[3, 0, 1, 2]].tolist()
            arguments = dict(observation_id=obs.id, targets=targets, steps=5,
                             reason="L1 visual alignment: " + phase)
            validate_direct_action(arguments, obs)
            # The generic port enforces its own current-state bounds and stop.
            # Unknown action effects escape; no alternate move or retry follows.
            context.action("visual_servo_eef", dict(arguments, goal=goal), lambda:
                context.robot.act_eef(targets, 5, checkpoint=context.checkpoint,
                                      observation=obs))
            updates += 1
            new_obs, new_error = measure()
            if new_error is None:
                return finish("unobservable", "Landmarks disappeared after a confirmed action")
            # The uncontrolled camera/arm must remain fixed. Translation-only
            # mode additionally excludes unmodeled rotation of the moving arm.
            other = "left" if arm == "right" else "right"
            rotated = any(abs(sum(a*b for a, b in zip(obs.eef[s]["quaternion_wxyz"],
                new_obs.eef[s]["quaternion_wxyz"]))) < np.cos(.03/2) for s in ((arm, other) if dofs == 3 else (other,)))
            if rotated or np.linalg.norm(np.asarray(new_obs.eef[other]["position"])-obs.eef[other]["position"]) > .003:
                return finish("residual", "An uncontrolled pose changed beyond the fitted motion model", new_error)
            actual = np.asarray(new_obs.eef[arm]["position"])-obs.eef[arm]["position"]
            actual_rotation = (rotation(new_obs.eef[arm]["quaternion_wxyz"])*
                               rotation(obs.eef[arm]["quaternion_wxyz"]).inv()).as_rotvec()
            transition = dict(before=obs.id, after=new_obs.id, goal=goal,
                              commanded_translation=delta[:3].tolist(), measured_translation=actual.tolist(),
                              commanded_rotation=delta[3:].tolist() if dofs == 6 else [0., 0., 0.],
                              measured_rotation=actual_rotation.tolist(),
                              image_error_change=(new_error-error).tolist(), phase=phase)
            transitions.append(transition)
            context.episode.event("visual_servo_transition", level="L1", transition=transition,
                                  transition_sha256=digest(transition))
            actual_step = np.r_[actual, actual_rotation] if dofs == 6 else actual
            stalled = stalled+1 if np.linalg.norm(actual_step) < 1e-5 else 0
            regressed = regressed+1 if phase != "identify_response" and np.linalg.norm(new_error) > np.linalg.norm(error)*1.1 else 0
            obs, error = new_obs, new_error
            if stalled >= 3 or regressed >= 2:
                return finish("residual", "Observed motion stalled or image error repeatedly increased", error)
