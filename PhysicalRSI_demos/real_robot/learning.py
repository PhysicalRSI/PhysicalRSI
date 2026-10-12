"""Offline recovery of development evidence and L2/L3-to-L1 proposals.

This entry point imports no robot, SDK or ROS modules. It reads execution evidence
and terminal outcome labels; it never reads the private evaluation directory.
"""
import json
from pathlib import Path

from .contracts import Observation, Task, digest
from .evidence import metrics, write_new
from .library import PolicyLibrary, foundation_identity
from .references import use_physicalrsi


def read_development_episode(directory):
    root = Path(directory).resolve(strict=True)
    metadata = json.loads((root / "episode.json").read_text())
    if metadata.get("schema") != "tianji.dual-system.episode/v1" or metadata.get("split") != "development":
        raise ValueError("Only development episodes may enter learning")
    task = Task(**metadata["task"])
    task.public()
    result = json.loads((root / "result.json").read_text())
    if (result.get("schema") != "tianji.dual-system.result/v1"
            or result.get("split") != "development" or result.get("task") != task.id
            or result.get("qualification") != metadata.get("qualification")
            or result.get("qualification") not in {"offline", "physical"}
            or result.get("outcome") not in {"success", "failure", "uncertain"}
            or result.get("solved_by") not in {None, "L1", "L2", "L3"}):
        raise ValueError("Episode outcome metadata differs")
    public = root / "execution"
    if any(p.is_symlink() for p in (public, public / "observations", public / "images")):
        raise ValueError("Execution evidence directories must not be symlinks")
    events = []
    for index, path in enumerate(sorted(public.glob("event-*.json"))):
        if path.is_symlink() or path.name != f"event-{index:06d}.json":
            raise ValueError("Execution event history is not contiguous physical files")
        event = json.loads(path.read_text())
        if event.get("index") != index or event.get("level") not in {"L1", "L2", "L3"}:
            raise ValueError("Invalid execution event identity")
        events.append(event)
    if digest(events) != result.get("events_sha256"):
        raise ValueError("Execution history differs from the terminal result")
    from .contracts import Frame
    for event in events:
        if event["kind"] == "video_reference":
            reference = event["reference"]
            if (reference.get("schema") != "tianji.historical-rgb-reference/v1"
                    or reference.get("has_motor_or_eef_trajectory") is not False
                    or reference.get("current_task_success") is not False):
                raise ValueError("Historical reference cannot be current execution or trajectory evidence")
            for item in reference["frames"]:
                path = Path(item["path"])
                if path.is_symlink() or not path.resolve().is_relative_to(public.resolve() / "video-references"):
                    raise ValueError("Reference pixels must be retained in this episode's video-references")
                Frame(**{k:item[k] for k in ("camera", "path", "captured_ns", "sha256")}).validate()
    for level in ("L2", "L3"):
        count = sum(e["kind"] == "controller_enter" and e["level"] == level for e in events)
        if type(result.get(level.lower()+"_calls")) is not int or result[level.lower()+"_calls"] != count:
            raise ValueError("Controller counts differ from recorded execution")
    observations = []
    for path in (public / "observations").glob("*.json"):
        if path.is_symlink():
            raise ValueError("Observation records must not be symlinks")
        packet = json.loads(path.read_text())
        if path.name != digest(packet["id"])+".json":
            raise ValueError("Observation filename does not bind its identity")
        for frame in packet["frames"]:
            source = Path(frame["path"])
            if source.is_symlink() or not source.resolve().is_relative_to(public.resolve() / "images"):
                raise ValueError("Learning images must be retained execution evidence")
        obs = Observation.from_public(packet)
        observations.append(obs.public())
    observations.sort(key=lambda o: (o["captured_ns"], o["id"]))
    if not observations:
        raise ValueError("Learning requires retained RGB and measured observations")
    # Construct the allowlisted projection; never forward root/result metadata
    # wholesale, nor read or attach private judge explanations.
    feedback = dict(task=task.id, split="development", instruction=task.public(),
                    outcome=result["outcome"], solved_by=result["solved_by"],
                    observations=observations, execution=events,
                    l2_calls=result["l2_calls"], l3_calls=result["l3_calls"])
    return task, feedback, result


def learn_from_episodes(directories, *, library, model, output, parent=None):
    """Recover a proposal after model timeout or learn from several rescue runs.

    The output is a candidate, not an automatically promoted incumbent. Its
    public feedback is frozen before the model call, including failed rescues.
    """
    paths = [Path(p).resolve(strict=True) for p in directories]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("Declare distinct development episodes")
    entries = [read_development_episode(p) for p in paths]
    task = entries[0][0]
    if any(t.public() != task.public() for t, _, _ in entries):
        raise ValueError("Task instructions and criteria must match across development episodes")
    if not any(r["l2_calls"] or r["l3_calls"] or r.get("usage", {}).get("l3_uses") for _, _, r in entries):
        raise ValueError("No L2/L3 rescue evidence is present")
    original_parent = parent
    if parent is not None:
        previous = PolicyLibrary.load(parent)
        config = json.loads((Path(parent["root"]) / "configuration.json").read_text())
        if config["task"] != task.public():
            raise ValueError("Parent policy belongs to another task")
        try:
            PolicyLibrary.load_for_execution(parent, foundation=library.foundation)
        except ValueError:
            parent = library.freeze(task=task, source=previous["source"], memory=previous["memory"])
    summary = metrics([r for _, _, r in entries])
    feedback = [row for _, row, _ in entries]
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "development-public.json", feedback)
    write_new(output / "input.json", dict(episodes=[str(p) for p in paths],
              feedback_sha256=digest(feedback), parent=parent, original_parent=original_parent, metrics=summary))
    candidate = library.propose(task=task, parent=parent, feedback=feedback,
                               model=model, output=output / "proposal")
    # Re-read evidence after model latency: changed input cannot silently produce
    # a candidate presented as derived from the original frozen development set.
    if digest([read_development_episode(p)[1] for p in paths]) != digest(feedback):
        raise ValueError("Development evidence changed during proposal")
    record = dict(candidate=candidate, parent=parent, original_parent=original_parent, metrics=summary,
                  feedback_sha256=digest(feedback), promoted=False,
                  stage="candidate_requires_independent_validation" if candidate else "unchanged",
                  physical_success_improvement=None)
    write_new(output / "learning.json", record)
    return record


def selected_policy(lineage, *, task, qualification, environment_capture=None):
    """Resolve a verified SelfHarness selection for subsequent execution.

    Bootstrap is not promotion. Offline comparison results cannot become physical
    qualifications simply because the caller now has a robot connected.
    """
    use_physicalrsi()
    from PhysicalRSI_core.lineage import HarnessState
    record = HarnessState(lineage).resolve()
    if record["action"] != "selection":
        raise ValueError("Lineage has no independently selected incumbent")
    manifest = record["harness"]
    PolicyLibrary.load_for_execution(manifest)
    config = json.loads((Path(manifest["root"]) / "configuration.json").read_text())
    if config["task"] != task.public():
        raise ValueError("Selected policy task/criteria differ")
    # SelfHarness keeps every trial as part of its hash-verified evidence closure.
    results = []
    trial_directories = []
    for path in record["evidence"]:
        p = Path(path)
        if p.name == "result.json":
            value = json.loads(p.read_text())
            if value.get("schema") == "tianji.dual-system.result/v1" and value.get("policy_id") == manifest["id"]:
                results.append(value)
                trial_directories.append(p.parent)
    if (not results or qualification not in {"offline", "physical"}
            or any(r["qualification"] != qualification or r["split"] != "validation"
                   or r["status"] != "completed" or r["l2_calls"] or r["l3_calls"] for r in results)):
        raise ValueError("Selected incumbent lacks matching independent L1-only qualification")
    if qualification == "physical":
        from .host_environment import verify_trial_environment
        if not callable(environment_capture):
            raise ValueError("Physical lineage requires an explicit host environment capture")
        current = environment_capture()
        for directory in trial_directories:
            verify_trial_environment(directory, current, evidence=record["evidence"])
    return manifest
