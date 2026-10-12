"""Append-only execution evidence, with private evaluation in a separate tree."""
from pathlib import Path
import hashlib
import json
import os
import threading
import time

from .contracts import Assessment, Level, Observation, Task, digest, json_copy


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


class Episode:
    def __init__(self, root, task: Task, *, split="development", policy_id=None, qualification="offline"):
        if split not in {"development", "validation"} or qualification not in {"offline", "physical"}:
            raise ValueError("Explicit episode split and qualification required")
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        self.public = self.root / "execution"
        self.private = self.root / "evaluation"
        self.public.mkdir()
        self.private.mkdir(mode=0o700)
        self.task, self.split = task, split
        self.lock = threading.Lock()
        self.events = []
        self.assessments = []
        self.observations = {}
        self.observation_inputs = {}
        self.manifest = dict(schema="tianji.dual-system.episode/v1", task=task.public(),
                             split=split, policy_id=policy_id, qualification=qualification,
                             created_ns=time.time_ns())
        write_new(self.root / "episode.json", self.manifest)

    def event(self, kind, **data):
        # Trusted callers construct fields. Never pass an assessment or arbitrary metadata here.
        with self.lock:
            event = dict(index=len(self.events), kind=kind, time_ns=time.time_ns(), **json_copy(data))
            write_new(self.public / f"event-{len(self.events):06d}.json", event)
            self.events.append(event)
            return event["index"]

    def observation(self, obs: Observation):
        packet = obs.public()
        with self.lock:
            if obs.id in self.observations:
                if self.observation_inputs[obs.id] != digest(packet):
                    raise ValueError("Observation identity was reused for different evidence")
                return
            original_digest = digest(packet)
            # Keep pixels in the episode's evidence closure. External cache paths
            # may be removed after capture and are not immutable trial evidence.
            for frame in packet["frames"]:
                data = Path(frame["path"]).read_bytes()
                if hashlib.sha256(data).hexdigest() != frame["sha256"]:
                    raise ValueError("Camera evidence changed during retention")
                target = self.public / "images" / frame["sha256"]
                target.parent.mkdir(exist_ok=True)
                if target.exists():
                    if hashlib.sha256(target.read_bytes()).hexdigest() != frame["sha256"]:
                        raise ValueError("Retained camera evidence changed")
                else:
                    with target.open("xb") as stream:
                        stream.write(data)
                        stream.flush()
                        os.fsync(stream.fileno())
                frame["path"] = str(target)
            write_new(self.public / "observations" / (digest(obs.id) + ".json"), packet)
            self.observations[obs.id] = packet
            self.observation_inputs[obs.id] = original_digest

    def assessment(self, assessment: Assessment):
        if assessment.observation_id not in self.observations:
            raise ValueError("Assessment must reference recorded observation evidence")
        with self.lock:
            write_new(self.private / f"assessment-{len(self.assessments):06d}.json", assessment.private())
            self.assessments.append(assessment)

    def finish(self, *, outcome, solved_by, status="completed"):
        from .usage import usage
        if solved_by is not None and not isinstance(solved_by, Level):
            raise ValueError("Unknown controller attribution")
        entries = [e["level"] for e in self.events if e["kind"] == "controller_enter"]
        selected = [e["policy_id"] for e in self.events if e["kind"] == "l1_policy_selected"]
        result = dict(schema="tianji.dual-system.result/v1", task=self.task.id,
                      split=self.split, qualification=self.manifest["qualification"],
                      policy_id=selected[-1] if selected else self.manifest["policy_id"], outcome=outcome.value,
                      solved_by=solved_by.value if solved_by else None, status=status,
                      l1_success=(outcome.value == "success" and solved_by == Level.L1
                                  and not any(e in {"L2", "L3"} for e in entries)
                                  and not any(e["kind"] == "human_guidance" for e in self.events)),
                      l2_calls=entries.count("L2"), l3_calls=entries.count("L3"),
                      human_judgments=sum(a.source == "human" for a in self.assessments),
                      usage=usage(self.events), attribution="unassisted-L1/v2",
                      events_sha256=digest(self.events), finished_ns=time.time_ns())
        write_new(self.root / "result.json", result)
        return result

    def development_feedback(self, result):
        if self.split != "development":
            raise ValueError("Validation episodes cannot enter policy learning")
        # Reward labels may supervise selection/learning. Private explanations,
        # human answers, validation data, and evaluator files are not exported.
        return dict(task=self.task.id, split="development", instruction=self.task.public(),
                    outcome=result["outcome"], solved_by=result["solved_by"],
                    observations=json_copy(list(self.observations.values())),
                    execution=json_copy(self.events),
                    l2_calls=result["l2_calls"], l3_calls=result["l3_calls"])


def metrics(results):
    rows = list(results)
    qualifications = {r["qualification"] for r in rows}
    if len(qualifications) > 1:
        raise ValueError("Report offline and physical results separately")
    n = len(rows)
    return dict(episodes=n, qualification=next(iter(qualifications), None),
                overall_success_rate=sum(r["outcome"] == "success" for r in rows) / n if n else None,
                l1_success_rate=sum(r["l1_success"] for r in rows) / n if n else None,
                l2_episode_rate=sum(r["l2_calls"] > 0 for r in rows) / n if n else None,
                l3_episode_rate=sum(r["l3_calls"] > 0 for r in rows) / n if n else None,
                l2_calls=sum(r["l2_calls"] for r in rows), l3_calls=sum(r["l3_calls"] for r in rows),
                human_judgments=sum(r["human_judgments"] for r in rows),
                uncertain=sum(r["outcome"] == "uncertain" for r in rows),
                denominator="all supplied episode records, including uncertain outcomes")
