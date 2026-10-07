"""Durable reminders for an active research agent; never fake agent reviews.

The local timer queues one overdue review. An agent must supply evidence,
reflection, alternatives and a next experiment to complete it. No LLM, shell
experiment, cloud provisioning or publication runs implicitly in the timer.
"""

import argparse
import math
from pathlib import Path
import time

from PhysicalRSI_core.infra.storage import atomic_json, digest, file_digest, locked, read_json


class ResearchHeartbeat:
    def __init__(self, root, *, interval_seconds=1800):
        if type(interval_seconds) is not int or interval_seconds < 1:
            raise ValueError("Heartbeat interval must be a positive integer")
        self.root = Path(root)
        self.interval = interval_seconds

    def tick(self, *, now=None):
        now = time.time() if now is None else now
        if type(now) not in (float, int) or not math.isfinite(now) or now < 0:
            raise ValueError("Invalid heartbeat time")
        with locked(self.root / ".lock"):
            state_path = self.root / "state.json"
            state = read_json(state_path) if state_path.exists() else dict(
                interval_seconds=self.interval, next_due=now, pending=None, completed=[])
            if state["interval_seconds"] != self.interval:
                raise ValueError("Heartbeat interval changed; use a new workspace")
            if state["pending"] is None and now >= state["next_due"]:
                request = dict(schema="physicalrsi.research-heartbeat/v1", due_at=state["next_due"],
                    created_at=now, interval_seconds=self.interval,
                    requirements=["Verify live processes and new completed evidence",
                        "Assess the current causal hypothesis and remaining uncertainty",
                        "Use autoresearch to compare alternatives if progress stalls",
                        "Declare the next bounded experiment or evidence-based wait"],
                    qualification=None)
                key = digest(request)
                atomic_json(self.root / "requests" / (key + ".json"), request)
                state["pending"] = key
            atomic_json(state_path, state)
            return state

    def complete(self, request_id, review, *, now=None):
        now = time.time() if now is None else now
        if type(now) not in (float, int) or not math.isfinite(now) or now < 0:
            raise ValueError("Invalid review time")
        required = {"evidence", "reflection", "alternatives", "decision", "next_experiment"}
        if not isinstance(review, dict) or set(review) != required:
            raise ValueError("Review needs evidence, reflection, alternatives, decision and next_experiment")
        if any(not isinstance(review[k], str) or not review[k].strip() for k in required - {"evidence"}):
            raise ValueError("Review reasoning must be nonempty")
        if not isinstance(review["evidence"], dict) or not review["evidence"]:
            raise ValueError("Review requires hashed local evidence")
        for path, sha in review["evidence"].items():
            if file_digest(Path(path)) != sha:
                raise ValueError("Review evidence changed")
        with locked(self.root / ".lock"):
            state = read_json(self.root / "state.json")
            if state["pending"] != request_id:
                raise ValueError("Review does not match the pending heartbeat")
            request = read_json(self.root / "requests" / (request_id + ".json"))
            if digest(request) != request_id or now < request["created_at"]:
                raise ValueError("Heartbeat request changed or review predates request")
            result = dict(request_id=request_id, completed_at=now, review=review, qualification=None)
            revision = digest(result)
            atomic_json(self.root / "reviews" / (revision + ".json"), result)
            state["completed"].append(revision)
            state["pending"] = None
            # Keep the original cadence; a delayed agent does not shift it.
            missed = max(1, int((now - state["next_due"]) // self.interval) + 1)
            state["next_due"] += missed * self.interval
            atomic_json(self.root / "state.json", state)
            return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--interval-seconds", type=int, default=1800)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    heartbeat = ResearchHeartbeat(args.workspace, interval_seconds=args.interval_seconds)
    previous = object()
    while True:
        state = heartbeat.tick()
        if state["pending"] != previous:
            print(state, flush=True)
            previous = state["pending"]
        if not args.watch:
            return
        time.sleep(min(30, args.interval_seconds))


if __name__ == "__main__":
    main()
