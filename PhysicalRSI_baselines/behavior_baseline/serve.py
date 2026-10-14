"""Serve a frozen reviewed program using the official BEHAVIOR websocket protocol."""
import argparse
import logging
import os
from pathlib import Path
from time import monotonic

import numpy as np

from PhysicalRSI_core.contracts import Context
from PhysicalRSI_core.infra.storage import digest, file_digest, read_json
from PhysicalRSI_core.self_harness.artifacts import verify_harness
from .openpi_backend import OpenPIBackend, select_task
from .program import ProgramPolicy
from .engine import verify as verify_engine
from .diagnostics import DevelopmentTrace, validate_binding


class ServingPolicy:
    def __init__(self, program, *, task, episode_seconds=3600):
        if episode_seconds <= 0:
            raise ValueError("A finite episode deadline is required")
        import math
        if not math.isfinite(episode_seconds):
            raise ValueError("A finite episode deadline is required")
        self.program, self.task, self.episode_seconds = program, task, episode_seconds
        self.context = None
        self.episodes = 0

    def reset(self):
        if self.context is not None:
            self.program.end_episode(self.context)
        self.context = None

    def act(self, observation):
        if self.context is None:
            self.episodes += 1
            self.context = Context(f"served-{self.episodes}", deadline=monotonic() + self.episode_seconds,
                                   harness_revision=self.program.revision)
            self.program.begin_episode(self.task, {}, observation, self.context)
        action = np.asarray(self.program.act(observation, self.context), dtype=np.float32)
        # Upstream WebsocketPolicyServer serializes torch tensors.
        import torch
        return torch.from_numpy(action)

    def metadata(self):
        identity = self.program.identity()
        return {"schema": "physicalrsi.behavior-policy-service/v1", "task": self.task,
                "service_pid": os.getpid(), "launch_nonce": os.environ.get("PHYSICALRSI_SERVICE_NONCE"),
                "candidate_sha256": self.program.revision, "policy_identity": identity,
                "policy_identity_sha256": digest(identity), "episode_seconds": self.episode_seconds,
                "service_implementation_sha256": file_digest(Path(__file__)),
                "observation_scope": "onboard_projection", "qualification": None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path, help="Core harness manifest JSON")
    parser.add_argument("--task", help="Explicit task for a frozen multitask checkpoint")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--episode-seconds", type=float, default=3600)
    parser.add_argument("--development-binding", type=Path)
    parser.add_argument("--development-trace", type=Path)
    args = parser.parse_args()
    candidate = read_json(args.candidate)
    verify_harness(candidate)
    verify_engine(candidate)
    specification = read_json(Path(candidate["root"]) / "backend.json")
    if (args.development_binding is None) != (args.development_trace is None):
        raise ValueError("Supply both the development binding and trace destination")
    task = select_task(specification, args.task)
    case_id = None
    if args.development_binding is not None:
        case_id = validate_binding(read_json(args.development_binding),
                                   candidate_sha256=verify_harness(candidate), task=task)
    if specification.get("kind") == "champion-pi05-b1k/v1":
        from .champion_backend import ChampionBackend, transport_class
        if case_id is not None:
            # The binding above admits development cases only. Upstream logs
            # explain stage voting and compression without changing actions.
            logging.getLogger("b1k.shared.eval_b1k_wrapper").setLevel(logging.INFO)
        backend = ChampionBackend(specification, task=task)
        WebsocketPolicyServer = transport_class(specification)
    else:
        backend = OpenPIBackend(specification, task=task)
        from openpi.serving.websocket_b1k_server import WebsocketPolicyServer
    recorder = (DevelopmentTrace(args.development_trace, split="development",
                                 candidate_sha256=verify_harness(candidate),
                                 task=task, case_id=case_id)
                if case_id is not None else None)
    program = ProgramPolicy(candidate, backend, diagnostics=recorder)
    policy = ServingPolicy(program, task=task, episode_seconds=args.episode_seconds)
    WebsocketPolicyServer(policy, host=args.host, port=args.port, metadata=policy.metadata()).serve_forever()


if __name__ == "__main__":
    main()
