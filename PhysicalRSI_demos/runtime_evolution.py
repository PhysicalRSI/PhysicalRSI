"""Self-Harness compares memory candidates through actual CPU service processes.

Only local software contracts are evaluated. The model is fixed and scripted;
the example tests evidence/infra integration, not autonomous robot improvement.
"""

import shutil
import sys
import uuid
from pathlib import Path

from PhysicalRSI_demos.runtime import SPEC
from PhysicalRSI_demos.runtime_service import ENVIRONMENT, MODEL
from PhysicalRSI.Embodied_Harness.memory.store import MemoryStore
from PhysicalRSI.Embodied_Harness.memory.workspace import MemoryWorkspace
from PhysicalRSI.Embodied_Harness.skills.composition import sequence
from PhysicalRSI.Embodied_Harness.skills.remote import policy_skill
from PhysicalRSI_core.contracts import Context, Contract
from PhysicalRSI_core.infra.clients import EnvironmentClient, ModelClient
from PhysicalRSI_core.infra.execution import Execution
from PhysicalRSI_core.infra.resources import ResourcePool
from PhysicalRSI_core.infra.services import Services, ServiceSpec
from PhysicalRSI_core.infra.storage import atomic_json, digest, file_digest, read_json
from PhysicalRSI_core.infra.trajectory import EpisodeWriter, validate_episode
from PhysicalRSI_core.lineage import HarnessState
from PhysicalRSI_core.self_harness import SelfHarness, now, selection
from PhysicalRSI_core.self_harness.artifacts import CLOSURE, bind_harness_modes, verify_harness

SCOPE = "local_counter_service_contract; no physical qualification"
PROTOCOL = "counter-service-v1"


def implementation():
    root = Path(__file__).resolve().parents[1]
    paths = [
        *root.joinpath("PhysicalRSI_core").rglob("*.py"),
        *root.joinpath("PhysicalRSI/Embodied_Harness").rglob("*.py"),
        *root.joinpath("PhysicalRSI_demos").glob("runtime*.py"),
    ]
    return {str(path.relative_to(root)): file_digest(path) for path in paths}


def manifest(root, name, **metadata):
    return bind_harness_modes(dict(
        id=name,
        root=str(root),
        components={
            kind: {kind + ".json": file_digest(root / (kind + ".json"))}
            for kind in CLOSURE
        },
        **metadata,
    ))


class Evaluation:
    def __init__(self, environment, model):
        self.environment, self.model = environment, model

    def identity(self):
        return {"kind": PROTOCOL, "implementation": implementation()}

    def admit(self, candidate, output):
        root = Path(candidate["root"])
        accepted = (
            read_json(root / "dependencies.json") == implementation()
            and read_json(root / "memory_rules.json").get("target") in {0, 2}
            and read_json(root / "foundation.json") == MODEL
        )
        return dict(
            accepted=accepted,
            freeze_sha256=verify_harness(candidate),
            evidence={"source": implementation()},
            reason="Pinned local adapter and model",
        )

    def execute(self, candidate, case, output):
        memory = MemoryStore(output / "memory").snapshot(
            {"goal": read_json(Path(candidate["root"]) / "memory_rules.json")}
        )
        self.model.reset(uuid.uuid4().hex)
        initial = self.environment.reset(uuid.uuid4().hex)
        recorder = EpisodeWriter(
            output / "raw",
            spec=SPEC,
            initial_observation=initial,
            metadata={
                "harness_revision": verify_harness(candidate),
                "memory_revision": memory.revision,
                "model": MODEL,
                "case": case,
                "scope": SCOPE,
            },
        )
        skill = policy_skill(
            "skill.counter",
            digest(implementation()),
            environment=self.environment,
            model=self.model,
            recorder=recorder,
            max_chunks=2,
        )
        composed = sequence(
            "memory_policy", memory.reader("goal", Contract("goal")), skill
        )
        execution = Execution(
            output / "steps",
            observe=lambda op, ctx: {
                "state": self.environment.observe()["state"].tolist()
            },
        )
        composed(
            None,
            Context(
                case, execution=execution, harness_revision=verify_harness(candidate)
            ),
        )
        # Independent outcome read from the environment, not the skill's prose.
        measured = self.environment.observe()["state"].tolist()
        metadata = validate_episode(recorder.path, spec=SPEC)
        passed = measured == [2.0] and metadata["is_success"]
        atomic_json(
            recorder.path / "judgment.json", {"observed": measured, "passed": passed}
        )
        evidence = {
            str(path.relative_to(output)): file_digest(path)
            for path in recorder.path.iterdir()
            if path.is_file()
        }
        return passed, evidence

    def validation(self, comparison, output):
        cases = [
            uuid.uuid4().hex
            for _ in range(comparison["profile"]["tasks"]["counter"]["episodes"])
        ]
        atomic_json(output / "cases.json", cases)
        return dict(
            split="validation",
            comparison_sha256=digest(comparison),
            generated_at=now(),
            layouts={"counter": [digest(case) for case in cases]},
            admission_evidence={"cases.json": file_digest(output / "cases.json")},
        )

    def evaluate(self, candidate, comparison, cohort, output):
        rows = []
        for case in read_json(output / "cases.json"):
            passed, evidence = self.execute(
                candidate, case + "_" + candidate["id"], output
            )
            rows.append(
                dict(
                    task="counter",
                    layout_sha256=digest(case),
                    state="completed",
                    score=int(passed),
                    success=passed,
                    evidence_sha256=evidence,
                )
            )
        return dict(
            candidate_id=candidate["id"],
            kind="local_contract_evaluation",
            native_exit_code=0,
            comparison_sha256=digest(comparison),
            cohort_sha256=digest(cohort),
            freeze_sha256=verify_harness(candidate),
            evaluator_revision=PROTOCOL,
            episodes=rows,
        )


class Proposal:
    def __init__(self, evaluator):
        self.evaluator = evaluator

    def identity(self):
        return dict(kind="declared_local_memory_patch", implementation=implementation())

    def develop(self, parent, output):
        passed, evidence = self.evaluator.execute(parent, "development", output)
        return dict(
            split="evolve", passed=passed, evidence=evidence, costs={"model_calls": 2}
        )

    def propose(self, parent, feedback, output):
        store = MemoryStore(output / "candidates_memory")
        parent_memory = store.snapshot(
            read_json(Path(parent["root"]) / "memory_rules.json")
        )
        inbox = MemoryWorkspace(store, parent_memory, "proposal")
        draft = inbox.propose({"target": 2}, evidence=feedback["evidence"])
        candidate_memory = inbox.materialize(draft["id"])
        root = output / "candidate"
        shutil.copytree(parent["root"], root)
        atomic_json(root / "memory_rules.json", candidate_memory.read())
        return [
            manifest(
                root,
                "memory_candidate",
                parent_sha256=verify_harness(parent),
                method="development_memory_patch",
                changes={"memory_rules": draft["changes"]},
                environment=ENVIRONMENT,
                evidence=feedback["evidence"],
                costs={"proposals": 1},
            )
        ]


class Selector:
    def identity(self):
        return {"source": file_digest(Path(selection.__file__))}

    def select(self, *args, **kwargs):
        return selection.select_survivor(*args, **kwargs)


def run(workspace):
    root = Path(workspace).resolve()
    state = HarnessState(root / "state")
    if (state.root / "current.json").exists():
        current = state.resolve()
        return {
            "state": "retained",
            "harness": current["harness"]["id"],
            "qualification": None,
        }
    seed = root / "seed"
    seed.mkdir(parents=True, exist_ok=True)
    for kind in CLOSURE:
        value = {"kind": kind, "scope": SCOPE}
        if kind == "foundation":
            value = MODEL
        elif kind == "dependencies":
            value = implementation()
        elif kind == "memory_rules":
            value = {"target": 0}
        atomic_json(seed / (kind + ".json"), value)
    pool = ResourcePool({"cpu": 2})
    specs = [
        ServiceSpec(
            kind,
            meta,
            command=(
                sys.executable,
                "-m",
                "PhysicalRSI_demos.runtime_service",
                kind,
                "--journal",
                str(root / "journals" / kind / uuid.uuid4().hex),
            ),
            resources={"cpu": 1},
            sessions=kind == "model",
        )
        for kind, meta in [("environment", ENVIRONMENT), ("model", MODEL)]
    ]
    with Services(root / "services", pool=pool) as services:
        clients = services.start(specs)
        evaluator = Evaluation(
            EnvironmentClient(clients["environment"], expected=ENVIRONMENT),
            ModelClient(clients["model"], expected=MODEL),
        )
        loop = SelfHarness(
            root / "comparison",
            state=state,
            proposer=Proposal(evaluator),
            evaluator=evaluator,
            selector=Selector(),
            scope=SCOPE,
            profile={
                "tasks": {
                    "counter": {
                        "weight": 1,
                        "episodes": 3,
                        "score_range": [0, 1],
                        "maximum_regression": 0,
                    }
                },
                "minimum_gain": 0,
                "tie_tolerance": 1e-10,
            },
            protocol={
                "identity": PROTOCOL,
                "evaluation_kind": "local_contract_evaluation",
            },
        )
        state.initialize(manifest(seed, "parent"), policy=loop.config, scope=SCOPE)
        result = loop.run()
    return {
        "state": result["action"],
        "harness": result["harness"]["id"],
        "resources": pool.status(),
        "qualification": None,
        "scope": SCOPE,
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    print(json.dumps(run(parser.parse_args().workspace), indent=2))
