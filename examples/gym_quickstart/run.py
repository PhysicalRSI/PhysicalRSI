"""Improve a small CartPole controller with the PhysicalRSI Self-Harness."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from physicalrsi.environments.gymnasium import (  # noqa: E402
    CallablePolicy,
    GymnasiumEnvironment,
    GymnasiumVerifier,
)
from PhysicalRSI_core.contracts import Contract  # noqa: E402
from PhysicalRSI_core.embodiment import Revision  # noqa: E402
from PhysicalRSI_core.infra.storage import atomic_json, digest, file_digest, read_json  # noqa: E402
from PhysicalRSI_core.infra.trial_quota import TrialQuota  # noqa: E402
from PhysicalRSI_core.lineage import HarnessState  # noqa: E402
from PhysicalRSI_core.self_harness import SelfHarness, selection  # noqa: E402
from PhysicalRSI_core.self_harness.artifacts import CLOSURE, verify_harness  # noqa: E402
from PhysicalRSI_core.self_harness.campaign import ImprovementCampaign  # noqa: E402
from PhysicalRSI_core.self_harness.evaluation import ExperimentEvaluator  # noqa: E402
from PhysicalRSI_core.self_harness.rollout import RolloutProposer  # noqa: E402
from PhysicalRSI_core.self_harness.search_space import ParameterSpace, SearchField  # noqa: E402


OBSERVATION = Contract(
    "cartpole-observation", embodiment="cartpole-v1", shape=(4,), dtype="float32"
)
ACTION = Contract(
    "cartpole-action", embodiment="cartpole-v1", shape=(), dtype="int64"
)


def make_cartpole():
    import gymnasium as gym

    return gym.make("CartPole-v1")


def controller(params, observation):
    x, x_dot, angle, angle_dot = observation
    score = (
        params["angle_gain"] * angle
        + params["angular_velocity_gain"] * angle_dot
        + params["position_gain"] * x
        + params["velocity_gain"] * x_dot
        + params["bias"]
    )
    return 1 if score >= 0 else 0


CART_LIMIT = 0.3  # metres; tighter than Gymnasium's 2.4 so cart feedback matters


def solved(case, trace):
    """Balance for target_steps while keeping the cart within CART_LIMIT."""
    if len(trace) - 1 < case["target_steps"]:
        return False
    return all(abs(row["observation"][0]) <= CART_LIMIT for row in trace)


def measurements(case, trace):
    return {"steps": len(trace) - 1}


def manifest(root, name):
    return {
        "id": name,
        "root": str(root),
        "components": {
            kind: {f"{kind}.json": file_digest(root / f"{kind}.json")}
            for kind in CLOSURE
        },
    }


class CartPoleSuite:
    def identity(self):
        return {"kind": "gym-cartpole-quickstart/v1", "implementation": file_digest(Path(__file__))}

    def admit(self, candidate):
        return {"accepted": True, "evidence": {"harness_sha256": verify_harness(candidate)}}

    def development_cases(self, parent):
        # Fresh development seeds per parent, so retained rounds do not replay the same cases.
        base = 10_000 + int(parent["components"]["control"]["control.json"][:8], 16) % 1_000_000
        return {"cartpole": [{"seed": base + i, "target_steps": 500} for i in range(2)]}

    def validation_cases(self, comparison):
        # The core forbids reusing validation inputs across rounds: derive a fresh,
        # reproducible block of seeds from the frozen comparison itself.
        import hashlib, json
        digest = hashlib.sha256(json.dumps(comparison, sort_keys=True).encode()).hexdigest()
        base = 2_000_000 + int(digest[:12], 16) % 1_000_000_000
        return {"cartpole": [{"seed": base + i, "target_steps": 500} for i in range(24)]}

    def claim_test(self, *, candidate_sha256, selection_sha256, claim_root):
        return {"cartpole": [{"seed": 1000 + i, "target_steps": 500} for i in range(8)]}

    def environment(self, task):
        return GymnasiumEnvironment(
            make_cartpole,
            observation_contract=OBSERVATION,
            action_contract=ACTION,
            embodiment_name="cartpole-v1",
            mode="simulation",
            max_steps=500,
        )

    def policy(self, candidate, task):
        return CallablePolicy(
            candidate,
            controller,
            observation_contract=OBSERVATION,
            action_contract=ACTION,
            name="cartpole-linear-controller",
        )

    def verifier(self, task):
        return GymnasiumVerifier(solved, measurements)


class GaussianEvolver:
    """Small bounded Gaussian proposer over the declared SearchFields."""

    def __init__(self, space, *, seed=7, population=2):
        self.space = space
        self.seed = seed
        self.population = population

    def identity(self):
        return {
            "kind": "gaussian-search/v1",
            "space": self.space.identity(),
            "seed": self.seed,
            "population": self.population,
        }

    def propose(self, parent, evaluations, context, output):
        inspected = self.space.inspect(parent)
        evidence = {}
        for evaluation in evaluations:
            evidence.update(evaluation["evidence"])
        # Mix in the round directory so a retained parent still gets new proposals.
        seed_material = f"{self.seed}:{verify_harness(parent)}:{Path(output).name}".encode()
        rng = random.Random(int(hashlib.sha256(seed_material).hexdigest()[:16], 16))
        values_seen = {tuple(inspected["values"])}
        proposals = []
        for index in range(self.population):
            for _attempt in range(128):
                units = [
                    (value - field.lower) / (field.upper - field.lower)
                    for value, field in zip(inspected["values"], self.space.fields)
                ]
                candidate_units = [rng.gauss(unit, 0.1) for unit in units]
                values = tuple(self.space.quantize(candidate_units))
                if values not in values_seen:
                    values_seen.add(values)
                    proposals.append(values)
                    break
        report = {
            "schema": "physicalrsi.example-gaussian/v1",
            "parent_sha256": verify_harness(parent),
            "proposals": [list(values) for values in proposals],
        }
        report_path = Path(output) / "gaussian-proposals.json"
        atomic_json(report_path, report)
        evidence[str(report_path.relative_to(output))] = file_digest(report_path)
        children = []
        for index, values in enumerate(proposals):
            name = f"gaussian-{digest([verify_harness(parent), values])[:12]}-{index}"
            candidate, changes = self.space.candidate(
                parent,
                inspected,
                values,
                output=Path(output) / name,
                name=name,
            )
            children.append(
                dict(
                    candidate,
                    parent_sha256=verify_harness(parent),
                    method="gaussian-search",
                    changes=changes,
                    environment="CartPole-v1",
                    evidence=evidence,
                    costs={"proposals": 1},
                    hypothesis="Gaussian perturbations will improve pole survival",
                    expected_effect="Increase held-out CartPole survival",
                    required_checks=["fresh-paired-evaluation", "held-out-survival"],
                )
            )
        return children


class Selector:
    def identity(self):
        return {"source": file_digest(Path(selection.__file__))}

    def select(self, *args, **kwargs):
        return selection.select_survivor(*args, **kwargs)


def build_initial(root):
    seed = Path(root) / "initial"
    seed.mkdir(parents=True, exist_ok=True)
    controls = {
        # Balances the pole with weak cart feedback: it keeps the cart within
        # bounds in about 60 percent of episodes. Stronger cart feedback
        # (about 0.2 and 0.3) reaches 100 percent, one or two search steps away.
        "angle_gain": 4.0,
        "angular_velocity_gain": 0.8,
        "position_gain": 0.1,
        "velocity_gain": 0.05,
        "bias": 0.0,
    }
    for kind in CLOSURE:
        payload = controls if kind == "control" else {"kind": kind}
        atomic_json(seed / f"{kind}.json", payload)
    return manifest(seed, "parent")


def run(root):
    root = Path(root).resolve()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    initial = build_initial(root)
    state = HarnessState(root / "state")
    quota = TrialQuota(root / "quota", max_trials=1000)
    suite = CartPoleSuite()
    evaluator = ExperimentEvaluator(
        suite=suite,
        budgets={"cartpole": __import__("PhysicalRSI_core.experiments", fromlist=["Budget"]).Budget(500, 30)},
        scope="simulation",
        system2=Revision("gym-quickstart", "1"),
        quota=quota,
    )
    space = ParameterSpace(
        [
            SearchField("control", "control.json", ("angle_gain",), 0.0, 8.0, 2),
            SearchField("control", "control.json", ("angular_velocity_gain",), 0.0, 2.0, 2),
            SearchField("control", "control.json", ("position_gain",), -1.0, 1.0, 2),
            SearchField("control", "control.json", ("velocity_gain",), -1.0, 1.0, 2),
        ]
    )
    evolver = GaussianEvolver(space, population=4)

    def build_loop(directory):
        return SelfHarness(
            directory,
            state=state,
            proposer=RolloutProposer(evaluator, evolver),
            evaluator=evaluator,
            selector=Selector(),
            scope="simulation",
            max_candidates=4,
            profile={
                "tasks": {"cartpole": {"weight": 1, "episodes": 24, "score_range": [0, 1], "maximum_regression": 0}},
                "minimum_gain": 0,
                "tie_tolerance": 0,
                "statistics": {"method": "paired-permutation", "alpha": 0.05, "max_comparisons": 24, "resamples": 4000},
            },
            protocol={"identity": evaluator.revision, "evaluation_kind": "local_contract_evaluation"},
        )

    result = ImprovementCampaign(root / "campaign", state=state, build_loop=build_loop, max_rounds=6).run(initial)
    for index, record in enumerate(result["rounds"], 1):
        incumbent = state.read(record["result_revision"])["harness"]["id"]
        print(f"round {index}: incumbent={incumbent} outcome={record['outcome']}")
    current = result["current"]["harness"]
    report = evaluator.report_test(current, selection_sha256=result["current"]["revision"], output=root / "heldout")
    metric = report["metrics"]["cartpole"]
    print(f"held-out success: {metric['successes']}/{metric['planned']} ({metric['success_rate']:.3f})")
    return result, report


if __name__ == "__main__":
    run(Path("/tmp/physicalrsi-gym-quickstart"))
