"""Bounded Self-Harness improvement with native MuJoCo evidence and lineage.

A deterministic System 2 can enable a reviewed feedback skill after observed
development failures. This is simulation integration, not model training,
open-ended invention, statistical robot evaluation or physical qualification.
"""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import random
import shutil
import uuid

from PhysicalRSI_core.embodiment import Revision
from PhysicalRSI_core.experiments import Budget
from PhysicalRSI_core.infra.devices import DeviceRegistry
from PhysicalRSI_core.infra.language import ModelConfig
from PhysicalRSI_core.infra.isolated_policy import IsolatedPolicy
from PhysicalRSI_core.infra.isolated_program import PythonIsolation
from PhysicalRSI_core.infra.isolated_proposal import IsolatedProposal
from PhysicalRSI_core.infra.proposal_model import ModelProposal
from PhysicalRSI_core.infra.storage import atomic_json, digest, file_digest, locked, read_json
from PhysicalRSI_core.infra.trial_quota import TrialQuota
from PhysicalRSI_core.lineage import HarnessState
from PhysicalRSI_core.self_harness import SelfHarness, selection
from PhysicalRSI_core.self_harness.artifacts import CLOSURE, verify_harness
from PhysicalRSI_core.self_harness.campaign import ImprovementCampaign
from PhysicalRSI_core.self_harness.evaluation import ExperimentEvaluator, freeze_json
from PhysicalRSI_core.self_harness.proposals import (
    EnumeratedEdits, JsonEditContract, ProposalLimits, StructuredProposer,
)
from . import mujoco_control
from . import isolated_candidates
from .mujoco_control import GAINS, SCOPE, TIMING, JointPolicy, JointVerifier, controller_environments


def implementation():
    import mujoco
    root = Path(__file__).resolve().parents[1]
    paths = [*root.joinpath("PhysicalRSI_core").rglob("*.py"),
             Path(__file__), Path(mujoco_control.__file__), Path(isolated_candidates.__file__)]
    return dict(mujoco=mujoco.__version__, sources={str(p.relative_to(root)): file_digest(p) for p in paths})


def components(isolation=None):
    controller = file_digest(Path(mujoco_control.__file__))
    return dict(
        foundation=dict(kind="reviewed-joint-controller", revision=controller, trainable=False),
        skill_selection=dict(route="joint-effort-control"),
        skills=dict(joint_effort_controller=controller),
        tools=dict(controller_gateway="PhysicalRSI_core.infra.controller"),
        control=dict(mode="zero-effort"), prompts=dict(enabled=False), memory_rules=dict(enabled=False),
        dependencies=implementation(), weights=dict(included=False),
        assets={model: file_digest(Path(__file__).with_name("mujoco_models") / (model + ".xml")) for model in GAINS},
        configuration=dict(models=list(GAINS), timing=asdict(TIMING), isolation=isolation.identity() if isolation else None))


def manifest(root, name, **metadata):
    return dict(id=name, root=str(root), components={
        kind: {kind + ".json": file_digest(root / (kind + ".json"))} for kind in CLOSURE}, **metadata)


class HarnessJointPolicy(JointPolicy):
    def __init__(self, candidate, model):
        self.candidate = candidate
        super().__init__(model, read_json(Path(candidate["root"]) / "control.json")["mode"])

    def identity(self):
        return dict(super().identity(), harness_revision=verify_harness(self.candidate))

    def describe(self):
        return replace(super().describe(), revision=verify_harness(self.candidate))


def case(model, initial, target):
    return dict(embodiment=model, initial=initial, target=target,
                position_tolerance=.02, velocity_tolerance=.05)


class JointSuite:
    def __init__(self, environments, *, isolation=None, worker_output=None):
        self.environments = environments
        self.isolation, self.worker_output = isolation, worker_output

    def identity(self):
        return dict(kind="mujoco-joint-repair/v1", implementation=implementation(),
                    isolation=self.isolation.identity() if self.isolation else None,
                    environments={task: env.identity() for task, env in self.environments.items()})

    def admit(self, candidate):
        expected = components(self.isolation)
        observed = {kind: read_json(Path(candidate["root"]) / (kind + ".json")) for kind in CLOSURE}
        mode = observed["control"].get("mode")
        expected["control"] = dict(mode=mode)
        accepted = observed == expected and mode in {"zero-effort", "feedback"}
        return dict(accepted=accepted, reason="Only the pinned reviewed controller modes are admitted",
                    evidence=dict(components_sha256=digest(observed), implementation=implementation()))

    def development_cases(self, parent):
        return {model: [case(model, 0., .3)] for model in GAINS}

    def validation_cases(self, comparison):
        # Actual initial states and targets vary after pool freeze. A random
        # identifier alone would not create a new physical test condition.
        rng = random.Random(int(digest(comparison), 16))
        return {model: [case(model, round(rng.uniform(-.1, .1), 6),
                            round(sign * rng.uniform(.22, .4), 6)) for sign in (-1, 1)]
                for model in GAINS}

    def environment(self, task):
        return self.environments[task]

    def policy(self, candidate, task):
        policy = HarnessJointPolicy(candidate, task)
        if self.isolation is None:
            return policy
        source = isolated_candidates.joint_policy(mode=policy.mode, gains=GAINS[task], period_seconds=TIMING.period_seconds)
        return IsolatedPolicy(source, specification=policy.describe(), isolation=self.isolation,
                              output=Path(self.worker_output) / uuid.uuid4().hex, max_actions=100)

    def verifier(self, task):
        return JointVerifier()


class FeedbackProposal:
    def __init__(self, evaluator):
        self.evaluator = evaluator

    def identity(self):
        return dict(kind="reviewed-feedback-repair", implementation=implementation())

    def develop(self, parent, output):
        return self.evaluator.development(parent, output)

    def resume_development(self, parent, output):
        return self.evaluator.development(parent, output)

    def propose(self, parent, feedback, output):
        outcomes = {row["outcome"] for row in feedback["episodes"]}
        if outcomes - {"success", "failure"}:
            raise ValueError("Undetermined development outcomes cannot justify this repair")
        mode = read_json(Path(parent["root"]) / "control.json")["mode"]
        if "failure" not in outcomes or mode == "feedback":
            return []
        root = output / "candidate-feedback"
        shutil.copytree(parent["root"], root)
        atomic_json(root / "control.json", dict(mode="feedback"))
        return [manifest(root, "feedback", parent_sha256=verify_harness(parent),
                         method="enable-reviewed-feedback-after-development-failure",
                         changes=dict(control=dict(before=mode, after="feedback")),
                         environment=self.evaluator.suite.identity(), evidence=feedback["evidence"],
                         costs=dict(proposals=1, training_steps=0))]


class Selector:
    def identity(self):
        return dict(source=file_digest(Path(selection.__file__)))

    def select(self, *args, **kwargs):
        return selection.select_survivor(*args, **kwargs)


def run(workspace, *, max_rounds=2, max_trials=12, proposal_strategy="reviewed",
        model_config=None, provider_revision=None, isolation_runtime=None):
    root = Path(workspace).resolve()
    isolation = PythonIsolation(isolation_runtime) if isolation_runtime is not None else None
    if proposal_strategy not in {"reviewed", "enumerated", "model", "isolated"}:
        raise ValueError("Unknown System 2 proposal strategy")
    if proposal_strategy == "model":
        if not isinstance(model_config, ModelConfig):
            raise ValueError("The model proposal strategy requires a ModelConfig")
        strategy = ModelProposal(model_config, provider_revision=provider_revision)
    elif model_config is not None or provider_revision is not None:
        raise ValueError("Model configuration requires the model proposal strategy")
    elif proposal_strategy == "isolated":
        if isolation is None:
            raise ValueError("Isolated System 2 requires an explicit isolation runtime")
        strategy = IsolatedProposal(isolated_candidates.PROPOSAL, isolation=isolation, output=root / "isolated-system2")
    else:
        strategy = EnumeratedEdits([{"control.json": dict(mode="feedback")}])
    contract = JsonEditContract({"control.json": dict(component="control", schema=dict(
        type="object", properties=dict(mode=dict(enum=["zero-effort", "feedback"])),
        required=["mode"], additionalProperties=False))})
    objective = ("Improve independent position and velocity success on both native MuJoCo fixtures. "
                 "The reviewed feedback mode enables the pinned proportional/derivative controller. "
                 "Only control.json may change; there is no training or physical qualification.")
    origin = (Revision("reviewed-feedback-repair", digest(implementation())) if proposal_strategy == "reviewed"
              else Revision("structured-" + proposal_strategy, digest(dict(strategy=strategy.identity(),
                            contract=contract.public(), objective=objective))))
    with locked(root / ".evolution.lock"):
        for kind, value in components(isolation).items():
            freeze_json(root / "seed" / (kind + ".json"), value)
        initial = manifest(root / "seed", "zero-effort")
        state = HarnessState(root / "state")
        quota = TrialQuota(root / "quota", max_trials=max_trials)
        with controller_environments(root) as environments:
            evaluator = ExperimentEvaluator(
                suite=JointSuite(environments, isolation=isolation, worker_output=root / "isolated-system1"),
                budgets={model: Budget(100, 30) for model in GAINS},
                scope=SCOPE, system2=origin,
                quota=quota, device_registry=DeviceRegistry(root / "devices"))
            proposer = (FeedbackProposal(evaluator) if proposal_strategy == "reviewed" else StructuredProposer(
                evaluator=evaluator, strategy=strategy, contract=contract, objective=objective,
                limits=ProposalLimits(max_candidates=1)))

            def build_loop(directory):
                return SelfHarness(
                    directory, state=state, proposer=proposer, evaluator=evaluator,
                    selector=Selector(), scope=SCOPE, max_candidates=1,
                    profile=dict(tasks={model: dict(weight=1, episodes=2, score_range=[0, 1],
                                 maximum_regression=0) for model in GAINS}, minimum_gain=0, tie_tolerance=1e-10),
                    protocol=dict(identity=evaluator.revision, evaluation_kind="simulation_evaluation"))

            campaign = ImprovementCampaign(root / "campaign", state=state, build_loop=build_loop,
                                           max_rounds=max_rounds)
            result = campaign.run(initial)
        current = result["current"]
        report = dict(status=result["status"], completed_rounds=result["completed_rounds"],
                      rounds=[{k: v for k, v in row.items() if k != "files"} for row in result["rounds"]],
                      harness=current["harness"]["id"], revision=current["revision"],
                      selection=current.get("decision"), quota=quota.status(), scope=SCOPE, qualification=None,
                      lineage_committed=current["action"] == "selection", stop=result["stop"],
                      proposal_strategy=proposal_strategy, system1_isolation=isolation.identity() if isolation else None)
        atomic_json(root / "result.json", report)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--proposer", choices=("reviewed", "enumerated", "model", "isolated"), default="reviewed")
    parser.add_argument("--isolation-runtime", type=Path, help="Pinned stdlib runtime for isolated System 1 and optional isolated System 2")
    parser.add_argument("--model-config", help="JSON ModelConfig; credentials stay in the named environment variable")
    parser.add_argument("--provider-revision", help="Declared external model/provider revision, not verified weights")
    args = parser.parse_args()
    config = ModelConfig(**read_json(args.model_config)) if args.model_config else None
    print(json.dumps(run(args.workspace, max_rounds=args.rounds, max_trials=args.trials,
                         proposal_strategy=args.proposer, model_config=config,
                         provider_revision=args.provider_revision, isolation_runtime=args.isolation_runtime), indent=2))


if __name__ == "__main__":
    main()
