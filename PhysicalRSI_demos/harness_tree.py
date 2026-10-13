"""Grow scoped control branches without forgetting previously solved tasks.

The calibration provider sees development trajectories only. A deliberately
global repair improves the weighted score but loses older capabilities; the
historical retention selector rejects it. This is a CPU integration study.
"""
from pathlib import Path
import shutil

from PhysicalRSI_core import harness_tree as tree_module
from PhysicalRSI_core.embodiment import Revision
from PhysicalRSI_core.experiments import Budget
from PhysicalRSI_core.harness_tree import HarnessTree, RoutedPolicy
from PhysicalRSI_core.infra.storage import atomic_json, digest, file_digest, read_json
from PhysicalRSI_core.infra.trial_quota import TrialQuota
from PhysicalRSI_core.lineage import HarnessState
from PhysicalRSI_core.self_harness import SelfHarness
from PhysicalRSI_core.self_harness.archive import CandidateArchive
from PhysicalRSI_core.self_harness.artifacts import CLOSURE, EvolutionSpec, verify_harness
from PhysicalRSI_core.self_harness.evaluation import ExperimentEvaluator, freeze_json
from PhysicalRSI_core.self_harness.protocol import CaseProtocol, PreregisteredSuite
from PhysicalRSI_core.self_harness.research_campaign import ResearchCampaign
from PhysicalRSI_core.self_harness.research_memory import ResearchMemory
from PhysicalRSI_core.self_harness.retention import RetentionSelector
from PhysicalRSI_core.self_harness.rollout import RolloutProposer
from .recursive_research import manifest
from .rollout_evolution import Environment, Policy, Verifier


TASKS = {"axis-default": ("axis", "standard", 1.), "vector-default": ("vector", "standard", 1.),
         "axis-shifted": ("axis", "shifted", 2.), "vector-shifted": ("vector", "shifted", .5)}
KEYS = ["action_contract", "drive", "task"]
EVOLUTION = EvolutionSpec(("control", "skill_selection"))


def identity(kind):
    return dict(kind=kind, implementation=file_digest(Path(__file__)))


class Backend:
    def identity(self):
        return identity("tree-calibration-policy-backend/v1")

    def build(self, harness, task):
        return Policy(TASKS[task][0], verify_harness(harness), read_json(Path(harness["root"]) / "control.json")["coefficient"])


class PublicCapabilities:
    def identity(self):
        return identity("public-actuator-capabilities/v1")

    def extract(self, observation):
        return {key: observation[key] for key in ("drive", "action_contract")}


class Suite:
    def __init__(self, workspace):
        self.workspace, self.resets = Path(workspace), []

    def identity(self):
        return dict(identity("tree-calibration-suite/v1"), tasks={k:list(v) for k,v in TASKS.items()},
                    workspace=str(self.workspace), tree=file_digest(Path(tree_module.__file__)))

    def admit(self, candidate):
        tree = HarnessTree(candidate)
        for task in TASKS:
            self.policy(candidate, task).identity()
        return dict(accepted=True, evidence=dict(tree=tree.revision, routing=digest(tree.body)))

    def environment(self, task):
        contract, _, gain = TASKS[task]

        class PublicEnvironment(Environment):
            def reset(self, case, context):
                result = super().reset(case, context)
                result["observation"].update(drive=case["drive"], action_contract=contract)
                return result

        return PublicEnvironment(contract, gain, self.resets)

    def policy(self, candidate, task):
        return RoutedPolicy(candidate, task=task, backend=Backend(), extractor=PublicCapabilities(), workspace=self.workspace)

    def verifier(self, task):
        return Verifier()


class Evolver:
    def identity(self):
        return identity("observed-specialization-tree-evolver/v1")

    def propose(self, parent, evaluations, context, output):
        failed = sorted((v for v in evaluations if v["outcome"] == "failure"), key=lambda v: v["trial"]["task"])
        if not failed:
            return []
        value = failed[0]
        numerator = denominator = 0.
        for previous, current in zip(value["trajectory"], value["trajectory"][1:]):
            action = current["action"]
            commands = action["joint_delta"] if isinstance(action, dict) else [action]
            for p, q, a in zip(previous["observation"]["position"], current["observation"]["position"], commands):
                numerator += a * (q-p)
                denominator += a*a
        if numerator == 0 or denominator == 0:
            return []
        coefficient = denominator/numerator
        tree = HarnessTree(parent)
        nodes = {node: dict(parent=row["parent"], when=row["when"],
                           harness=tree.materialize(node, output / "branch-inputs"))
                 for node, row in tree.body["nodes"].items()}
        source = nodes[tree.body["root"]]["harness"]
        draft = output / "fitted-branch"
        shutil.copytree(source["root"], draft)
        atomic_json(draft / "control.json", dict(coefficient=coefficient))
        fitted = manifest(draft, "fitted-branch")
        capabilities = PublicCapabilities().extract(value["trajectory"][0]["observation"])
        when = {"drive": capabilities["drive"]}
        parents = [name for name, row in nodes.items() if row["when"] == when]
        parent_node = parents[0] if parents else tree.body["root"]
        if parents:
            when["action_contract"] = capabilities["action_contract"]
        node = "specialist-" + str(len(nodes))
        nodes[node] = dict(parent=parent_node, when=when, harness=fitted)
        child = HarnessTree.create(nodes, keys=KEYS, output=output / "scoped-tree", name="scoped-" + node, evolution=EVOLUTION)
        children = [child]
        if parents:
            # A measured response applied globally can improve the heavily
            # weighted new task while destroying previously solved tasks.
            children.append(HarnessTree.create({"fallback":dict(parent=None, when={}, harness=fitted)},
                keys=KEYS, output=output / "global-tree", name="global-repair", evolution=EVOLUTION))
        for child in children:
            child.update(parent_sha256=verify_harness(parent), method="observed-response-specialization",
                changes=dict(control=coefficient, routing=child["id"]), environment="software-calibration",
                evidence={p: sha for e in evaluations for p, sha in e["evidence"].items()}, costs=dict(proposals=1),
                used_lessons=[], hypothesis="Specialize the failing public capability without changing its ancestors",
                expected_effect="Solve new regimes while retaining previously measured capability scores",
                required_checks=["paired-success", "historical-retention"])
        return children


class Reflector:
    def identity(self):
        return identity("retain-tree-research-memory/v1")

    def reflect(self, context, development, output):
        return dict(schema="physicalrsi.memory-delta/v1", operations=[])


def cases(targets, *, test=False):
    result = {}
    for task, (contract, drive, _) in TASKS.items():
        result[task] = [dict(initial=[0.] * (1 if contract == "axis" else 2),
                            target=[v] if contract == "axis" else [v, -2*v],
                            drive="unseen-standard" if test and drive == "standard" else drive) for v in targets]
    return result


def setup(root, *, max_trials=60):
    root = Path(root).resolve()
    freeze_json(root / "config.json", dict(implementation=identity("harness-tree-acceptance/v1"), max_trials=max_trials))
    for kind in CLOSURE:
        freeze_json(root / "seed" / (kind + ".json"), dict(coefficient=1.) if kind == "control" else dict(kind=kind))
    source = manifest(root / "seed", "fallback")
    initial = HarnessTree.create({"fallback":dict(parent=None, when={}, harness=source)},
        keys=KEYS, output=root / "initial-tree", name="initial-tree", evolution=EVOLUTION)
    state, suite, archive = HarnessState(root / "state"), Suite(root / "leaf-cache"), CandidateArchive(root / "archive")
    memory = ResearchMemory(root / "memory")
    memory_revision = memory.write({"retention":dict(conditions={"study":"capability-tree"},
        hypothesis="Grow scoped branches without discarding earlier capabilities", required_checks=["historical-retention"],
        counterexample_checks=[], evidence={str(root / "seed/control.json"):file_digest(root / "seed/control.json")})})
    quota = TrialQuota(root / "quota", max_trials=max_trials)

    def build(directory, context):
        index = context["index"]
        protocol = CaseProtocol(directory / "cases", splits=dict(development=cases([1+index]),
            validation=cases([10+10*index,12+10*index]), test=cases([100,101],test=True)),
            identity_keys=("initial","target","drive"))
        evaluator = ExperimentEvaluator(suite=PreregisteredSuite(suite,protocol),
            budgets={task:Budget(1,30) for task in TASKS}, scope="software",
            system2=Revision("capability-tree-research","1"), quota=quota)
        selector = RetentionSelector(state=state, parent_revision=context["parent_revision"],
                                     maximum_regression={task:0. for task in TASKS})
        return SelfHarness(directory, state=state, evaluator=evaluator,
            proposer=RolloutProposer(evaluator,Evolver(),archive=archive), selector=selector,
            profile=dict(tasks={task:dict(weight=5 if task=="vector-shifted" else 1, episodes=2,
                score_range=[0,1], maximum_regression=1) for task in TASKS}, minimum_gain=0,tie_tolerance=1e-10),
            protocol=dict(identity=evaluator.revision,evaluation_kind="local_contract_evaluation"),
            scope="software", max_candidates=2, round_context=context, evolution=EVOLUTION, archive=archive)

    campaign = ResearchCampaign(root / "campaign",state=state,build_loop=build,memory=memory,memory_revision=memory_revision,
        context={"study":"capability-tree"},reflector=Reflector(),max_rounds=3)
    return campaign, initial, suite, quota


def run(root):
    campaign, initial, suite, quota = setup(root)
    research = campaign.run(initial)
    return dict(schema="physicalrsi.harness-tree-acceptance/v1", research=research,
                final_test=campaign.report_test(initial), quota=quota.status(), scope="software", qualification=None)
