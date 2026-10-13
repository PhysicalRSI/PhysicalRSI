"""Retained comparisons must preserve historical capability measurements.

An authored diagnostic tie precedes an observed vector calibration deliberately
applied to both action contracts. The global repair improves the paired score
but loses the old axis behavior. Post-selection replays of known cases verify
that loss; they are not fresh tests and never enter candidate proposals.
"""
from pathlib import Path
import shutil

from PhysicalRSI_core.embodiment import Revision
from PhysicalRSI_core.experiments import Budget
from PhysicalRSI_core.infra.storage import atomic_json, file_digest, read_json
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
from .harness_tree import Reflector
from .recursive_research import manifest
from .rollout_evolution import Policy, Suite as CalibrationSuite, fit_coefficients


def identity(kind):
    return dict(kind=kind, implementation=file_digest(Path(__file__)))


class LimitedPolicy(Policy):
    def __init__(self, task, revision, coefficient, limit):
        super().__init__(task, revision, coefficient)
        self.limit = limit

    def identity(self):
        return dict(super().identity(), limit=self.limit, controller=identity("limited-calibration/v1"))

    def act(self, observation, context):
        action = super().act(observation, context)
        return max(-self.limit, min(self.limit, action)) if self.task == "axis" else action


class Suite(CalibrationSuite):
    def __init__(self):
        super().__init__({"axis": 1., "vector": .5})

    def identity(self):
        return dict(super().identity(), limited_control=identity("retained-observation-suite/v1"))

    def policy(self, candidate, task):
        control = read_json(Path(candidate["root"]) / "control.json")
        return LimitedPolicy(task, verify_harness(candidate), control[task], control["axis_limit"])


class Evolver:
    def identity(self):
        return identity("diagnostic-tie-then-observed-calibration/v1")

    def propose(self, parent, evaluations, context, output):
        root = output / "candidate"
        shutil.copytree(parent["root"], root)
        if context["index"]:
            coefficients = fit_coefficients(evaluations)
            control = read_json(root / "control.json")
            control.update(coefficients)
            # Authored forgetting control: apply the observed vector response
            # globally and expand the command bound by one unit.
            control["axis"] = coefficients["vector"]
            control["axis_limit"] += 1.
            atomic_json(root / "control.json", control)
        atomic_json(root / "prompts.json", dict(diagnostic_round=context["index"]))
        return [manifest(root, "candidate-" + str(context["index"]),
            parent_sha256=verify_harness(parent), method="diagnostic-then-observed-calibration",
            changes=dict(control=bool(context["index"]), diagnostic=True), environment="software-calibration",
            evidence={p: sha for e in evaluations for p, sha in e["evidence"].items()}, costs=dict(proposals=1),
            used_lessons=[], hypothesis="Test a globally applied response estimate with an expanded action range",
            expected_effect="Improve vector response; expose forgetting through historical retention checks",
            required_checks=["paired-success", "historical-retention"])]


def setup(root):
    root = Path(root).resolve()
    for kind in CLOSURE:
        freeze_json(root / "seed" / (kind + ".json"),
            dict(axis=1., vector=1., axis_limit=3.) if kind == "control" else dict(kind=kind))
    initial = manifest(root / "seed", "initial")
    state, suite = HarnessState(root / "state"), Suite()
    archive, quota = CandidateArchive(root / "archive"), TrialQuota(root / "quota", max_trials=24)
    memory = ResearchMemory(root / "memory")
    revision = memory.write({"retention": dict(conditions={"study": "retained-observations"},
        hypothesis="Keep measured capability floors even when no candidate is promoted",
        required_checks=["historical-retention"], counterexample_checks=[],
        evidence={str(root / "seed/control.json"): file_digest(root / "seed/control.json")})})

    def build(directory, context):
        index = context["index"]
        def cases(axis, vector):
            return dict(axis=[dict(initial=[0.], target=[x]) for x in axis],
                        vector=[dict(initial=[0., 0.], target=[x, -2*x]) for x in vector])
        protocol = CaseProtocol(directory / "cases", splits=dict(
            development=cases([11+index], [31+index]),
            validation=cases([1+2*index, 2+2*index], [10+10*index, 12+10*index]),
            test=cases([101], [101])), identity_keys=("initial", "target"))
        evaluator = ExperimentEvaluator(suite=PreregisteredSuite(suite, protocol),
            budgets={task: Budget(1, 30) for task in suite.gains}, scope="software",
            system2=Revision("retained-observation-research", "1"), quota=quota)
        return SelfHarness(directory, state=state,
            proposer=RolloutProposer(evaluator, Evolver(), archive=archive), evaluator=evaluator,
            selector=RetentionSelector(state=state, parent_revision=context["parent_revision"],
                                       maximum_regression={task: 0. for task in suite.gains}),
            profile=dict(tasks={task: dict(weight=1, episodes=2, score_range=[0, 1], maximum_regression=1)
                                for task in suite.gains}, minimum_gain=0, tie_tolerance=1e-10),
            protocol=dict(identity=evaluator.revision, evaluation_kind="local_contract_evaluation"),
            scope="software", max_candidates=1, round_context=context, archive=archive,
            evolution=EvolutionSpec(("control", "prompts")))

    campaign = ResearchCampaign(root / "campaign", state=state, build_loop=build, memory=memory,
        memory_revision=revision, context={"study": "retained-observations"},
        reflector=Reflector(), max_rounds=2)
    return campaign, initial, suite, quota


def replay_known_cases(campaign, result, suite, quota):
    """Diagnose a completed study on known validation cases, without selection."""
    if result["status"] != "completed" or campaign.state.resolve() != result["current"]:
        raise ValueError("Known-case replay requires the unchanged completed study")
    first = campaign.root / "rounds/round-0001/execution"
    case_file = first / "validation-cases.json"
    cases = read_json(case_file)["axis"]
    parent = read_json(first / "parent.json")["harness"]
    child = read_json(campaign.root / "rounds/round-0002/execution/propose.json")["output"][0]

    class ReplaySuite:
        def identity(self):
            return dict(study=suite.identity(), known_cases=cases,
                        source_cases_sha256=file_digest(case_file), role="known-case-diagnostic")
        def development_cases(self, candidate):
            return {"axis": cases}
        def environment(self, task):
            return suite.environment(task)
        def policy(self, candidate, task):
            return suite.policy(candidate, task)
        def verifier(self, task):
            return suite.verifier(task)

    evaluator = ExperimentEvaluator(suite=ReplaySuite(), budgets={"axis": Budget(1, 30)},
        scope="software", system2=Revision("known-case-retention-replay", "1"), quota=quota)
    replay = {}
    for candidate in (parent, child):
        evidence = evaluator.development(candidate, campaign.root.parent / "known-case-replay")
        replay[candidate["id"]] = dict(successes=sum(e["outcome"] == "success" for e in evidence["episodes"]),
                                      planned=len(cases), evidence=evidence["evidence"])
    if campaign.state.resolve() != result["current"]:
        raise ValueError("Known-case diagnostics changed the selected revision")
    return dict(role="post-selection known-case diagnostic; not fresh held-out tests", candidates=replay,
                source_cases=str(case_file), source_cases_sha256=file_digest(case_file), qualification=None)


def run(root):
    campaign, initial, suite, quota = setup(root)
    result = campaign.run(initial)
    replay = replay_known_cases(campaign, result, suite, quota)
    return dict(research=result, known_case_replay=replay, quota=quota.status(), scope="software", qualification=None)
