"""Recursive native Isaac acceptance using shared core contracts.

The default evolver is the existing bounded public-clearance repair control.
Pass an independent RolloutEvolver to study model-driven proposals. Neither
this control nor a successful run establishes general engineering ability.
"""
from pathlib import Path
import shutil

from PhysicalRSI_core.embodiment import Revision
from PhysicalRSI_core.experiments import Budget
from PhysicalRSI_core.infra.devices import DeviceRegistry
from PhysicalRSI_core.infra.scheduler import JobScheduler
from PhysicalRSI_core.infra.storage import atomic_json, file_digest, read_json
from PhysicalRSI_core.infra.trial_quota import TrialQuota
from PhysicalRSI_core.lineage import HarnessState
from PhysicalRSI_core.self_harness import SelfHarness
from PhysicalRSI_core.self_harness.archive import CandidateArchive
from PhysicalRSI_core.self_harness.artifacts import EvolutionSpec, verify_harness
from PhysicalRSI_core.self_harness.evaluation import ExperimentEvaluator
from PhysicalRSI_core.self_harness.fidelity import MultiFidelityEvaluator
from PhysicalRSI_core.self_harness.protocol import CaseProtocol, PreregisteredSuite
from PhysicalRSI_core.self_harness.parameter_search import ScalarParameter, ScalarRefinementEvolver
from PhysicalRSI_core.self_harness.research_campaign import ResearchCampaign
from PhysicalRSI_core.self_harness.research_memory import ResearchMemory
from PhysicalRSI_core.self_harness.rollout import RolloutProposer
from .harness import Selector, create_seed, implementation, manifest
from .policy import propose_repair
from .task import TASK, SCOPE
from .trials import Suite
from .feedback import ClearanceFeedback


class ClearanceEvolver:
    def identity(self):
        return dict(kind="bounded-public-clearance-control/v1", implementation=implementation())

    def propose(self, parent, evaluations, context, output):
        memory = read_json(Path(parent["root"]) / "memory_rules.json")
        repairs = []
        for value in evaluations:
            if value["outcome"] not in {"success", "failure"}:
                continue
            public = value["trajectory"][0]["observation"]["public_observation"]
            repair = propose_repair(dict(success=value["outcome"] == "success", public_observation=public), memory)
            if repair is not None:
                repairs.append(repair)
        if not repairs:
            return []
        repair = max(repairs, key=lambda value: value["transit_height_m"])
        root = output / "clearance-candidate"
        shutil.copytree(parent["root"], root)
        atomic_json(root / "memory_rules.json", repair)
        return [manifest(root, "clearance-" + str(context["index"]), parent_sha256=verify_harness(parent),
            method="bounded-public-clearance-control", changes=dict(memory_rules=repair),
            environment=dict(backend="isaacsim-5.1"),
            evidence={path: sha for value in evaluations for path, sha in value["evidence"].items()},
            costs=dict(proposals=1), used_lessons=[],
            hypothesis="Raise transit above the observed obstruction plus the declared sample clearance",
            expected_effect="Improve independently checked sample transfer on fresh layouts",
            required_checks=["native-transfer-and-disturbance"])]


class ClearancePoolEvolver(ClearanceEvolver):
    """Native screening study: one observed repair and two declared controls.

    Low-clearance controls test whether actual development rollouts exclude
    ineffective candidates. They are not learned proposals or a search result.
    """
    def identity(self):
        return dict(kind="native-clearance-screening-control/v1", implementation=implementation(),
                    control_heights=[.10, .20], candidate_roles=["observed-repair", "negative-control", "negative-control"])

    def propose(self, parent, evaluations, context, output):
        children = super().propose(parent, evaluations, context, output)
        if not children:
            return []
        for index, height in enumerate((.10, .20)):
            memory = read_json(Path(parent["root"]) / "memory_rules.json")
            memory["transit_height_m"] = height
            root = output / ("low-clearance-control-" + str(index))
            shutil.copytree(parent["root"], root)
            atomic_json(root / "memory_rules.json", memory)
            children.append(manifest(root,"low-clearance-" + str(context["index"]) + "-" + str(index),
                parent_sha256=verify_harness(parent), method="declared-negative-clearance-control",
                changes=dict(memory_rules=memory), environment=dict(backend="isaacsim-5.1"),
                evidence={path:sha for value in evaluations for path,sha in value["evidence"].items()},
                costs=dict(proposals=1),used_lessons=[],
                hypothesis="A low carrying height should fail measured transfer and disturbance checks",
                expected_effect="Test rejection by development screening before full validation",
                required_checks=["native-transfer-and-disturbance"]))
        return children


class Reflector:
    def identity(self):
        return dict(kind="native-clearance-reflector/v1", implementation=implementation())

    def reflect(self, context, development, output):
        operations = []
        if "observed-clearance" not in context["lessons"] and any(
                row["outcome"] == "failure" for row in development["episodes"]):
            operations.append(dict(kind="add", lesson_id="observed-clearance", lesson=dict(
                conditions=context["scope"],
                hypothesis="Test carrying clearance using public RGB-D after a failed transfer",
                required_checks=["native-transfer-and-disturbance"], counterexample_checks=[],
                evidence=development["evidence"])))
        return dict(schema="physicalrsi.memory-delta/v1", operations=operations)


def setup(workspace, python, *, episodes=3, rounds=2, gui=False, device_registry=None,
          resource="isaac-gpu-0", evolver=None, staged=False, screen_incumbent=False, feedback_search=False):
    if type(episodes) is not int or not 1 <= episodes <= 50 or type(rounds) is not int or not 1 <= rounds <= 20:
        raise ValueError("Use 1 to 50 validation cases and 1 to 20 rounds")
    if type(feedback_search) is not bool or (feedback_search and evolver is not None):
        raise ValueError("Feedback search declares its own bounded evolver")
    if type(staged) is not bool or type(screen_incumbent) is not bool:
        raise ValueError("Screening options must be boolean")
    staged = staged or feedback_search
    if screen_incumbent and not staged:
        raise ValueError("Declare staged screening before requesting incumbent screening")
    root = Path(workspace).resolve()
    seed = root / "seed"
    initial = manifest(seed, "initial") if seed.exists() else create_seed(seed)
    state, memory = HarnessState(root / "state"), ResearchMemory(root / "memory")
    archive = CandidateArchive(root / "archive")
    memory_revision = memory.write({"task-checks": dict(conditions=dict(task=TASK),
        hypothesis="Require release, stable target placement and undisturbed surrounding objects",
        required_checks=["native-transfer-and-disturbance"], counterexample_checks=[],
        evidence={str(seed / "source" / "task.py"): file_digest(seed / "source" / "task.py")})})
    max_candidates = 3 if staged and not feedback_search else 1
    screening_trials = max_candidates + int(screen_incumbent) if staged else 0
    quota = TrialQuota(root / "quota", max_trials=rounds * (1 + screening_trials + (1+max_candidates)*episodes) + episodes)
    registry = DeviceRegistry(device_registry or root / "devices")
    if feedback_search:
        evolver = ScalarRefinementEvolver(archive=archive, classifier=ClearanceFeedback(),
            parameter=ScalarParameter("memory_rules", "memory_rules.json", ("transit_height_m",), .10, .55))
    else:
        evolver = (ClearancePoolEvolver() if staged else ClearanceEvolver()) if evolver is None else evolver
    scheduler = JobScheduler(root / "scheduler",capacities={"native_worker":1}) if staged else None

    def build(directory, context):
        protocol = CaseProtocol(directory / "cases", splits=dict(
            development={TASK: [dict(seed=1000 if feedback_search else 5000 + context["index"] if staged else 17)]},
            validation={TASK: [dict(seed=(40000 if feedback_search else 20000 if staged else 1000)
                                          + context["index"] * 100 + i) for i in range(episodes)]},
            test={TASK: [dict(seed=(50000 if feedback_search else 30000 if staged else 100000) + i) for i in range(episodes)]}))
        suite = Suite(python, root / "native" / ("round-" + str(context["index"])), resource=resource, gui=gui)
        declared = PreregisteredSuite(suite, protocol)
        stages = {}
        evaluator_class = ExperimentEvaluator
        if staged:
            evaluator_class = MultiFidelityEvaluator
            # Development and screening can repeat one case/candidate. Native
            # directories must distinguish those separately charged executions.
            screening_suite = PreregisteredSuite(Suite(python, suite.root / "screening",
                resource=resource, gui=gui), protocol)
            screening = ExperimentEvaluator(suite=screening_suite, budgets={TASK:Budget(1,1200)}, scope=SCOPE,
                system2=Revision("isaac-development-screening","1"),quota=quota,device_registry=registry)
            stages = dict(screening=screening,screening_resources={"native_worker":1},minimum_success_rate=1.,
                screen_incumbent=screen_incumbent,
                scheduler=scheduler,resources={"native_worker":1},workers=1,wait_seconds=7200)
        evaluator = evaluator_class(suite=declared,
            budgets={TASK: Budget(1, 1200)}, scope=SCOPE,
            system2=Revision("isaac-rollout-research", "1"), quota=quota, device_registry=registry, **stages)
        return SelfHarness(directory, state=state, proposer=RolloutProposer(evaluator, evolver, archive=archive),
            evaluator=evaluator, selector=Selector(), evolution=EvolutionSpec(("memory_rules",)),
            archive=archive, max_candidates=max_candidates, scope=SCOPE, round_context=context,
            profile=dict(tasks={TASK: dict(weight=1, episodes=episodes, score_range=[0, 1], maximum_regression=0)},
                         minimum_gain=0, tie_tolerance=1e-10),
            protocol=dict(identity=evaluator.revision, evaluation_kind="local_contract_evaluation",
                          backend="isaacsim-5.1", qualification=None))

    campaign = ResearchCampaign(root / "campaign", state=state, build_loop=build, memory=memory,
        memory_revision=memory_revision, context=dict(task=TASK), reflector=Reflector(), max_rounds=rounds)
    return campaign, initial, quota


def run(workspace, python, **options):
    campaign, initial, quota = setup(workspace, python, **options)
    result = campaign.run(initial)
    report = campaign.report_test(initial)
    summary = dict(state="completed", selected_revision=result["current"]["revision"],
        selected_policy=result["current"]["harness"]["id"], scope=SCOPE, qualification=None,
        benchmark_wide_score=None, rounds=result["rounds"], final_test=report, quota=quota.status(),
        proposal_provider=read_json(campaign.root / "config.json")["policy"]["identities"]["proposer"]["evolver"])
    if options.get("staged", False) or options.get("feedback_search", False):
        summary["scheduling"] = JobScheduler(Path(workspace)/"scheduler",capacities={"native_worker":1}).status()
        summary["screening"] = dict(role="screening_only",fidelity="one development layout before full paired validation",
            screen_incumbent=options.get("screen_incumbent",False),
            proposal_roles=(["development-scalar-refinement"] if options.get("feedback_search", False)
                            else ["observed-repair", "negative-control", "negative-control"]),native_workers=1)
    atomic_json(Path(workspace) / "result.json", summary)
    return summary
