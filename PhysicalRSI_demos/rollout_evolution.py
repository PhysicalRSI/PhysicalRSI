"""CPU rollout-driven calibration across scalar and structured action tasks.

The evolver estimates actuator gain from observed transitions. It has no access
to environment gains or validation cases. This is an algorithmic integration
example, not a simulator, a learned world model or physical qualification.
"""
from pathlib import Path
import shutil

from PhysicalRSI_core.contracts import Contract
from PhysicalRSI_core.embodiment import Embodiment, Revision, System1
from PhysicalRSI_core.experiments import Budget
from PhysicalRSI_core.infra.storage import atomic_json, digest, file_digest, read_json
from PhysicalRSI_core.infra.trial_quota import TrialQuota
from PhysicalRSI_core.lineage import HarnessState
from PhysicalRSI_core.self_harness import SelfHarness
from PhysicalRSI_core.self_harness.archive import CandidateArchive
from PhysicalRSI_core.self_harness.artifacts import CLOSURE, EvolutionSpec, verify_harness
from PhysicalRSI_core.self_harness.evaluation import ExperimentEvaluator
from PhysicalRSI_core.self_harness.protocol import CaseProtocol, PreregisteredSuite
from PhysicalRSI_core.self_harness.research_campaign import ResearchCampaign
from PhysicalRSI_core.self_harness.research_memory import ResearchMemory
from PhysicalRSI_core.self_harness.rollout import RolloutProposer
from .recursive_research import Selector, manifest


def identity(kind, **values):
    return dict(kind=kind, implementation=file_digest(Path(__file__)), **values)


def contracts(task):
    return (Contract(task + "-position-target", embodiment="software-calibration"),
            Contract(task + "-command", embodiment="software-calibration"))


class Environment:
    def __init__(self, task, gain, resets):
        self.task, self.gain, self.resets = task, gain, resets

    def identity(self):
        return identity("calibration-environment", task=self.task, dynamics=digest(self.gain))

    def describe(self):
        return Embodiment("calibration", digest(self.gain), "software", *contracts(self.task))

    def reset(self, case, context):
        self.resets.append((self.task, case))
        self.position, self.target = list(case["initial"]), list(case["target"])
        return dict(ready=True, observation=dict(position=self.position, target=self.target))

    def step(self, action, context):
        command = [action] if self.task == "axis" else action["joint_delta"]
        self.position = [p + self.gain * a for p, a in zip(self.position, command)]
        return dict(observation=dict(position=self.position, target=self.target), terminated=True)


class Policy:
    def __init__(self, task, revision, coefficient):
        self.task, self.revision, self.coefficient = task, revision, coefficient

    def identity(self):
        return identity("calibration-policy", task=self.task, revision=self.revision, coefficient=self.coefficient)

    def describe(self):
        return System1("calibration", self.revision, *contracts(self.task))

    def begin_episode(self, task, case, observation, context):
        pass

    def act(self, observation, context):
        commands = [self.coefficient * (t - p) for p, t in zip(observation["position"], observation["target"])]
        return commands[0] if self.task == "axis" else dict(joint_delta=commands)


class Verifier:
    def identity(self):
        return identity("position-verifier")

    def verify(self, case, trace):
        residual = sum((p-t)**2 for p, t in zip(trace[-1]["observation"]["position"], case["target"]))
        return dict(outcome="success" if residual < 1e-12 else "failure",
                    reason="Measured squared position error", measurements=dict(squared_error=residual))


class Suite:
    def __init__(self, gains):
        self.gains, self.resets = gains, []

    def identity(self):
        return identity("calibration-suite", dynamics=digest(self.gains))

    def admit(self, candidate):
        return dict(accepted=True, evidence=dict(closure=verify_harness(candidate)))

    def environment(self, task):
        return Environment(task, self.gains[task], self.resets)

    def policy(self, candidate, task):
        return Policy(task, verify_harness(candidate), read_json(Path(candidate["root"]) / "control.json")[task])

    def verifier(self, task):
        return Verifier()


def fit_coefficients(evaluations):
    coefficients = {}
    for value in evaluations:
        task, trace = value["trial"]["task"], value["trajectory"]
        numerator = denominator = 0.0
        for previous, current in zip(trace, trace[1:]):
            action = [current["action"]] if task == "axis" else current["action"]["joint_delta"]
            for p, q, a in zip(previous["observation"]["position"], current["observation"]["position"], action):
                numerator += a * (q - p)
                denominator += a * a
        if denominator == 0 or numerator == 0:
            return None  # No identifiable response; preserve uncertainty.
        coefficients[task] = denominator / numerator
    return coefficients


class LeastSquaresEvolver:
    def identity(self):
        return identity("observed-transition-least-squares")

    def propose(self, parent, evaluations, context, output):
        if all(value["outcome"] == "success" for value in evaluations):
            return []
        coefficients = fit_coefficients(evaluations)
        if coefficients is None:
            return []
        root = output / "calibrated-candidate"
        shutil.copytree(parent["root"], root)
        atomic_json(root / "control.json", coefficients)
        return [manifest(root, "calibrated-" + str(context["index"]),
            parent_sha256=verify_harness(parent), method="observed-transition-least-squares",
            changes=dict(control=coefficients), environment="software-calibration",
            evidence={path: sha for value in evaluations for path, sha in value["evidence"].items()},
            costs=dict(proposals=1), used_lessons=[],
            hypothesis="Compensate the gain measured from action-observation transitions",
            expected_effect="Reduce position error on fresh targets", required_checks=["paired-position-success"])]


class Reflector:
    def identity(self):
        return identity("calibration-reflector")

    def reflect(self, context, development, output):
        operations = []
        if "response-calibration" not in context["lessons"] and any(
                row["outcome"] == "failure" for row in development["episodes"]):
            operations.append(dict(kind="add", lesson_id="response-calibration", lesson=dict(
                conditions=context["scope"], hypothesis="Estimate response from observed actions before choosing feedback gain",
                required_checks=["paired-position-success"], counterexample_checks=[], evidence=development["evidence"])))
        return dict(schema="physicalrsi.memory-delta/v1", operations=operations)


def setup(root, *, gains=None, evolver_factory=None, max_candidates=1, task_weights=None, max_trials=14,
          scheduled_workers=None, screening_threshold=None, screen_incumbent=False):
    root = Path(root).resolve()
    gains = dict(gains or {"axis": 2.0, "vector": 0.5})
    seed = root / "seed"
    if not seed.exists():
        for kind in CLOSURE:
            atomic_json(seed / (kind + ".json"), {task: 1.0 for task in gains} if kind == "control" else dict(kind=kind))
    initial = manifest(seed, "uncalibrated")
    state, suite = HarnessState(root / "state"), Suite(gains)
    archive = CandidateArchive(root / "archive")
    evolver = evolver_factory(archive) if evolver_factory else LeastSquaresEvolver()
    memory = ResearchMemory(root / "memory")
    memory_revision = memory.write({"baseline": dict(conditions={"study": "rollout-calibration"},
        hypothesis="Use independently measured position error", required_checks=["paired-position-success"],
        counterexample_checks=[], evidence={str(seed / "control.json"): file_digest(seed / "control.json")})})
    quota = TrialQuota(root / "quota", max_trials=max_trials)
    scheduling = {}
    evaluator_class = ExperimentEvaluator
    if scheduled_workers is not None:
        from PhysicalRSI_core.infra.scheduler import JobScheduler
        from PhysicalRSI_core.self_harness.scheduled import ScheduledEvaluator
        evaluator_class = ScheduledEvaluator
        scheduling = dict(scheduler=JobScheduler(root / "scheduler", capacities={"cpu": scheduled_workers}),
                          resources={"cpu": 1}, workers=scheduled_workers)
    if screening_threshold is not None:
        if scheduled_workers is None:
            raise ValueError("The staged demo requires a scheduler")
        from PhysicalRSI_core.self_harness.fidelity import MultiFidelityEvaluator
        evaluator_class = MultiFidelityEvaluator

    def build(directory, context):
        def cases(targets):
            return {task: [dict(initial=[0] * (1 if task == "axis" else 2),
                               target=[n] if task == "axis" else [n, -2*n]) for n in targets] for task in gains}
        protocol = CaseProtocol(directory / "cases", splits=dict(development=cases([1]),
            validation=cases([3+context["index"]*4, 5+context["index"]*4]), test=cases([23])),
            identity_keys=("initial", "target"))
        declared_suite = PreregisteredSuite(suite, protocol)
        stages = {}
        if screening_threshold is not None:
            stages = dict(screening=ExperimentEvaluator(suite=declared_suite,
                budgets={task: Budget(1, 30) for task in gains}, scope="software",
                system2=Revision("development-screening", "1"), quota=quota),
                screening_resources={"cpu": 1}, minimum_success_rate=screening_threshold,
                screen_incumbent=screen_incumbent)
        evaluator = evaluator_class(suite=declared_suite,
            budgets={task: Budget(1, 30) for task in gains}, scope="software",
            system2=Revision("rollout-least-squares", "1"), quota=quota, **scheduling, **stages)
        return SelfHarness(directory, state=state, proposer=RolloutProposer(evaluator, evolver, archive=archive),
            evaluator=evaluator, selector=Selector(), evolution=EvolutionSpec(("control",)),
            archive=archive, max_candidates=max_candidates, scope="software", round_context=context,
            profile=dict(tasks={task: dict(weight=(task_weights or {}).get(task, 1), episodes=2, score_range=[0, 1],
                         maximum_regression=0) for task in gains}, minimum_gain=0, tie_tolerance=1e-10),
            protocol=dict(identity=evaluator.revision, evaluation_kind="local_contract_evaluation"))

    campaign = ResearchCampaign(root / "campaign", state=state, build_loop=build, memory=memory,
        memory_revision=memory_revision, context={"study": "rollout-calibration"}, reflector=Reflector(), max_rounds=2)
    return campaign, initial, suite, quota


def run(root, *, gains=None):
    campaign, initial, _, quota = setup(root, gains=gains)
    result = campaign.run(initial)
    return dict(result, final_test=campaign.report_test(initial), quota=quota.status())
