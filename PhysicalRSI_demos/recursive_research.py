"""Two CPU tasks exercising evidence-driven research memory across rounds.

The scripted hypothesis is deliberately simple. This is an executable contract
and ablation example, not evidence of autonomous scientific or physical gains.
"""
from dataclasses import replace
from pathlib import Path
import shutil

from PhysicalRSI_core.contracts import Contract
from PhysicalRSI_core.embodiment import Embodiment, Revision, System1
from PhysicalRSI_core.experiments import Budget
from PhysicalRSI_core.infra.storage import atomic_json, file_digest, read_json
from PhysicalRSI_core.infra.trial_quota import TrialQuota
from PhysicalRSI_core.lineage import HarnessState
from PhysicalRSI_core.self_harness import SelfHarness, selection
from PhysicalRSI_core.self_harness.artifacts import CLOSURE, verify_harness
from PhysicalRSI_core.self_harness.evaluation import ExperimentEvaluator
from PhysicalRSI_core.self_harness.protocol import CaseProtocol, PreregisteredSuite
from PhysicalRSI_core.self_harness.research_campaign import ResearchCampaign
from PhysicalRSI_core.self_harness.research_memory import ResearchMemory
from .experiment import CounterEnvironment, CounterPolicy, CounterVerifier


def identity(kind):
    return dict(kind=kind, implementation=file_digest(Path(__file__)))


def manifest(root, name, **metadata):
    return dict(id=name, root=str(root), components={
        kind: {kind + ".json": file_digest(root / (kind + ".json"))} for kind in CLOSURE}, **metadata)


TEXT_OBSERVATION = Contract("text-prefix-and-target", embodiment="software-text")
TEXT_ACTION = Contract("append-character", embodiment="software-text")


class TextEnvironment:
    def identity(self):
        return identity("text-environment")

    def describe(self):
        return Embodiment("software-text", "1", "software", TEXT_OBSERVATION, TEXT_ACTION)

    def reset(self, case, context):
        self.text, self.target = case["initial"], case["target"]
        return dict(ready=True, observation=dict(text=self.text, target=self.target))

    def step(self, action, context):
        self.text += action["append"]
        return dict(observation=dict(text=self.text, target=self.target),
                    terminated=len(self.text) >= len(self.target))


class TextPolicy:
    def __init__(self, revision, copy_target):
        self.revision, self.copy_target = revision, copy_target

    def identity(self):
        return dict(identity("text-policy"), revision=self.revision, copy_target=self.copy_target)

    def describe(self):
        return System1("text-policy", self.revision, TEXT_OBSERVATION, TEXT_ACTION)

    def begin_episode(self, task, case, observation, context):
        pass

    def act(self, observation, context):
        index = len(observation["text"])
        return dict(append=observation["target"][index] if self.copy_target else "x")


class TextVerifier:
    def identity(self):
        return identity("text-verifier")

    def verify(self, case, trace):
        success = trace[-1]["observation"]["text"] == case["target"]
        return dict(outcome="success" if success else "failure", reason="Exact text equality",
                    measurements=dict(characters=len(trace)-1))


class Suite:
    def __init__(self):
        self.resets = []

    def identity(self):
        return identity("two-task-suite")

    def admit(self, candidate):
        return dict(accepted=True, evidence=dict(closure=verify_harness(candidate)))

    def environment(self, task):
        base = CounterEnvironment if task == "counter" else TextEnvironment
        resets = self.resets

        class Environment(base):
            def reset(self, case, context):
                resets.append((task, case["target"]))
                return super().reset(case, context)

        return Environment()

    def policy(self, candidate, task):
        revision = verify_harness(candidate)
        control = read_json(Path(candidate["root"]) / "control.json")
        if task == "text":
            return TextPolicy(revision, control["copy_target"])

        class Policy(CounterPolicy):
            def describe(self):
                return replace(super().describe(), revision=revision)

        return Policy(control["increment"])

    def verifier(self, task):
        return CounterVerifier() if task == "counter" else TextVerifier()


class Proposer:
    def __init__(self, evaluator):
        self.evaluator = evaluator

    def identity(self):
        return identity("memory-conditioned-proposer")

    def develop(self, parent, output):
        return self.evaluator.development(parent, output)

    resume_development = develop

    def propose_with_context(self, parent, feedback, context, output):
        root = output / "candidate"
        shutil.copytree(parent["root"], root)
        learned = "unit-step-and-copy" in context["lessons"]
        atomic_json(root / "control.json", dict(increment=1 if learned else 2, copy_target=learned))
        atomic_json(root / "prompts.json", dict(round=context["index"]))
        return [manifest(root, "candidate-" + str(context["index"]),
                         parent_sha256=verify_harness(parent), method="scripted-memory-ablation",
                         changes=dict(control=learned), environment="two-cpu-tasks",
                         evidence=feedback["evidence"], costs=dict(proposals=1),
                         used_lessons=["unit-step-and-copy"] if learned else [],
                         hypothesis="Unit increments and target copying solve the observed failures",
                         expected_effect="Higher exact-match success", required_checks=["paired-success"])]


class Reflector:
    def __init__(self):
        self.calls = 0

    def identity(self):
        return identity("development-reflector")

    def reflect(self, context, development, output):
        self.calls += 1
        operations = []
        if ("unit-step-and-copy" not in context["lessons"]
                and any(row["outcome"] == "failure" for row in development["episodes"])):
            operations.append(dict(kind="add", lesson_id="unit-step-and-copy", lesson=dict(
                conditions=context["scope"], hypothesis="Try unit increments and public target copying",
                required_checks=["paired-success"], counterexample_checks=[], evidence=development["evidence"])))
        return dict(schema="physicalrsi.memory-delta/v1", operations=operations)


class Selector:
    def identity(self):
        return dict(implementation=file_digest(Path(selection.__file__)))

    def select(self, *args, **kwargs):
        return selection.select_survivor(*args, **kwargs)


def setup(root, *, update_memory=True, max_rounds=3, max_trials=30):
    root = Path(root).resolve()
    seed = root / "seed"
    if not seed.exists():
        for kind in CLOSURE:
            atomic_json(seed / (kind + ".json"), dict(increment=2, copy_target=False)
                        if kind == "control" else dict(kind=kind))
    initial = manifest(seed, "parent")
    memory = ResearchMemory(root / "memory")
    evidence = {str(seed / "control.json"): file_digest(seed / "control.json")}
    memory_revision = memory.write({"baseline": dict(conditions={"study": "two-task-cpu"},
        hypothesis="Measure exact-match failures before proposing changes", required_checks=["paired-success"],
        counterexample_checks=[], evidence=evidence)})
    state, suite, reflector = HarnessState(root / "state"), Suite(), Reflector()
    quota = TrialQuota(root / "quota", max_trials=max_trials)

    def build(directory, context):
        index = context["index"]
        counter = lambda target: dict(initial=0, target=target)
        text = lambda target: dict(initial="", target=target)
        # Reset identities include the actual target, not auxiliary metadata.
        splits = dict(development=dict(counter=[counter(3)], text=[text("ab")]),
                      validation=dict(counter=[counter(5+4*index), counter(7+4*index)],
                                      text=[text("b" * (index+1)), text("a" + "b" * (index+1))]),
                      test=dict(counter=[counter(21)], text=[text("final")]))
        # Avoid the development target "ab" in the first validation round.
        splits["validation"]["text"][1] = text("ba" + "b" * index)
        plan = CaseProtocol(directory / "cases", splits=splits, identity_keys=("initial", "target"))
        evaluator = ExperimentEvaluator(suite=PreregisteredSuite(suite, plan),
            budgets={task: Budget(32, 30) for task in ("counter", "text")}, scope="software",
            system2=Revision("scripted-research", "1"), quota=quota)
        return SelfHarness(directory, state=state, proposer=Proposer(evaluator), evaluator=evaluator,
            selector=Selector(), scope="software", max_candidates=1, round_context=context,
            profile=dict(tasks={task: dict(weight=1, episodes=2, score_range=[0, 1],
                         maximum_regression=0) for task in ("counter", "text")},
                         minimum_gain=0, tie_tolerance=1e-10),
            protocol=dict(identity=evaluator.revision, evaluation_kind="local_contract_evaluation"))

    campaign = ResearchCampaign(root / "campaign", state=state, build_loop=build,
        memory=memory, memory_revision=memory_revision, context={"study": "two-task-cpu"},
        reflector=reflector, max_rounds=max_rounds, update_memory=update_memory)
    return campaign, initial, suite, quota


def run(root):
    campaign, initial, _, quota = setup(root, max_trials=32)
    result = campaign.run(initial)
    report = campaign.report_test(initial)
    return dict(result, quota=quota.status(), final_test=report)
