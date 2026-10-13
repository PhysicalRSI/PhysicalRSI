"""Compile scalar rollout experience into a reusable vector action program.

This example transfers within a shared linear actuator family, not to arbitrary
robot dynamics. A later development-only dynamics shift supplies a counterexample.
"""
from copy import deepcopy
from dataclasses import replace
import math
from pathlib import Path
import shutil
import stat

from PhysicalRSI_core.contracts import Context, Contract, Operation
from PhysicalRSI_core.embodiment import Dependency, Revision
from PhysicalRSI_core.experiments import Budget
from PhysicalRSI_core.infra.storage import atomic_json, digest, file_digest, locked, read_json
from PhysicalRSI_core.infra.trial_quota import TrialQuota
from PhysicalRSI_core.lineage import HarnessState
from PhysicalRSI_core.skills import SkillLibrary, SkillUnavailable
from PhysicalRSI_core.self_harness import SelfHarness
from PhysicalRSI_core.self_harness.archive import CandidateArchive
from PhysicalRSI_core.self_harness.campaign import _manifest
from PhysicalRSI_core.self_harness.artifacts import CLOSURE, EvolutionSpec, verify_harness
from PhysicalRSI_core.self_harness.evaluation import ExperimentEvaluator, freeze_json
from PhysicalRSI_core.self_harness.protocol import CaseProtocol, PreregisteredSuite
from PhysicalRSI_core.self_harness.research_campaign import ResearchCampaign
from PhysicalRSI_core.self_harness.research_memory import ResearchMemory
from PhysicalRSI_core.self_harness.rollout import RolloutProposer
from PhysicalRSI_core.self_harness.skill_compiler import ExperienceArchive, ExperienceCompiler
from .recursive_research import Selector, manifest
from .rollout_evolution import Policy as CalibrationPolicy, Suite as CalibrationSuite


CONDITIONS = {"actuator_family": "shared-linear", "observation": "public-position-target"}
LESSON = "inverse-response"


def identity(kind):
    return dict(kind=kind, implementation=file_digest(Path(__file__)))


def command(action):
    return list(action["joint_delta"]) if isinstance(action, dict) else [action]


class VectorBackend:
    """Reviewed, bounded data program; it cannot import or evaluate Python."""
    def identity(self):
        return identity("vector-action-program/v1")

    def build(self, root, entrypoint):
        program = read_json(Path(root) / entrypoint)
        if (set(program) != {"schema", "steps"} or program["schema"] != "vector-action/v1"
                or not isinstance(program["steps"], list) or not 2 <= len(program["steps"]) <= 32
                or program["steps"][0] != {"op": "difference"}):
            raise ValueError("Invalid vector action program")
        for step in program["steps"][1:]:
            if (set(step) != {"op", "factor"} or step["op"] != "scale"
                    or type(step["factor"]) not in {int, float} or not math.isfinite(step["factor"])):
                raise ValueError("Unknown vector operation or nonfinite parameter")

        def execute(observation, context):
            position, target = observation["position"], observation["target"]
            if not position or len(position) != len(target):
                raise ValueError("Vector observation dimensions differ")
            values = [t-p for p, t in zip(position, target)]
            for step in program["steps"][1:]:
                values = [v * step["factor"] for v in values]
            if any(not math.isfinite(v) for v in values):
                raise ValueError("Nonfinite compiled action")
            return values

        return Operation("inverse-response", digest(dict(backend=self.identity(), program=program)),
            Contract("position-target-vector", embodiment="software-calibration"),
            Contract("displacement-vector", embodiment="software-calibration"), execute)


class ResponseCompiler:
    def __init__(self):
        self.calls = 0

    def identity(self):
        return identity("observed-response-skill-compiler/v1")

    def compile(self, evaluations, request, output):
        self.calls += 1
        numerator = denominator = 0.
        for evaluation in evaluations:
            trace = evaluation["trajectory"]
            for previous, current in zip(trace, trace[1:]):
                for p, q, a in zip(previous["observation"]["position"], current["observation"]["position"], command(current["action"])):
                    numerator += a * (q-p)
                    denominator += a*a
        if numerator == 0 or denominator == 0:
            raise ValueError("Source transitions do not identify an actuator response")
        program = dict(schema="vector-action/v1", steps=[dict(op="difference"), dict(op="scale", factor=denominator/numerator)])
        freeze_json(output / "program.json", program)
        return dict(entrypoint="program.json", files={"program.json": file_digest(output / "program.json")},
            hypothesis="The inverse response measured on one axis applies across the declared actuator family")

    # Pure deterministic synthesis can verify and reuse its own existing output.
    resume_compile = compile


class InverseReplayCheck:
    def identity(self):
        return identity("inverse-transition-replay-check/v1")

    def check(self, operation, evaluations, request):
        cases = []
        for evaluation in evaluations:
            trace = evaluation["trajectory"]
            for previous, current in zip(trace, trace[1:]):
                observation = dict(position=previous["observation"]["position"], target=current["observation"]["position"])
                expected = command(current["action"])
                actual = operation(observation, Context("compile-check-" + str(len(cases))))
                passed = len(actual) == len(expected) and all(abs(a-b) < 1e-9 for a, b in zip(actual, expected))
                cases.append(dict(trial=evaluation["trial_sha256"], input=observation, expected=expected, actual=actual, passed=passed))
        return {"inverse-transition": dict(passed=bool(cases) and all(c["passed"] for c in cases),
            trials=sorted({c["trial"] for c in cases}), cases=cases)}

    resume_check = check


class SkillPolicy(CalibrationPolicy):
    def __init__(self, task, revision, operation, skill_revision, memory_revision, calls):
        super().__init__(task, revision, 1.)
        self.operation, self.skill_revision, self.memory_revision, self.calls = operation, skill_revision, memory_revision, calls

    def identity(self):
        return dict(identity("compiled-response-policy/v1"), task=self.task, revision=self.revision,
                    operation=self.operation.revision, skill=self.skill_revision, memory=self.memory_revision)

    def describe(self):
        return replace(super().describe(), dependencies=(
            Dependency("inverse-response", self.skill_revision, "skill"),
            Dependency("compiled-skill-memory", self.memory_revision, "memory")))

    def act(self, observation, context):
        # Explicit contract adaptation: scalar/native dictionaries <-> vector program.
        values = self.operation(observation, context)
        self.calls.append(self.task)
        return values[0] if self.task == "axis" else dict(joint_delta=values)


class TransferSuite(CalibrationSuite):
    def __init__(self, gains, library, memory):
        super().__init__(gains)
        self.library, self.memory, self.skill_calls = library, memory, []

    def identity(self):
        return dict(super().identity(), transfer=identity("skill-transfer-suite/v1"),
                    library=self.library.identity(), memory_root=str(self.memory.root.resolve()), backend=VectorBackend().identity())

    def _skill(self, candidate):
        root = Path(candidate["root"])
        revision = read_json(root / "skills.json").get("revision")
        if revision is None:
            return None
        memory = read_json(root / "memory_rules.json")["revision"]
        lesson = read_json(root / "skill_selection.json")["lesson"]
        if self.memory.read(memory)["lessons"][lesson]["skill"]["revision"] != revision:
            raise ValueError("Harness skill and memory references differ")
        return (self.library.bind(memory=self.memory, memory_revision=memory, lesson_id=lesson,
            context=CONDITIONS, backend=VectorBackend(), package=root / "compiled" / revision), revision, memory)

    def admit(self, candidate):
        result = super().admit(candidate)
        try:
            self._skill(candidate)
        except SkillUnavailable as error:
            return dict(accepted=False, reason=str(error), evidence=result["evidence"])
        return result

    def policy(self, candidate, task):
        bound = self._skill(candidate)
        if bound is None:
            return super().policy(candidate, task)
        operation, skill, memory = bound
        return SkillPolicy(task, verify_harness(candidate), operation, skill, memory, self.skill_calls)


class TransferEvolver:
    def __init__(self, library):
        self.library = library

    def identity(self):
        return dict(identity("compiled-skill-transfer-proposal/v1"), library=self.library.identity())

    def propose(self, parent, evaluations, context, output):
        if read_json(Path(parent["root"]) / "skills.json").get("revision") is not None:
            return []
        # Target-task evaluations do not enter compilation. Only the previously
        # compiled source skill and its pinned memory enter this proposal.
        lesson = context["lessons"][LESSON]
        revision = lesson["skill"]["revision"]
        root = output / "skill-candidate"
        shutil.copytree(parent["root"], root)
        atomic_json(root / "skills.json", dict(revision=revision))
        atomic_json(root / "memory_rules.json", dict(revision=context["memory_revision"]))
        atomic_json(root / "skill_selection.json", dict(lesson=LESSON))
        folder = self.library.materialize(revision, root / "compiled")
        candidate = manifest(root, "compiled-response", parent_sha256=verify_harness(parent),
            method="source-skill-transfer", changes=dict(skills=revision), environment=CONDITIONS,
            evidence=lesson["evidence"], costs=dict(proposals=1), used_lessons=[LESSON],
            hypothesis=lesson["hypothesis"], expected_effect="Transfer the source-axis response to structured actions",
            required_checks=["paired-position-success"])
        added = {str(p.relative_to(root)): file_digest(p) for p in folder.rglob('*') if p.is_file()}
        candidate['components']['skills'].update(added)
        candidate['file_modes'].update({name: stat.S_IMODE((root / name).stat().st_mode) for name in added})
        verify_harness(candidate)
        return [candidate]


class RetainMemory:
    def identity(self):
        return identity("retain-source-skill-memory/v1")

    def reflect(self, context, development, output):
        return dict(schema="physicalrsi.memory-delta/v1", operations=[])


def cases(tasks, targets):
    return {task: [dict(initial=[0.] * (1 if task == "axis" else 2),
                       target=[value] if task == "axis" else [value, -2*value]) for value in targets] for task in tasks}


def prepare(root, *, gain=2.):
    root = Path(root).resolve()
    freeze_json(root / "config.json", dict(implementation=identity("skill-transfer-acceptance/v1"), gain=gain))
    seed = root / "seed"
    for kind in CLOSURE:
        value = {"axis": 1., "vector": 1.} if kind == "control" else dict(kind=kind)
        freeze_json(seed / (kind + ".json"), value)
    initial = manifest(seed, "uncompiled")
    library, memory = SkillLibrary(root / "library"), ResearchMemory(root / "memory")
    compiler = ExperienceCompiler(library=library, compiler=ResponseCompiler(), backend=VectorBackend(), checker=InverseReplayCheck())
    quota = TrialQuota(root / "quota", max_trials=18)
    source_suite = CalibrationSuite({"axis": gain})
    protocol = CaseProtocol(root / "source/cases", splits={split: cases(["axis"], values)
        for split, values in dict(development=[1,-2], validation=[101,103], test=[1001]).items()},
        identity_keys=("initial", "target"))
    source = ExperimentEvaluator(suite=PreregisteredSuite(source_suite, protocol), budgets={"axis": Budget(1,30)},
        scope="software", system2=Revision("source-calibration", "1"), quota=quota)
    feedback = source.development(initial, root / "source/execution")
    experience = compiler.experiences.capture(initial, feedback, root / "source/execution")
    skill = compiler.compile(experience, name=LESSON, conditions=CONDITIONS, required_checks=["inverse-transition"],
                             output=root / "compilation")
    memory_revision = compiler.install(skill, memory=memory, memory_revision=None, lesson_id=LESSON)
    return root, initial, library, memory, compiler, quota, source_suite, experience, skill, memory_revision


def target_campaign(root, initial, library, memory, memory_revision, quota, *, gain):
    suite = TransferSuite({"axis": gain, "vector": gain}, library, memory)
    archive = CandidateArchive(root / "archive")
    state = HarnessState(root / "state")

    def build(directory, context):
        protocol = CaseProtocol(directory / "cases", splits=dict(development=cases(suite.gains,[11]),
            validation=cases(suite.gains,[3+4*context["index"],5+4*context["index"]]), test=cases(suite.gains,[23])),
            identity_keys=("initial","target"))
        evaluator = ExperimentEvaluator(suite=PreregisteredSuite(suite,protocol),
            budgets={task: Budget(1,30) for task in suite.gains}, scope="software",
            system2=Revision("skill-transfer-validation","1"), quota=quota)
        return SelfHarness(directory, state=state, evaluator=evaluator,
            proposer=RolloutProposer(evaluator,TransferEvolver(library),archive=archive), selector=Selector(),
            profile=dict(tasks={task: dict(weight=1,episodes=2,score_range=[0,1],maximum_regression=0)
                               for task in suite.gains}, minimum_gain=0,tie_tolerance=1e-10),
            protocol=dict(identity=evaluator.revision,evaluation_kind="local_contract_evaluation"),
            scope="software", max_candidates=1,round_context=context,archive=archive,
            evolution=EvolutionSpec(("skills","skill_selection","memory_rules")))

    return ResearchCampaign(root / "campaign",state=state,build_loop=build,memory=memory,memory_revision=memory_revision,
        context=CONDITIONS,reflector=RetainMemory(),max_rounds=2), suite


def run(root, *, gain=2.):
    with locked(Path(root) / ".run.lock"):
        return _run(root, gain=gain)


def _run(root, *, gain):
    prepared = prepare(root, gain=gain)
    root, initial, library, memory, compiler, quota, source_suite, experience, skill, memory_revision = prepared
    if (root / "completion.json").exists():
        result = read_json(root / "result.json")
        completion = read_json(root / "completion.json")
        files = {name: sha for name, sha in _manifest(root).items() if name != "completion.json"}
        if completion != dict(result_sha256=digest(result), files=files):
            raise ValueError("Completed skill transfer evidence changed")
        if result["skill_revision"] != skill or result["memory_revision"] != memory_revision:
            raise ValueError("Skill transfer result binding changed")
        library.read(skill)
        compiler.experiences.read(result["counterexample_experience"])
        memory.read(result["suspended_memory_revision"])
        if HarnessState(root / "state").resolve() != result["restored"]:
            raise ValueError("Skill withdrawal state changed")
        return result
    campaign, suite = target_campaign(root,initial,library,memory,memory_revision,quota,gain=gain)
    transfer_path = root / "transfer.json"
    if transfer_path.exists():
        transfer = read_json(transfer_path)
        research, final = transfer["research"], transfer["final_test"]
        if (campaign.state.read(research["current"]["revision"]) != research["current"]
                or transfer["campaign_files"] != _manifest(campaign.root)
                or transfer["archive_files"] != _manifest(root / "archive")
                or transfer["skill_revision"] != skill or transfer["memory_revision"] != memory_revision):
            raise ValueError("Frozen transfer evidence changed")
    else:
        research = campaign.run(initial)
        final = campaign.report_test(initial)
        restarted, restarted_suite = target_campaign(root,initial,library,memory,memory_revision,quota,gain=gain)
        if restarted.run(initial) != research or restarted.report_test(initial) != final or restarted_suite.resets:
            raise ValueError("Skill transfer restart was not a pure read")
        freeze_json(transfer_path, dict(research=research, final_test=final, skill_revision=skill,
            memory_revision=memory_revision, campaign_files=_manifest(campaign.root), archive_files=_manifest(root / "archive")))
    shifted = TransferSuite({"vector": gain*1.5},library,memory)
    protocol = CaseProtocol(root / "counterexample/cases",splits={split: cases(["vector"],[target])
        for split,target in dict(development=31,validation=41,test=51).items()},identity_keys=("initial","target"))
    evaluator = ExperimentEvaluator(suite=PreregisteredSuite(shifted,protocol),budgets={"vector":Budget(1,30)},
        scope="software",system2=Revision("skill-counterexample","1"),quota=quota)
    selected = research["current"]
    feedback = evaluator.development(selected["harness"],root / "counterexample/execution")
    counterexample = compiler.experiences.capture(selected["harness"],feedback,root / "counterexample/execution")
    suspended = compiler.suspend(counterexample,memory=memory,memory_revision=memory_revision,lesson_id=LESSON,
        check="changed-response",explanation="A fresh development rollout contradicts transfer across changed actuator gain")
    try:
        library.bind(memory=memory,memory_revision=suspended,lesson_id=LESSON,context=CONDITIONS,backend=VectorBackend())
    except SkillUnavailable:
        pass
    else:
        raise ValueError("Suspended skill remained eligible")
    restored = campaign.state.rollback(selected["revision"],reason="Development counterexample suspended the compiled skill")
    after = evaluator.development(restored["harness"],root / "after-withdrawal")
    if any(d["kind"] == "skill" for v in after["evaluations"] for d in v["receipt"]["binding"]["system1"]["dependencies"]):
        raise ValueError("Withdrawn skill still executed")
    result = dict(schema="physicalrsi.skill-transfer-acceptance/v1",source_experience=experience,skill_revision=skill,
        memory_revision=memory_revision,research=research,final_test=final,counterexample_experience=counterexample,
        suspended_memory_revision=suspended,restored=restored,withdrawn_trial=after["evaluations"][0],
        restart_new_trials=0,quota=quota.status(),scope="software",qualification=None)
    freeze_json(root / "result.json",result)
    freeze_json(root / "completion.json",dict(result_sha256=digest(result),files=_manifest(root)))
    return result
