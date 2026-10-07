from dataclasses import replace
from pathlib import Path
import uuid

import pytest

from PhysicalRSI_core.embodiment import Dependency, Revision, System1
from PhysicalRSI_core.experiments import Budget
from PhysicalRSI_core.infra.isolated_program import PythonIsolation
from PhysicalRSI_core.infra.storage import atomic_json, file_digest, read_json
from PhysicalRSI_core.infra.trial_quota import TrialQuota
from PhysicalRSI_core.lineage import HarnessState
from PhysicalRSI_core.self_harness import SelfHarness
from PhysicalRSI_core.self_harness.artifacts import CLOSURE, verify_harness
from PhysicalRSI_core.self_harness.campaign import ImprovementCampaign
from PhysicalRSI_core.self_harness.code_policy import IsolatedHarnessPolicy, PYTHON_SKILL_SCHEMA
from PhysicalRSI_core.self_harness.evaluation import ExperimentEvaluator
from PhysicalRSI_core.self_harness.proposals import EnumeratedEdits, JsonEditContract, ProposalLimits, StructuredProposer
from PhysicalRSI_demos.experiment import OBSERVATION, ACTION
from test_improvement_campaign import Selector, Suite, manifest
from test_isolated_policy import assert_stopped, context


def skill(increment):
    return dict(schema="physicalrsi.python-skill/v1", source=(
        "def begin_episode(task, case, observation): return None\n"
        f"def act(observation, state): return dict(action={increment}, state=state)\n"))


def seed(root, artifact=None):
    for component in CLOSURE:
        atomic_json(root / (component + ".json"),
                    (artifact or skill(2)) if component == "skills" else dict(kind=component))
    return manifest(root, "parent")


def program(candidate, profile, output, **kwargs):
    specification = kwargs.pop("specification", System1("candidate-code", verify_harness(candidate), OBSERVATION, ACTION))
    return IsolatedHarnessPolicy(candidate, skill_file="skills.json", specification=specification,
                                 isolation=profile, output=output, **kwargs)


def test_source_edits_execute_in_isolation_then_inherit_from_independent_evidence(tmp_path, isolation_runtime):
    parent = seed(tmp_path / "seed")
    profile = PythonIsolation(isolation_runtime)

    class CodeSuite(Suite):
        def identity(self):
            return dict(super().identity(), isolation=profile.identity(), source=file_digest(Path(__file__)))

        def policy(self, candidate, task):
            return program(candidate, profile, tmp_path / "workers" / uuid.uuid4().hex)

    suite = CodeSuite()
    quota = TrialQuota(tmp_path / "quota", max_trials=6)
    evaluator = ExperimentEvaluator(suite=suite, budgets={"counter": Budget(10, 15)},
        scope="software", system2=Revision("reviewed-source-search", "1"), quota=quota)
    contract = JsonEditContract({"skills.json": dict(component="skills", schema=PYTHON_SKILL_SCHEMA)})
    proposer = StructuredProposer(evaluator=evaluator,
        strategy=EnumeratedEdits([{"skills.json": skill(1)}]), contract=contract,
        objective="Compare a reviewed source edit using independent counter measurements",
        limits=ProposalLimits(max_candidates=1))
    state = HarnessState(tmp_path / "state")

    def build_loop(directory):
        return SelfHarness(directory, state=state, proposer=proposer, evaluator=evaluator,
            selector=Selector(), scope="software", max_candidates=1,
            profile=dict(tasks={"counter": dict(weight=1, episodes=2, score_range=[0, 1],
                maximum_regression=0)}, minimum_gain=0, tie_tolerance=1e-10),
            protocol=dict(identity=evaluator.revision, evaluation_kind="local_contract_evaluation"))

    campaign = ImprovementCampaign(tmp_path / "campaign", state=state, build_loop=build_loop, max_rounds=2)
    result = campaign.run(parent)
    assert [row["outcome"] for row in result["rounds"]] == ["inherited", "retained"]
    child = result["current"]["harness"]
    assert read_json(Path(child["root"]) / "skills.json") == skill(1)
    assert read_json(Path(parent["root"]) / "skills.json") == skill(2)
    assert child["components"]["foundation"] == parent["components"]["foundation"]
    trials = list((tmp_path / "campaign").rglob("experiment.json"))
    assert len(trials) == 6 and len(suite.resets) == 6
    for path in trials:
        experiment = read_json(path)
        identity = experiment["adapters"]["policy"]
        declaration = experiment["binding"]["system1"]
        assert declaration["revision"] == identity["harness_sha256"] == experiment["system2"]["candidate_sha256"]
        dependency, = declaration["dependencies"]
        assert dependency["kind"] == "skill" and dependency["name"] == "skills.json"
        assert dependency["revision"] in {parent["components"]["skills"]["skills.json"], child["components"]["skills"]["skills.json"]}
        assert read_json(path.parent / "policy.json")["stopped"] is True
    assert campaign.run(parent) == result and len(suite.resets) == 6
    assert_stopped(tmp_path / "workers")


def test_skill_loading_never_executes_source_on_host(tmp_path, isolation_runtime):
    marker = tmp_path / "must-not-exist"
    artifact = dict(skill(1), source=f"open({str(marker)!r}, 'w').write('host execution')\n" + skill(1)["source"])
    candidate = seed(tmp_path / "seed", artifact)
    policy = program(candidate, PythonIsolation(isolation_runtime), tmp_path / "workers")
    assert not marker.exists()
    ctx = context(policy)
    with pytest.raises(RuntimeError, match="worker stopped"):
        policy.begin_episode("counter", {}, {}, ctx)
    assert policy.end_episode(ctx)["stopped"] and not marker.exists()
    assert_stopped(tmp_path / "workers")


@pytest.mark.parametrize("change", ["revision", "alias", "dependency", "artifact", "drift"])
def test_skill_binding_rejects_misattribution_and_changed_source(tmp_path, isolation_runtime, change):
    candidate = seed(tmp_path / "seed")
    profile = PythonIsolation(isolation_runtime)
    specification = System1("candidate-code", verify_harness(candidate), OBSERVATION, ACTION)
    if change == "revision":
        specification = replace(specification, revision="different")
    elif change == "dependency":
        specification = replace(specification, dependencies=(Dependency("skills.json", "wrong", "skill"),))
    elif change == "alias":
        candidate["components"]["foundation"]["skills.json"] = candidate["components"]["skills"]["skills.json"]
        specification = replace(specification, revision=verify_harness(candidate))
    elif change == "artifact":
        atomic_json(tmp_path / "seed/skills.json", dict(source=skill(1)["source"]))
        candidate = manifest(tmp_path / "seed", "parent")
        specification = replace(specification, revision=verify_harness(candidate))
    if change == "drift":
        policy = program(candidate, profile, tmp_path / "workers")
        atomic_json(tmp_path / "seed/skills.json", skill(1))
        with pytest.raises(ValueError, match="Changed or missing"):
            policy.begin_episode("counter", {}, {}, context(policy))
    else:
        with pytest.raises(ValueError):
            program(candidate, profile, tmp_path / "workers", specification=specification)
    assert not (tmp_path / "workers").exists()
