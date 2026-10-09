"""Software fixtures verify inheritance; they are not robot task results."""
import json
from pathlib import Path
import uuid

import pytest
from PhysicalRSI_core.contracts import Contract
from PhysicalRSI_core.embodiment import Embodiment, Revision, System1
from PhysicalRSI_core.experiments import Budget
from PhysicalRSI_core.infra.trial_quota import TrialQuota
from PhysicalRSI_core.lineage import HarnessState
from PhysicalRSI_core.self_harness import SelfHarness
from PhysicalRSI_core.self_harness.artifacts import verify_harness
from PhysicalRSI_core.self_harness.campaign import ImprovementCampaign
from PhysicalRSI_core.self_harness.evaluation import ExperimentEvaluator

from PhysicalRSI_demos.dexterous_manipulation.common import save
from PhysicalRSI_demos.dexterous_manipulation.dexjoco_campaign import CaseEnvironment, Selector, freeze_candidate, verify_executing_source
from PhysicalRSI_demos.dexterous_manipulation.dexjoco_loop import CampaignCases, CampaignSuite, verify_loaded_rounds
from PhysicalRSI_demos.dexterous_manipulation.dexjoco_sim import DexJoCoProfile
from PhysicalRSI_demos.dexterous_manipulation.candidate_programs import validate_programs
from PhysicalRSI_core.contracts import Context
from PhysicalRSI_demos.dexterous_manipulation.dexjoco_sim import DexJoCoEnvironment


def test_campaign_schedule_fixes_identity_and_rejects_seed_reuse(tmp_path):
    args=dict(tasks=["fixture"],rounds=2,development_seed=3103,validation_seed=3203,test_seed=3303,seed_stride=1000)
    cases=CampaignCases(tmp_path/"plan",**args)
    class Base:
        def identity(self): return dict(name="unchanged-runtime")
    first=CampaignSuite(Base(),cases,"round-0001")
    second=CampaignSuite(Base(),cases,"round-0002")
    assert first.identity()==second.identity()
    assert first.development_cases({})["fixture"][0]["seed"]==3103
    assert second.development_cases({})["fixture"][0]["seed"]==4103
    with pytest.raises(ValueError,match="distinct non-final"):
        CampaignCases(tmp_path/"overlap",**dict(args,seed_stride=100))
    with pytest.raises(ValueError,match="different scheduled round"):
        second.validation_cases(dict(round_id="round-0001"))
    data=json.loads((tmp_path/"plan/schedule.json").read_text())
    data["rounds"]["round-0002"]["development"]["fixture"][0]["seed"]=9999
    save(tmp_path/"plan/schedule.json",data)
    with pytest.raises(ValueError,match="cases changed"):
        second.identity()


FIXTURE_CONTRACT=Contract("inheritance-fixture-counter",unit="count",embodiment="software")


class FixtureEnvironment:
    def __init__(self,root): self.root=Path(root)
    def identity(self): return dict(name="software-inheritance-fixture",revision="1")
    def describe(self): return Embodiment("software", "1", "software",FIXTURE_CONTRACT,FIXTURE_CONTRACT)
    def observe(self):
        path=self.root/"observations"/uuid.uuid4().hex/"observation.json"
        observation=dict(position=self.position,observation_path=str(path))
        save(path,observation)
        return observation
    def reset(self,case,context):
        self.position=0
        return dict(ready=True,observation=self.observe())
    def step(self,action,context):
        self.position+=action
        return dict(observation=self.observe(),terminated=self.position>=3)


class FixturePolicy:
    def __init__(self,candidate,root):
        self.candidate,self.root=candidate,root
    def identity(self): return dict(name="software-fixture",freeze_sha256=verify_harness(self.candidate))
    def describe(self): return System1("software-fixture",verify_harness(self.candidate),FIXTURE_CONTRACT,FIXTURE_CONTRACT)
    def begin_episode(self,task,case,observation,context):
        # Exercise the same artifact loading as the real GPT-as-Policy runtime.
        profile=DexJoCoProfile(memory=Path(self.candidate["root"])/"memory.md")
        profile.skill_root=Path(self.candidate["root"])/"skill"
        profile.candidate=self.candidate
        (self.root/"galbot").mkdir()
        agent=profile.prepare_workspace(self.root/"galbot")
        self.increment=int((agent/"context/candidate_memory.md").read_text().splitlines()[0])
    def act(self,observation,context): return self.increment


class FixtureVerifier:
    def identity(self): return dict(name="software-counter-verifier",revision="1")
    def verify(self,case,trace):
        measured=trace[-1]["observation"]["position"]
        return dict(outcome="success" if measured==3 else "failure",reason="Software fixture only",
                    measurements=dict(position=measured))


class FixtureSuite:
    def __init__(self,root): self.root=root
    def identity(self): return dict(name="software-fixture-suite",revision="1")
    def admit(self,candidate):
        verify_executing_source(candidate)
        return dict(accepted=True,evidence=dict(closure=verify_harness(candidate)))
    def environment(self,task):
        self.live=self.root/"live"/task/uuid.uuid4().hex
        return FixtureEnvironment(self.live)
    def policy(self,candidate,task): return FixturePolicy(candidate,self.live)
    def verifier(self,task): return FixtureVerifier()


class FixtureProposal:
    def __init__(self,evaluator): self.evaluator=evaluator
    def identity(self): return dict(name="scripted-unit-test-proposal",revision="1")
    def develop(self,parent,output): return self.evaluator.development(parent,output)
    def propose(self,parent,feedback,output):
        return [freeze_candidate(output/"candidate","child-"+output.name,parent=parent,
            memory="1\nSoftware fixture "+output.name,provenance=feedback["evidence"],
            programs=[dict(name="fixture_helper.py",source="def increment():\n    return 1\n",
                           usage="Software fixture only: increment returns one.")])]


def test_actual_core_campaign_loads_selected_memory_next_round_and_resumes(tmp_path):
    root=tmp_path/"campaign"
    cases=CampaignCases(root/"case_protocol",tasks=["fixture"],rounds=2,
        development_seed=3103,validation_seed=3203,test_seed=3303,seed_stride=1000)
    parent=freeze_candidate(root/"candidates/parent","parent",memory="2\nSoftware fixture")
    state=HarnessState(root/"lineage")
    quota=TrialQuota(root/"quota",max_trials=6)
    def factory(directory):
        evaluator=ExperimentEvaluator(suite=CampaignSuite(FixtureSuite(directory),cases,directory.name),
            budgets={"fixture":Budget(5,60)},scope="software-inheritance-fixture",
            system2=Revision("test-fixture","1"),quota=quota)
        return SelfHarness(directory,state=state,proposer=FixtureProposal(evaluator),evaluator=evaluator,
            selector=Selector(),profile=dict(tasks=dict(fixture=dict(weight=1,episodes=1,
                score_range=[0,1],maximum_regression=0)),minimum_gain=0,tie_tolerance=0),
            protocol=dict(identity=evaluator.revision,evaluation_kind="local_contract_evaluation"),
            scope="software-inheritance-fixture",max_candidates=1)
    campaign=ImprovementCampaign(root,state=state,build_loop=factory,max_rounds=2)
    result=campaign.run(parent)
    assert [r["outcome"] for r in result["rounds"]]==["inherited","retained"]
    report=verify_loaded_rounds(root)
    assert report["verified_next_round_loads"]==1
    assert report["rounds"][1]["parent_id"]=="child-round-0001"
    assert report["rounds"][1]["parent_revision"]==report["rounds"][0]["result_revision"]
    loaded=json.loads(Path(report["rounds"][1]["loaded"][0]["receipt"]).read_text())
    assert set(loaded["program_sha256"])=={"fixture_helper.py"}
    assert quota.status()["reserved_trials"]==6
    assert campaign.run(parent)==result
    assert quota.status()["reserved_trials"]==6
    # Audit detects changed workspace evidence instead of silently replaying.
    receipt=Path(report["rounds"][1]["loaded"][0]["receipt"])
    receipt.write_text("{}")
    with pytest.raises(ValueError,match="round changed"):
        campaign.run(parent)


def test_closed_runtime_detaches_auth_links_without_modifying_target(tmp_path):
    secret=tmp_path/"login/auth.json"
    secret.parent.mkdir()
    secret.write_text("unit-test-placeholder")
    audit=tmp_path/"audit"
    link=audit/"private_codex_home/auth.json"
    link.parent.mkdir(parents=True)
    link.symlink_to(secret)
    DexJoCoProfile().cleanup_workspace(audit)
    assert not link.is_symlink()
    assert secret.read_text()=="unit-test-placeholder"
    link.write_text("refreshed-private-fixture")
    DexJoCoProfile().cleanup_workspace(audit)
    assert not link.exists() and secret.read_text()=="unit-test-placeholder"


@pytest.mark.parametrize("program",[
    dict(name="../escape.py",source="x=1",usage="bad path"),
    dict(name="invalid.py",source="return 1",usage="invalid module"),
])
def test_program_candidate_rejects_invalid_paths_and_code(program):
    with pytest.raises(ValueError):
        validate_programs([program])


def test_program_admission_does_not_execute_submitted_code(tmp_path):
    sentinel=tmp_path/"must-not-exist"
    # Syntax admission must never execute top-level statements. This fixture is
    # intentionally unsuitable for a policy helper; it is never deployed.
    source=f"from pathlib import Path\nPath({str(sentinel)!r}).touch()\n"
    validate_programs([dict(name="fixture.py",source=source,usage="Admission isolation test")])
    assert not sentinel.exists()


def test_paired_reset_checks_actual_images_and_detects_disguised_scene_reuse(tmp_path,monkeypatch):
    packet=dict(state=[1.],images=[dict(camera="base",sha256="1"*64)],
                camera_calibration=dict(base="fixed"),observation_path="fixture-only")
    monkeypatch.setattr(DexJoCoEnvironment,"reset",lambda *a,**kw:dict(ready=True,observation=packet))
    env=object.__new__(CaseEnvironment)
    env.initial_reference_root=tmp_path
    env.task="fixture"
    assert env.reset(dict(seed=3103),Context("first"))["ready"]
    assert env.reset(dict(seed=3103),Context("paired"))["ready"]
    assert not env.reset(dict(seed=4103),Context("duplicate-scene"))["ready"]
    packet["images"][0]["sha256"]="2"*64
    assert not env.reset(dict(seed=3103),Context("mismatched-pair"))["ready"]
    assert env.reset(dict(seed=4103),Context("new-scene"))["ready"]
