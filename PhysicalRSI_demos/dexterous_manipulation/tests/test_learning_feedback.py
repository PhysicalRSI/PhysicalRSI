"""Check failure learning without weakening native selection or case separation."""
import json
from pathlib import Path

import pytest

from PhysicalRSI_core.embodiment import Revision
from PhysicalRSI_core.experiments import Budget
from PhysicalRSI_core.infra.trial_quota import TrialQuota
from PhysicalRSI_core.lineage import HarnessState
from PhysicalRSI_core.self_harness import SelfHarness
from PhysicalRSI_core.self_harness.campaign import ImprovementCampaign
from PhysicalRSI_core.self_harness.evaluation import ExperimentEvaluator
from PhysicalRSI_core.self_harness.research_memory import ResearchMemory

from PhysicalRSI_demos.dexterous_manipulation.common import file_hash, save
from PhysicalRSI_demos.dexterous_manipulation.dexjoco_campaign import Selector, freeze_candidate
from PhysicalRSI_demos.dexterous_manipulation.dexjoco_loop import CampaignCases, CampaignSuite
from PhysicalRSI_demos.dexterous_manipulation import research_feedback
from PhysicalRSI_demos.dexterous_manipulation.research_feedback import build_research_feedback, freeze_imports, inspect_round
from PhysicalRSI_demos.dexterous_manipulation.task_diagnostics import diagnostic_inputs, summarize
from test_dexjoco_loop import FixtureSuite


def test_photo_diagnosis_requires_simultaneous_conditions():
    rows=[dict(region_pass=True,angle_pass=False,shutter_pressed=False,index_shutter_pressed=False),
          dict(region_pass=False,angle_pass=True,shutter_pressed=True,index_shutter_pressed=True)]
    report=summarize("bimanual_photograph",rows)
    assert report["measurements"]["simultaneous_pass_steps"]==0
    assert report["observed_bottlenecks"]==["conditions_satisfied_at_different_times"]
    rows.append(dict(region_pass=True,angle_pass=True,shutter_pressed=True,index_shutter_pressed=False))
    assert summarize("bimanual_photograph",rows)["measurements"]["simultaneous_pass_steps"]==1


def test_assembly_diagnosis_distinguishes_end_alignment_and_stability():
    row=dict(step=1,bottom_contact=False,stable_count=0,tip_xy=[0.,0.],tip_z=.35,
             other_xy=[0.,0.],other_z=.126,axis_degrees=177.,rim=.126,bottom=.0155,clearance=.0024)
    report=summarize("bimanual_assembly",[row])
    assert "opposite_end_presented_near_socket" in report["observed_bottlenecks"]
    assert report["measurements"]["correct_tip_near_socket_steps"]==0
    aligned=dict(row,tip_xy=[.005,.003],tip_z=.126,axis_degrees=5.)
    assert "closest_tip_offset_exceeds_nominal_clearance" in summarize("bimanual_assembly",[aligned])["observed_bottlenecks"]
    contact=dict(aligned,bottom_contact=True,stable_count=1)
    assert "bottom_contact_not_sustained_for_30_steps" in summarize("bimanual_assembly",[contact])["observed_bottlenecks"]


@pytest.mark.parametrize("case,outcome",[(dict(seed=1,split="development"),"failure"),
                                        (dict(seed=999,split="test"),"failure"),
                                        (dict(seed=999,split="development"),"interrupted")])
def test_diagnostics_reject_final_test_or_uncertain_episodes(tmp_path,case,outcome):
    save(tmp_path/"experiment.json",dict(case=case))
    save(tmp_path/"receipt.json",dict(outcome=outcome))
    with pytest.raises(ValueError,match="completed, non-final"):
        diagnostic_inputs(tmp_path/"trajectory.json",tmp_path)


class FailedProposal:
    def __init__(self,evaluator): self.evaluator=evaluator
    def identity(self): return dict(name="deliberately-failing-software-fixture",revision="1")
    def develop(self,parent,output): return self.evaluator.development(parent,output)
    def propose(self,parent,feedback,output):
        return [freeze_candidate(output/"candidate","failed-child",parent=parent,
                memory="4\nThis fixture changes behavior and still fails.",provenance=feedback["evidence"])]


def failed_campaign(root):
    cases=CampaignCases(root/"case_protocol",tasks=["fixture"],rounds=1,
        development_seed=3103,validation_seed=3203,test_seed=3303,seed_stride=1000)
    parent=freeze_candidate(root/"candidates/parent","parent",memory="2\nFailed fixture parent.")
    state=HarnessState(root/"lineage")
    quota=TrialQuota(root/"quota",max_trials=3)
    def factory(directory):
        evaluator=ExperimentEvaluator(suite=CampaignSuite(FixtureSuite(directory),cases,directory.name),
            budgets={"fixture":Budget(5,60)},scope="software-inheritance-fixture",
            system2=Revision("test-fixture","1"),quota=quota)
        return SelfHarness(directory,state=state,proposer=FailedProposal(evaluator),evaluator=evaluator,
            selector=Selector(),profile=dict(tasks=dict(fixture=dict(weight=1,episodes=1,
                score_range=[0,1],maximum_regression=0)),minimum_gain=0,tie_tolerance=0),
            protocol=dict(identity=evaluator.revision,evaluation_kind="local_contract_evaluation"),
            scope="software-inheritance-fixture",max_candidates=1)
    result=ImprovementCampaign(root,state=state,build_loop=factory,max_rounds=1).run(parent)
    assert result["current"]["harness"]["id"]=="parent"
    return root/"rounds/round-0001"


def test_rejected_candidate_is_retrieved_without_promotion_and_imports_forbid_reused_cases(tmp_path,monkeypatch):
    root=tmp_path/"old"
    directory=failed_campaign(root)
    inspected=inspect_round(directory)
    assert inspected["selection"]["decision"]=="retain_parent"
    with pytest.raises(ValueError,match="overlaps future"):
        freeze_imports(tmp_path/"bad",[directory],{3203})
    new=tmp_path/"new"
    freeze_imports(new,[directory],{7103,7203,7303})
    def fixture_diagnose(trajectory,source,output):
        result=dict(binding=dict(inputs={str(trajectory):file_hash(trajectory)}),
                    summary=dict(observed_bottlenecks=["software-fixture-only"]))
        save(output,result)
        return result
    monkeypatch.setattr(research_feedback,"diagnose_trajectory",fixture_diagnose)
    output=new/"rounds/round-0001"
    packet=build_research_feedback(new,output,source=tmp_path,tasks=["fixture"])
    assert len(packet["lessons"])==2
    child=next(v for v in packet["lessons"].values() if v["observation"]["candidate_id"]=="failed-child")
    assert child["observation"]["native_success"] is False
    assert child["observation"]["was_selected"] is False
    assert child["observation"]["rejection"]["reason"]=="no_sufficient_improvement"
    store=ResearchMemory(output/"research_memory")
    assert not store.retrieve(packet["revision"],dict(task="unrelated",environment="dexjoco",track="astra-direct"))
    assert store.read(packet["revision"])["qualification"] is None
    assert HarnessState(root/"lineage").resolve()["harness"]["id"]=="parent"
    evidence=str(directory/"selection.json")
    Path(evidence).write_text("changed")
    with pytest.raises(ValueError,match="evidence changed"):
        store.read(packet["revision"])
    with pytest.raises(ValueError,match="historical comparison changed"):
        build_research_feedback(new,output,source=tmp_path,tasks=["fixture"])


def test_research_history_automatically_reads_completed_campaign_rounds(tmp_path,monkeypatch):
    directory=failed_campaign(tmp_path)
    def fixture_diagnose(trajectory,source,output):
        result=dict(binding=dict(inputs={str(trajectory):file_hash(trajectory)}),summary={})
        save(output,result)
        return result
    monkeypatch.setattr(research_feedback,"diagnose_trajectory",fixture_diagnose)
    packet=build_research_feedback(tmp_path,tmp_path/"rounds/round-0002",source=tmp_path,tasks=["fixture"])
    assert {v["observation"]["candidate_id"] for v in packet["lessons"].values()}=={"parent","failed-child"}
    with pytest.raises(ValueError,match="own reflection history"):
        build_research_feedback(tmp_path,directory,source=tmp_path,tasks=["fixture"])
