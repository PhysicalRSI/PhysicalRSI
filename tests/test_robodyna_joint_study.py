"""CPU joint orchestration contracts; no benchmark claims."""
import importlib.util
from pathlib import Path
import pytest
from PhysicalRSI_core.infra.storage import read_json

from examples.robodyna import native_study as joint
import test_robodyna_native_study as helpers


def test_both_task_plan_profile_and_quota(tmp_path):
    args = helpers.arguments()
    args.update(evolution_tasks=(joint.RAMP,joint.VALLEY), proposal_trigger='prediction-calibration',
                development=(301,302), validation=(401,402,403), heldout=(10003,10004,10005))
    campaign, parent, baseline, quota, config = joint.setup(tmp_path, **args)
    prepared = campaign.build_loop(tmp_path/'prepare', campaign._context(0,'0'*64,campaign.initial_memory,None))
    assert set(prepared.config['profile']['tasks']) == {joint.RAMP,joint.VALLEY}
    assert all(row['maximum_success_regression'] == 0 for row in prepared.config['profile']['tasks'].values())
    assert quota.max_trials == 34


def test_joint_successful_parent_still_compares_calibration_and_rejects_regression(tmp_path):
    donor = tmp_path/'donor'
    args = helpers.arguments()
    joint.run(donor, **args)
    output=tmp_path/'joint'
    result=joint.run_joint(output, predecessor_root=donor, suite=helpers.ContractSuite(),
        candidate_parameters={'gain':.5}, development=(301,), validation=(401,), heldout=(10003,),
        seconds=10,max_steps=1)
    assert result['same_A1_A2']
    config=read_json(output/'config.json')
    assert config['parent']=={'gain':1.}
    assert config['predecessor']['inherited_parameters_sha256']
    folder=output/'campaign/rounds/round-0001/execution'
    assert (folder/'candidate/dynamic_interception.json').exists()
    plans = list((folder/'experiments/validation').rglob('plan.json'))
    assert len(plans) == 2
    assert all(set(read_json(p)['cases']) == {joint.RAMP,joint.VALLEY} for p in plans)
    assert joint.run_joint(output, predecessor_root=donor,suite=helpers.ContractSuite(),
        candidate_parameters={'gain':.5},development=(301,),validation=(401,),heldout=(10003,),seconds=10,max_steps=1)==result
    with pytest.raises(ValueError,match='fresh'):
        joint.run_joint(tmp_path/'leaked',predecessor_root=donor,suite=helpers.ContractSuite(),development=(11,))
