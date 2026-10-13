import copy
import numpy as np
import pytest
from PhysicalRSI_demos.auto_engineering.task import layout,judge
from PhysicalRSI_demos.auto_engineering.policy import plan,propose_repair
from PhysicalRSI_demos.auto_engineering.perception import detect


def test_evaluator_rejects_held_wrong_or_disturbed_sample():
    s=layout(17)
    m=dict(tube_position=[*s['receiver'][:2],.081],tube_orientation_wxyz=[1,0,0,0],
           tube_velocity=[0,0,0],obstacle_position=s['obstacle'],distractor_position=s['distractor'],gripper_opening=.08)
    assert judge(s,m)['success']
    for update in [dict(gripper_opening=.01),dict(tube_position=s['tube']),
                   dict(tube_orientation_wxyz=[.7071,.7071,0,0]),dict(tube_velocity=[.1,0,0]),
                   dict(obstacle_position=[1,1,1]),dict(distractor_position=[1,1,1])]:
        assert not judge(s,dict(m,**update))['success']


def test_layout_replay_and_novelty():
    assert layout(17)==layout(17)
    assert len({tuple(layout(i)['tube']) for i in range(20)})==20


def test_rgbd_detection_ignores_nonfinite_background():
    rgb=np.zeros((20,20,3),dtype=np.uint8);rgb[2:8,2:8]=[10,20,240]
    rgb[10:16,10:16]=[10,200,20];depth=np.ones((20,20));depth[0]=np.nan
    result=detect(rgb,depth,lambda uv,d:np.column_stack([uv*.001,d]))
    assert result['tube']['pixels']==36 and result['receiver']['pixels']==36
    assert result['obstacle']['status']=='not_detected'
    rgb[:]=0
    assert detect(rgb,depth,lambda uv,d:uv)['tube']['status']=='not_detected'


def test_repair_uses_public_height_and_does_not_mutate_parent():
    memory={'transit_height_m':.13,'sample_height_m':.12};original=copy.deepcopy(memory)
    public={name:dict(status='estimated_surface',center_xy=[.4,.2],top_z=z) for name,z in [('tube',.13),('receiver',.02),('obstacle',.18)]}
    feedback={'success':False,'public_observation':public}
    repaired=propose_repair(feedback,memory)
    assert memory==original and repaired['transit_height_m']==.36
    assert plan(public,repaired)['transit_height_m']==.36
    assert propose_repair(dict(feedback,success=True),memory) is None
    public['tube']={'status':'not_detected'}
    with pytest.raises(ValueError):plan(public,memory)


def test_public_cli_help_does_not_import_isaac(capsys):
    from PhysicalRSI.cli import main
    with pytest.raises(SystemExit) as e:main(['auto-engineering','--help'])
    assert e.value.code==0
    assert '--isaac-python' in capsys.readouterr().out


def test_native_evidence_tampering_is_rejected(tmp_path):
    from PhysicalRSI_core.infra.storage import atomic_json,digest,file_digest
    from PhysicalRSI_demos.auto_engineering.harness import verify_episode
    spec=layout(17);atomic_json(tmp_path/'layout.json',spec)
    atomic_json(tmp_path/'result.json',dict(state='completed',backend='isaacsim-5.1',
        layout_sha256=digest(spec),evidence={'layout.json':file_digest(tmp_path/'layout.json')}))
    verify_episode(tmp_path)
    atomic_json(tmp_path/'layout.json',layout(18))
    with pytest.raises(ValueError,match='evidence changed'):verify_episode(tmp_path)


@pytest.mark.parametrize('child_improves', [True, False])
def test_self_harness_selects_only_a_paired_gain(tmp_path,monkeypatch,child_improves):
    # Contract-level test double only: never used as a demo physics backend.
    import sys
    from PhysicalRSI_core.infra.storage import atomic_json,digest,file_digest,read_json
    from PhysicalRSI_demos.auto_engineering import harness,report
    def execute(self,candidate,seed,output):
        output.mkdir(parents=True)
        memory=read_json(Path(candidate['root'])/'memory_rules.json')
        success=bool(child_improves and memory['transit_height_m']>.2)
        result=dict(success=success,layout_sha256=digest(layout(seed)),
            public_observation={'obstacle':dict(status='estimated_surface',top_z=.18)},
            evidence={})
        atomic_json(output/'result.json',result)
        return result
    from pathlib import Path
    monkeypatch.setattr(harness.Evaluation,'execute',execute)
    monkeypatch.setattr(report,'render',lambda root:None)
    result=harness.run(tmp_path,sys.executable,episodes=2)
    assert result['selected_policy']==('clearance-repair' if child_improves else 'initial')
    assert result['qualification'] is None
    state=harness.HarnessState(tmp_path/'state').resolve()
    assert state['revision']==result['selected_revision']
    cases=read_json(tmp_path/'round-001/cases.json')
    assert len(set(cases))==2 and 17 not in cases


def test_grasp_fraction_changes_only_public_pick_depth_and_preserves_legacy_default():
    public={name:dict(status='estimated_surface',center_xy=[.4,.2],top_z=z)
            for name,z in [('tube',.13),('receiver',.02),('obstacle',.18)]}
    memory=dict(transit_height_m=.34,sample_height_m=.12)
    original=plan(public,memory)
    assert plan(public,dict(memory,grasp_depth_fraction=.25))==original
    changed=plan(public,dict(memory,grasp_depth_fraction=.4))
    assert changed['pick']==[.4,.2,pytest.approx(.082)]
    assert {k:v for k,v in changed.items() if k!='pick'}=={k:v for k,v in original.items() if k!='pick'}
    assert memory==dict(transit_height_m=.34,sample_height_m=.12)
