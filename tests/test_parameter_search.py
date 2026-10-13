"""Verified CPU integration plus isolated numerical-search contract tests."""
from copy import deepcopy
from pathlib import Path
import shutil

import pytest

from PhysicalRSI_core.infra.storage import atomic_json, digest, file_digest, read_json
from PhysicalRSI_core.self_harness.archive import CandidateArchive
from PhysicalRSI_core.self_harness.artifacts import CLOSURE, verify_harness
from PhysicalRSI_core.self_harness.parameter_search import ScalarParameter, ScalarRefinementEvolver
from PhysicalRSI_demos.parameter_refinement import setup, Feedback
from PhysicalRSI_demos.recursive_research import manifest


def test_rejected_cpu_probe_drives_next_round_without_validation_feedback(tmp_path):
    campaign, initial, suite, quota = setup(tmp_path)
    result = campaign.run(initial)
    report = campaign.report_test(initial)
    rounds = tmp_path/'campaign/rounds'
    fits = [read_json(p) for p in sorted(rounds.glob('*/execution/search-fit.json'))]
    assert [r['outcome'] for r in result['rounds']] == ['all_candidates_rejected', 'inherit_child']
    assert [f['proposed_value'] for f in fits] == [5, 7]
    assert all(s['source'] == 'current_development' for s in fits[0]['samples'])
    rejected = [s for s in fits[1]['samples'] if s['value'] == 5]
    assert {s['task'] for s in rejected} == {'small-root', 'large-root'}
    assert all(s['classification'] == 'insufficient' and s['source']['role'] == 'development_screening'
               for s in rejected)
    assert all(s['case']['number'] in (2,3) for f in fits for s in f['samples'])
    assert all(v['success_rate'] == 1 for v in report['metrics'].values())
    assert len(suite.resets) == quota.status()['reserved_trials'] == 18
    archive = CandidateArchive(tmp_path/'archive')
    record = archive.read(next((archive.root/'entries').glob('*/record.json')).parent.name)
    view = archive.view(scope=record['scope'],policy_sha256=record['policy_sha256'])
    rows = archive.development(view)
    assert len(rows) == 8
    assert {r['role'] for r in rows} == {'incumbent_development', 'development_screening'}
    assert all(r['evaluation']['trial']['system2']['split'] == 'development' for r in rows)
    assert all('/archive/entries/' in p for r in rows for p in r['evaluation']['evidence'])
    assert campaign.run(initial) == result and campaign.report_test(initial) == report
    assert len(suite.resets) == 18
    # Owned observations still work when original executions are gone.
    shutil.rmtree(campaign.root)
    assert archive.development(view) == rows
    receipt = next(p for p in rows[0]['evaluation']['evidence'] if p.endswith('/receipt.json'))
    Path(receipt).write_text('{}')
    with pytest.raises(ValueError):
        archive.development(view)


class UnitArchive:
    """Isolate bracket mathematics; this fixture is not rollout evidence."""
    def __init__(self, rows):
        self.rows = rows

    def identity(self):
        return dict(kind='unit-archive')

    def development(self, view):
        return deepcopy(self.rows)


def candidate(root, value, *, offset=0):
    for kind in CLOSURE:
        atomic_json(root/(kind+'.json'), dict(iterations=value, offset=offset) if kind == 'control' else dict(kind=kind))
    return manifest(root, root.name)


def sample(harness, outcome='failure', *, case=1, split='development'):
    trial = dict(task='unit-root', case=dict(number=case), scope='software',
        system2=dict(split=split,candidate_id=harness['id'],candidate_sha256=verify_harness(harness)))
    return dict(trial=trial,trial_sha256=digest(trial),outcome=outcome,
        verdict=dict(measurements=dict(residual=1 if outcome=='failure' else 0)),evidence={})


def search(tmp_path, *, current=1, previous=(), direction='increase', upper=9):
    parent = candidate(tmp_path/'parent',current)
    rows=[]
    for index, spec in enumerate(previous):
        value, outcome, case, offset = spec
        child=candidate(tmp_path/('history-'+str(index)),value,offset=offset)
        rows.append(dict(archive_revision=str(index),role='development_screening',candidate=child,
                         evaluation=sample(child,outcome,case=case),archive_evidence={}))
    evolver = ScalarRefinementEvolver(archive=UnitArchive(rows),classifier=Feedback(),
        parameter=ScalarParameter('control','control.json',('iterations',),1,upper,precision=0,direction=direction))
    args=(parent,[sample(parent)],None,dict(view=dict(scope='software')),tmp_path/'proposal')
    return evolver,args


def test_rejected_probe_changes_fit_and_materialization_is_idempotent(tmp_path):
    evolver,args=search(tmp_path,previous=[(5,'failure',1,0)])
    children=evolver.propose_from_archive(*args)
    assert read_json(Path(children[0]['root'])/'control.json')['iterations']==7
    assert evolver.resume_proposal_from_archive(*args)==children
    assert read_json(Path(args[0]['root'])/'control.json')['iterations']==1
    assert set(children[0]['changes'])=={'control'}
    assert children[0]['parent_sha256']==verify_harness(args[0])


@pytest.mark.parametrize('boundary',['fit','rename'])
def test_interrupted_scalar_publication_resumes_exact_bytes(tmp_path,monkeypatch,boundary):
    from PhysicalRSI_core.self_harness import parameter_search as module, search_space
    evolver,args=search(tmp_path)
    original=module.freeze_json if boundary=='fit' else search_space.os.rename
    def interrupted(*a,**kw):
        original(*a,**kw)
        raise RuntimeError('after durable publication')
    with monkeypatch.context() as patch:
        patch.setattr(module if boundary=='fit' else search_space.os,'freeze_json' if boundary=='fit' else 'rename',interrupted)
        with pytest.raises(RuntimeError,match='durable publication'):
            evolver.propose_from_archive(*args)
    result=evolver.resume_proposal_from_archive(*args)
    assert read_json(Path(result[0]['root'])/'control.json')['iterations']==5
    assert evolver.resume_proposal_from_archive(*args)==result


@pytest.mark.parametrize('previous,expected,reason',[
    ([(5,'failure',1,0),(7,'success',1,0)],6,'refine_observed_failure_bound'),
    ([(5,'failure',1,0),(5,'success',1,0)],None,'contradictory_case_feedback'),
    ([(9,'failure',1,0)],None,'declared_bound_exhausted'),
    ([(9,'failure',2,0)],5,'refine_observed_failure_bound'),
    ([(9,'failure',1,1)],5,'refine_observed_failure_bound'),
])
def test_case_conditioned_brackets_and_abstention(tmp_path,previous,expected,reason):
    evolver,args=search(tmp_path,previous=previous)
    result=evolver.propose_from_archive(*args)
    report=read_json(args[-1]/'search-fit.json')
    assert report['proposed_value']==expected and report['reason']==reason
    assert bool(result)==(expected is not None)
    if previous[0][-1]:
        assert report['excluded'][0]['reason']=='other_controller_inputs_changed'


def test_decreasing_parameter_uses_directed_bounds(tmp_path):
    evolver,args=search(tmp_path,current=9,previous=[(5,'failure',1,0)],direction='decrease')
    result=evolver.propose_from_archive(*args)
    assert read_json(Path(result[0]['root'])/'control.json')['iterations']==3


@pytest.mark.parametrize('outcome',['success','unknown','timeout'])
def test_no_identified_failure_does_not_materialize_candidate(tmp_path,outcome):
    evolver,args=search(tmp_path)
    args[1][0]['outcome']=outcome
    assert evolver.propose_from_archive(*args)==[]
    assert not (args[-1]/'parameter-candidate').exists()


@pytest.mark.parametrize('damage',['split','candidate','hash','scope','trial'])
def test_wrong_attribution_is_rejected_before_fit(tmp_path,damage):
    evolver,args=search(tmp_path)
    value=args[1][0]
    if damage=='scope':value['trial']['scope']='foreign'
    elif damage=='trial':value['trial_sha256']='0'*64
    else:value['trial']['system2'][{'split':'split','candidate':'candidate_id','hash':'candidate_sha256'}[damage]]='wrong'
    if damage!='trial':value['trial_sha256']=digest(value['trial'])
    with pytest.raises(ValueError,match='attributed development'):
        evolver.propose_from_archive(*args)
    assert not (args[-1]/'search-fit.json').exists()


@pytest.mark.parametrize('options',[
    dict(component='weights'),dict(lower=float('nan')),dict(upper=True),dict(precision=True),
    dict(precision=-1),dict(direction='both'),dict(file='../control.json'),dict(key=()),dict(upper=1),
])
def test_invalid_search_spaces_fail(options):
    declared=dict(component='control',file='control.json',key=('iterations',),lower=1,upper=9)
    with pytest.raises((ValueError,TypeError)):
        ScalarParameter(**(declared|options))


def test_changed_provider_and_existing_candidate_are_not_reused(tmp_path):
    evolver,args=search(tmp_path)
    result=evolver.propose_from_archive(*args)
    path=Path(result[0]['root'])/'control.json'
    path.write_text('{}')
    with pytest.raises(ValueError,match='Changed or missing'):
        evolver.resume_proposal_from_archive(*args)
    evolver.classifier.identity=lambda:dict(kind='different')
    with pytest.raises(ValueError,match='implementation changed'):
        evolver.resume_proposal_from_archive(*args)
