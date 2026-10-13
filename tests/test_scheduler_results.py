from pathlib import Path

import pytest

from PhysicalRSI_core.contracts import ReconciliationRequired
from PhysicalRSI_core.infra.scheduler import JobScheduler
from PhysicalRSI_core.infra.storage import atomic_json, read_json


class Handler:
    def __init__(self,root):
        self.root=Path(root);self.calls=0;self.collections=0;self.revision=1
        self.change_on=None;self.fail_after_write=False;self.alias=False

    def identity(self):
        return dict(kind='result-boundary-test',revision=self.revision)

    def execute(self,payload):
        self.calls+=1
        atomic_json(self.root/(payload['name']+'.json'),dict(score=1))
        self.cached=dict(score=0 if self.alias else 1)
        if self.fail_after_write:raise RuntimeError('Lost acknowledgment after writing complete evidence')
        return self.cached

    def collect(self,payload):
        self.collections+=1
        if self.collections==self.change_on:self.revision+=1
        value=read_json(self.root/(payload['name']+'.json'))
        if self.alias:
            self.cached.update(value)
            return self.cached
        return value


def setup(root):
    queue=JobScheduler(root/'queue',capacities={'cpu':1});handler=Handler(root/'effects')
    ids=queue.submit([dict(name='first')],handler=handler,resources={'cpu':1})
    return queue,handler,ids


def test_shared_result_mutation_cannot_hide_worker_collector_disagreement(tmp_path):
    queue,handler,ids=setup(tmp_path);handler.alias=True
    with pytest.raises(ValueError,match='output differs'):
        queue.work_once(handler)
    status=queue.status();job=status['jobs'][ids[0]]
    assert job['state']=='needs_reconciliation' and not job['owner_active']
    assert status['used']=={'cpu':1} and handler.calls==1
    # Explicitly collect independently complete evidence; never repeat execute.
    assert queue.recover_completed(ids[0],handler)=={'score':1}
    assert handler.calls==1 and queue.status()['used']=={'cpu':0}
    assert queue.status()['jobs'][ids[0]]['recovered']


@pytest.mark.parametrize('route',['recovery','completed-check','final-collection'])
def test_completed_collector_version_change_is_rejected_without_dispatch(tmp_path,route):
    queue,handler,ids=setup(tmp_path);queue.work_once(handler)
    later=queue.submit([dict(name='later')],handler=handler,resources={'cpu':1})
    handler.change_on=handler.collections+(2 if route=='final-collection' else 1)
    with pytest.raises(ValueError,match='handler identity changed'):
        if route=='recovery':queue.recover_completed(ids[0],handler)
        elif route=='completed-check':queue.run_until(ids+later,handler)
        else:queue.run_until(ids,handler)
    status=queue.status()
    assert status['jobs'][ids[0]]['state']=='completed'
    assert status['jobs'][later[0]]['state']=='queued'
    assert status['used']=={'cpu':0} and handler.calls==1
    assert not (handler.root/'later.json').exists()


def test_uncertain_job_cannot_release_capacity_through_changed_collector(tmp_path):
    queue,handler,ids=setup(tmp_path);handler.fail_after_write=True
    with pytest.raises(RuntimeError,match='Lost acknowledgment'):queue.work_once(handler)
    handler.change_on=1
    with pytest.raises(ReconciliationRequired,match='no verified complete result'):
        queue.recover_completed(ids[0],handler)
    status=queue.status()
    assert status['used']=={'cpu':1} and handler.calls==1
    assert status['jobs'][ids[0]]['state']=='needs_reconciliation'


def test_json_snapshots_survive_provider_and_caller_mutation_and_restart(tmp_path):
    class JsonHandler:
        def identity(self):return {'kind':'json-roundtrip','shape':('x','y')}
        def execute(self,payload):
            assert payload['values']==[1,2]
            payload['values'].append(3)
            return {'values':(1,2,3)}
        def collect(self,payload):
            assert payload['values']==[1,2]
            return {'values':(1,2,3)}
    queue=JobScheduler(tmp_path/'queue',capacities={'cpu':1});handler=JsonHandler()
    payload={'values':[1,2]}
    ids=queue.submit([payload],handler=handler,resources={'cpu':1});payload['values'].append(99)
    result=queue.run_until(ids,handler)
    assert result==[{'values':[1,2,3]}]
    result[0]['values'].append(999)
    reopened=JobScheduler(tmp_path/'queue',capacities={'cpu':1})
    assert reopened.recover_completed(ids[0],handler)=={'values':[1,2,3]}
    assert reopened.status()['jobs'][ids[0]]['request']['payload']=={'values':[1,2]}


def test_nonfinite_provider_result_preserves_readable_queue_and_reserved_capacity(tmp_path):
    queue,handler,ids=setup(tmp_path)
    handler.execute=lambda payload:{'score':float('nan')}
    with pytest.raises(ValueError):queue.work_once(handler)
    status=queue.status()
    assert status['jobs'][ids[0]]['state']=='needs_reconciliation'
    assert status['used']=={'cpu':1} and handler.collections==0


@pytest.mark.parametrize('boundary',['execute','completed'])
def test_result_comparison_uses_json_identity_instead_of_python_numeric_equality(tmp_path,boundary):
    queue,handler,ids=setup(tmp_path)
    handler.execute=lambda payload:{'score':1}
    handler.collect=lambda payload:{'score':1}
    if boundary=='completed':queue.work_once(handler)
    handler.collect=lambda payload:{'score':True}
    with pytest.raises(ValueError,match='output differs|evidence changed'):
        if boundary=='execute':queue.work_once(handler)
        else:queue.recover_completed(ids[0],handler)
    assert queue.status()['jobs'][ids[0]]['state']==('needs_reconciliation' if boundary=='execute' else 'completed')


def test_provider_identity_cannot_change_resource_request_during_submission(tmp_path):
    resources={'cpu':1}
    class MutatingIdentity:
        def identity(self):
            resources['cpu']=0
            return {'kind':'snapshot-submission'}
    queue=JobScheduler(tmp_path/'queue',capacities={'cpu':1})
    ids=queue.submit([{'name':'one'}],handler=MutatingIdentity(),resources=resources)
    assert queue.status()['jobs'][ids[0]]['request']['resources']=={'cpu':1}
