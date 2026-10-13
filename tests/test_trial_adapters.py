from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
from threading import Barrier

import pytest

from PhysicalRSI_core.embodiment import Revision
from PhysicalRSI_core.experiments import Budget, TrialAdapters, trial_adapters
from PhysicalRSI_core.infra.storage import atomic_json, file_digest, read_json
from PhysicalRSI_core.infra.trial_quota import TrialQuota
from PhysicalRSI_core.self_harness.artifacts import CLOSURE, verify_harness
from PhysicalRSI_core.self_harness.evaluation import ExperimentEvaluator
from PhysicalRSI_core.self_harness.protocol import CaseProtocol, PreregisteredSuite
from PhysicalRSI_core.self_harness.research_memory import MemoryBoundSuite, ResearchMemory, ResearchMemoryGate
from PhysicalRSI_demos.auto_engineering.harness import create_seed
from PhysicalRSI_demos.auto_engineering.task import TASK
from PhysicalRSI_demos.auto_engineering.trials import Suite as IsaacSuite
from PhysicalRSI_demos.recursive_research import manifest
from PhysicalRSI_demos.rollout_evolution import Environment, Policy, Verifier


def candidate(root,name,coefficient):
    for kind in CLOSURE:
        atomic_json(root/(kind+'.json'),dict(axis=coefficient) if kind=='control' else dict(kind=kind))
    return manifest(root,name)


class BundleSuite:
    """No separate environment/policy/verifier factories exist."""
    def __init__(self,barrier=None):
        self.barrier,self.resets = barrier,[]

    def identity(self):
        return dict(kind='test-bundle-suite')

    def trial(self,candidate,task):
        environment=Environment(task,1.,self.resets)
        policy=Policy(task,verify_harness(candidate),read_json(Path(candidate['root'])/'control.json')[task])
        class BoundVerifier(Verifier):
            def verify(self,case,trace):
                assert environment.position==trace[-1]['observation']['position']
                return super().verify(case,trace)
        if self.barrier is not None:self.barrier.wait(timeout=10)
        return TrialAdapters(environment,policy,BoundVerifier())


def evaluator(root,suite):
    protocol=CaseProtocol(root/'cases',splits={split:{'axis':[dict(seed=target,initial=[0],target=[target])]}
        for split,target in (('development',1),('validation',3),('test',7))})
    return ExperimentEvaluator(suite=PreregisteredSuite(suite,protocol),budgets={'axis':Budget(1,30)},
        scope='software',system2=Revision('bundle-test','1'),quota=TrialQuota(root/'quota',max_trials=2))


def test_concurrent_trials_bind_independent_stateful_adapters_without_legacy_factories(tmp_path):
    suite=BundleSuite(Barrier(2));engine=evaluator(tmp_path,suite)
    candidates=[candidate(tmp_path/'good','good',1.),candidate(tmp_path/'bad','bad',0.)]
    def execute(h):return engine.development(h,tmp_path/('run-'+h['id']))
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(execute,candidates))
    assert [r['episodes'][0]['outcome'] for r in results]==['success','failure']
    assert len(suite.resets)==2
    for h,result in zip(candidates,results):
        value=result['evaluations'][0]
        assert value['trial']['system2']['candidate_sha256']==verify_harness(h)
        assert value['receipt']['binding']['system1']['revision']==verify_harness(h)
    assert engine.quota.status()['reserved_trials']==2


@pytest.mark.parametrize('mode',['bundle','legacy'])
def test_isaac_construction_keeps_policy_and_verifier_bound_to_their_own_environment(tmp_path,mode):
    suite=IsaacSuite(sys.executable,tmp_path/'native');seed=create_seed(tmp_path/'seed');barrier=Barrier(2)
    def build(index):
        if mode=='bundle':
            ports=trial_adapters(suite,seed,TASK)
            barrier.wait(timeout=10)
        else:
            environment=suite.environment(TASK);barrier.wait(timeout=10)
            ports=TrialAdapters(environment,suite.policy(seed,TASK),suite.verifier(TASK))
        assert ports.policy.environment is ports.environment
        assert ports.verifier.environment is ports.environment
        assert ports.environment.worker.process is None
        return ports
    with ThreadPoolExecutor(max_workers=2) as pool:
        left,right=pool.map(build,range(2))
    assert left.environment is not right.environment and left.environment.worker is not right.environment.worker
    assert not (tmp_path/'native').exists()


def test_factory_contract_change_is_rejected_before_effects(tmp_path):
    suite=BundleSuite();engine=evaluator(tmp_path,suite)
    suite.trial=None
    with pytest.raises(ValueError,match='identity changed'):
        engine.development(candidate(tmp_path/'seed','seed',1.),tmp_path/'run')
    assert suite.resets==[] and engine.quota.status()['reserved_trials']==0


def test_memory_wrapper_preserves_bundle_factory_and_identity(tmp_path):
    suite=BundleSuite();seed=candidate(tmp_path/'seed','seed',1.)
    evidence=tmp_path/'diagnostic.json';atomic_json(evidence,{'check':'bundle construction'})
    memory=ResearchMemory(tmp_path/'memory')
    revision=memory.write({'binding':dict(conditions={'task':'axis'},hypothesis='Bind independent adapters',
        required_checks=['binding'],counterexample_checks=[],evidence={str(evidence):file_digest(evidence)})})
    gate=ResearchMemoryGate(memory=memory,revision=revision,context={'task':'axis'},
        parent_sha256=verify_harness(seed),producer='bundle-test',reports={})
    wrapped=MemoryBoundSuite(suite,gate)
    before=wrapped.identity(),wrapped.policy_identity()
    ports=trial_adapters(wrapped,seed,'axis')
    assert isinstance(ports,TrialAdapters) and suite.resets==[]
    suite.trial=None
    assert wrapped.identity()!=before[0] and wrapped.policy_identity()!=before[1]


@pytest.mark.parametrize('bad_factory',[False,lambda *args: {'environment':'not-a-bundle'}])
def test_malformed_bundle_factory_is_rejected(tmp_path,bad_factory):
    suite=BundleSuite();suite.trial=bad_factory
    with pytest.raises(ValueError,match='factory'):
        trial_adapters(suite,candidate(tmp_path/'seed','seed',1.),'axis')
    assert suite.resets==[]


def test_adapter_constructor_cannot_change_candidate_before_reset(tmp_path):
    class MutatingSuite(BundleSuite):
        def trial(self,h,task):
            ports=super().trial(h,task)
            atomic_json(Path(h['root'])/'control.json',dict(axis=99.))
            return ports
    suite=MutatingSuite();engine=evaluator(tmp_path,suite)
    with pytest.raises(ValueError,match='Changed or missing harness input'):
        engine.development(candidate(tmp_path/'seed','seed',1.),tmp_path/'run')
    assert suite.resets==[]
