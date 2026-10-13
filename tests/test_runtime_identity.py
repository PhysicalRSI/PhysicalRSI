"""Mutate owned copies of source; never edit the implementation running pytest."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

import PhysicalRSI_core


PROBE = r'''
from pathlib import Path
import json,sys
from PhysicalRSI_core.embodiment import Revision
from PhysicalRSI_core.experiments import Budget,ExperimentRuntime,TrialAdapters
from PhysicalRSI_core.infra.storage import atomic_json
from PhysicalRSI_core.infra.trial_quota import TrialQuota
from PhysicalRSI_core.self_harness.artifacts import CLOSURE
from PhysicalRSI_core.self_harness.evaluation import ExperimentEvaluator
from PhysicalRSI_core.trials import Evaluation
from PhysicalRSI_demos.rollout_evolution import Suite as CalibrationSuite
from PhysicalRSI_demos.recursive_research import manifest
import PhysicalRSI_core

root=Path(sys.argv[1]);target=Path(PhysicalRSI_core.__file__).parent/sys.argv[2];phase=sys.argv[3]
def change():
    target.write_bytes(target.read_bytes()+b'\n# Changed owned dependency after evaluator freeze.\n')
class Suite(CalibrationSuite):
    def development_cases(self,parent):return {'axis':[dict(initial=[0],target=[1])]}
    def trial(self,candidate,task):
        bundle=TrialAdapters(self.environment(task),self.policy(candidate,task),self.verifier(task))
        if phase=='factory':change()
        return bundle
suite=Suite({'axis':1.})
for kind in CLOSURE:
    atomic_json(root/'seed'/(kind+'.json'),dict(axis=1.) if kind=='control' else dict(kind=kind))
seed=manifest(root/'seed','initial');quota=TrialQuota(root/'quota',max_trials=2)
def build():
    return ExperimentEvaluator(suite=suite,budgets={'axis':Budget(1,30)},scope='software',
        system2=Revision('identity-check','1'),quota=quota)
engine=build();before=engine.identity()
if phase=='completed':
    original=engine.development(seed,root/'run')
    receipt=next((root/'run').glob('**/trials/*/receipt.json'))
    change()
    historical=Evaluation.read(ExperimentRuntime(receipt.parent.parent),receipt.parent.name).to_dict()
    assert historical==original['evaluations'][0]
    assert build().revision!=engine.revision
    method=lambda:build().development(seed,root/'run')
else:
    if phase=='before':change()
    method=lambda:engine.development(seed,root/'run')
try:
    method()
except ValueError as error:
    message=str(error)
else:
    raise AssertionError('Changed dependency reached Trial execution under the old evaluator')
print(json.dumps(dict(error=message,identity_changed=engine.identity()!=before,resets=len(suite.resets),
    reserved_trials=quota.status()['reserved_trials'],historical_read_preserved=phase=='completed')))
'''


def probe(tmp_path,dependency,phase):
    root=tmp_path/'code'
    shutil.copytree(Path(PhysicalRSI_core.__file__).parent,root/'PhysicalRSI_core',
                    ignore=shutil.ignore_patterns('__pycache__'))
    repository=Path(__file__).resolve().parents[1]
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(root)+os.pathsep+str(repository))
    result=subprocess.run([sys.executable,'-c',PROBE,str(tmp_path/'workspace'),dependency,phase],
        cwd=root,env=env,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr or result.stdout
    return json.loads(result.stdout)


@pytest.mark.parametrize('dependency',[
    'infra/evidence.py','infra/storage.py','infra/devices.py','contracts.py',
    'embodiment.py','timing.py','self_harness/artifacts.py',
])
def test_owned_dependency_change_blocks_new_effects_before_quota(tmp_path,dependency):
    value=probe(tmp_path,dependency,'before')
    assert value['identity_changed'] and 'identity changed' in value['error']
    assert value['resets']==value['reserved_trials']==0


def test_factory_cannot_change_runtime_dependency_before_reset(tmp_path):
    value=probe(tmp_path,'infra/evidence.py','factory')
    assert value['identity_changed'] and 'identity changed' in value['error']
    assert value['resets']==0 and value['reserved_trials']==1


def test_new_code_cannot_reopen_frozen_plan_but_historical_evidence_remains_readable(tmp_path):
    value=probe(tmp_path,'infra/evidence.py','completed')
    assert value['identity_changed'] and 'Frozen evaluation artifact changed' in value['error']
    assert value['resets']==value['reserved_trials']==1
    assert value['historical_read_preserved']
