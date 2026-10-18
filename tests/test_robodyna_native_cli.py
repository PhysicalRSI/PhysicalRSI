"""CLI dispatch checks: no native simulator imports or execution."""
import importlib.util
import sys
from types import ModuleType
from pathlib import Path
import pytest

from physicalrsi import cli

@pytest.fixture
def calls(monkeypatch):
    result=[]
    adapter=ModuleType('PhysicalRSI_baselines.robodyna.adapter')
    class NativeSuite:
        def __init__(self,*args,**kwargs):
            result.append(('suite',args,kwargs))
    adapter.NativeSuite=NativeSuite
    study=ModuleType('examples.robodyna.native_study')
    def run(*args,**kwargs):
        result.append(('run',args,kwargs))
        return {'kind':'contract-only'}
    def joint(*args,**kwargs):
        result.append(('joint',args,kwargs))
        return {'kind':'contract-only'}
    study.run,study.run_joint=run,joint
    monkeypatch.setitem(sys.modules,adapter.__name__,adapter)
    monkeypatch.setitem(sys.modules,study.__name__,study)
    return result


def arguments():
    return ['study','robodyna','--native','--upstream','/upstream','--python','/native-python','--workspace','/workspace']


def test_joint_uses_wrapper_fresh_seed_defaults_and_local_staging(calls):
    assert cli.main(arguments()+['--joint-after','/prior','--worker-output','/tmp/local-workers','--evaluation-mode','async'])==0
    assert calls[0][2]['output_root']==Path('/tmp/local-workers')
    assert calls[1][0]=='joint'
    assert calls[1][2]['predecessor_root']=='/prior'
    assert calls[1][2]['mode']=='async'
    assert not {'development','validation','heldout','parent_parameters','baseline_parameters'}.intersection(calls[1][2])


def test_initial_native_defaults_unchanged(calls):
    assert cli.main(arguments())==0
    assert calls[0][2]['output_root']==Path('/workspace/workers')
    assert calls[1][0]=='run'
    assert calls[1][2]['development']==(101,102)
    assert calls[1][2]['validation']==(201,202,203)
    assert calls[1][2]['heldout']==(10000,10001,10002)


@pytest.mark.parametrize('flag',['--joint-after','--worker-output'])
def test_native_only_flags_rejected_with_fixture(flag):
    with pytest.raises(SystemExit) as error:
        cli.main(['study','robodyna','--fixture','--workspace','/unused',flag,'/value'])
    assert error.value.code==2
