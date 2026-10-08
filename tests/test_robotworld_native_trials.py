import json
from pathlib import Path
import pytest

from PhysicalRSI_baselines.robotworld.native_trials import RoboCasaTrials, validate_case
from PhysicalRSI_core.contracts import ReconciliationRequired
from PhysicalRSI_core.infra.storage import atomic_json, file_digest, read_json
from PhysicalRSI_core.self_harness.artifacts import CLOSURE


@pytest.fixture
def trial(tmp_path):
    seed = tmp_path / 'candidate'
    components = {}
    for kind in CLOSURE:
        path = seed / (kind + '.json')
        atomic_json(path, {'fixture': kind})
        components[kind] = {path.name: file_digest(path)}
    candidate = dict(id='fixture-parent', root=str(seed), components=components)
    case = dict(task='CloseDrawer', launcher_seed=7, reset_seed=12, task_index=5,
                horizon=450, task_set='all_tasks', split='pretrain', trial_slot=0)
    deployment = tmp_path / 'deployment.json'
    atomic_json(deployment, {'fixture': True})
    worker = tmp_path / 'worker.py'
    worker.write_text('''import argparse,json
from pathlib import Path
from PhysicalRSI_core.infra.storage import atomic_json,digest
p=argparse.ArgumentParser()
for name in ('deployment','request','output'):p.add_argument('--'+name,required=True)
a=p.parse_args();request=json.loads(Path(a.request).read_text());c=request['case'];out=Path(a.output)
out.mkdir();(out/'invocation.txt').write_text('called once')
atomic_json(out/'execution-binding.json',{'request_sha256':digest(request)})
folder=out/c['task']/'episode-000'
atomic_json(folder/'evaluation-config.json',dict(task=c['task'],seed=c['reset_seed'],horizon=c['horizon'],official_horizon=c['horizon'],horizon_override=None,probe_only=False,task_set=c['task_set'],split=c['split'],task_index=c['task_index'],global_episode_index=c['task_index']))
atomic_json(folder/'episode.json',dict(seed=c['reset_seed'],steps=c['horizon'],success=False,termination='horizon',probe_only=False))
''')
    runner = RoboCasaTrials(worker=worker, deployment=deployment)
    return runner, candidate, case, tmp_path / 'trial'


def test_completed_native_failure_replays_without_worker_execution(trial):
    runner, candidate, case, output = trial
    result = runner.run(candidate, case, output, seconds=10)
    assert result['score'] == 0 and result['outcome'] == 'failure'
    assert result['qualification'] is None
    # Worker refuses an existing output directory; a second execution would fail.
    assert runner.run(candidate, case, output, seconds=10) == result


def test_changed_receipt_cannot_replay_as_success(trial):
    runner, candidate, case, output = trial
    runner.run(candidate, case, output, seconds=10)
    path = output / 'native/CloseDrawer/episode-000/episode.json'
    changed = read_json(path)
    changed.update(success=True, termination='success')
    atomic_json(path, changed)
    with pytest.raises(ValueError, match='evidence changed'):
        runner.run(candidate, case, output, seconds=10)


def test_incomplete_launch_is_not_scored_or_automatically_retried(trial):
    runner, candidate, case, output = trial
    runner.worker.write_text('raise RuntimeError("fixture infrastructure failure")\n')
    runner = RoboCasaTrials(worker=runner.worker, deployment=runner.deployment)
    with pytest.raises(ReconciliationRequired):
        runner.run(candidate, case, output, seconds=10)
    assert read_json(output / 'trial.json')['state'] == 'unresolved'
    with pytest.raises(ReconciliationRequired, match='automatic replay'):
        runner.run(candidate, case, output, seconds=10)


def test_seed_metadata_must_match_effective_native_reset(trial):
    _, _, case, _ = trial
    case['reset_seed'] = case['launcher_seed']
    with pytest.raises(ValueError):
        validate_case(case)
