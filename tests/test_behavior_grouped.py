import shutil
from pathlib import Path

import pytest

from PhysicalRSI_baselines.behavior_baseline import grouped, native
from PhysicalRSI_baselines.behavior_baseline.protocol import verified_native_row
from PhysicalRSI_core.contracts import ReconciliationRequired
from PhysicalRSI_core.embodiment import Revision, System2Trial
from PhysicalRSI_core.infra.storage import atomic_json, digest, file_digest, read_json
from PhysicalRSI_core.infra.trial_quota import TrialLimitExceeded, TrialQuota


def requests(indices=(3, 0, 2, 1), *, split='validation', candidate='parent'):
    origin = System2Trial(Revision('fixture', '1'), candidate, 'a' * 64, split,
                          comparison_sha256='b' * 64, cohort_sha256='c' * 64)
    return [dict(task='radio', instance_index=i, system2=origin) for i in indices]


def fake_runtime(monkeypatch, *, exit_code=0, omit_instance=None):
    launches = []
    monkeypatch.setattr(native, 'verify_source', lambda *_: None)
    identity = dict(kind='software-fixture')
    monkeypatch.setattr(native, 'probe_policy', lambda *_: dict(
        schema='physicalrsi.behavior-policy-service/v1', candidate_sha256='a' * 64,
        task='radio', policy_identity=identity, policy_identity_sha256=digest(identity)))

    class Process:
        pid = 99999999

        def __init__(self, argv, **kwargs):
            launches.append(list(argv))
            assert '--max-steps' not in argv
            start, end = argv.index('--instance-indices') + 1, argv.index('--num-envs')
            indices = [int(x) for x in argv[start:end]]
            assert int(argv[end + 1]) == len(indices)
            output = Path(argv[argv.index('--output-dir') + 1])
            for i in indices:
                if i == omit_instance:
                    continue
                result = dict(task='radio', instance_id=i, rollout_id=0, success=False,
                              steps=123, q_score=dict(final=i / 10))
                atomic_json(output / 'json' / f'radio_{i}_0.json', result)

        def wait(self, timeout):
            return exit_code

        def poll(self):
            return exit_code

    monkeypatch.setattr(grouped.subprocess, 'Popen', Process)
    return launches


def test_group_plan_charges_all_logical_cases_and_freezes_order(tmp_path):
    small = TrialQuota(tmp_path / 'small', max_trials=3)
    with pytest.raises(TrialLimitExceeded):
        grouped.GroupedNativeTrials(tmp_path, tmp_path / 'rejected', trials=requests(), quota=small, group_size=2)
    assert small.status()['reserved_trials'] == 0
    quota = TrialQuota(tmp_path / 'quota', max_trials=4)
    plan = grouped.GroupedNativeTrials(tmp_path, tmp_path / 'run', trials=requests(), quota=quota, group_size=2)
    assert quota.status()['reserved_trials'] == 4
    assert [[m['case']['instance_index'] for m in g['specification']['members']]
            for g in plan.plan['groups']] == [[0, 1], [2, 3]]
    assert grouped.GroupedNativeTrials(tmp_path, tmp_path / 'run', trials=requests(), quota=quota, group_size=2).plan == plan.plan
    with pytest.raises(ValueError, match='plan changed'):
        grouped.GroupedNativeTrials(tmp_path, tmp_path / 'run', trials=requests(), quota=quota, group_size=3)
    for bad in (True, 1, 0, 2.5):
        with pytest.raises(ValueError, match='group size'):
            grouped.GroupedNativeTrials(tmp_path, tmp_path / 'bad', trials=requests(), quota=quota, group_size=bad)


def test_no_shortened_development_or_duplicate_group_cases(tmp_path):
    quota = TrialQuota(tmp_path / 'quota', max_trials=10)
    for values in [requests(split='development'), [dict(requests()[0], max_steps=32)]]:
        with pytest.raises(ValueError, match='full-budget'):
            grouped.GroupedNativeTrials(tmp_path, tmp_path / 'bad', trials=values, quota=quota, group_size=2)
    with pytest.raises(ValueError, match='Repeated trial'):
        grouped.GroupedNativeTrials(tmp_path, tmp_path / 'bad', trials=[requests()[0], dict(requests()[0], port=8001)], quota=quota, group_size=2)
    assert quota.status()['reserved_trials'] == 0


def test_shared_launch_publishes_independent_case_evidence_and_recovers_projection(tmp_path, monkeypatch):
    launches = fake_runtime(monkeypatch)
    quota = TrialQuota(tmp_path / 'quota', max_trials=4)
    plan = grouped.GroupedNativeTrials(tmp_path, tmp_path / 'run', trials=requests(), quota=quota, group_size=2)
    for i in range(4):
        result = plan.run(i)
        assert result['state'] == 'completed'
    assert len(launches) == 2
    for row in plan.plan['trials']:
        case = row['specification']['case']
        folder = plan.output / 'trials' / row['id']
        result = verified_native_row(folder, expected_specification=row['specification'],
            comparison_sha256='b' * 64, task='radio', instance_id=case['instance_id'])
        assert result['q_score'] == case['instance_id'] / 10
        assert row['specification']['argv'] in launches
    missing = plan.output / 'trials' / plan.plan['trials'][0]['id']
    shutil.rmtree(missing)
    plan.reconcile()
    assert native.read_receipt(missing)['state'] == 'completed'
    assert len(launches) == 2


def test_unresolved_group_blocks_all_new_effects(tmp_path, monkeypatch):
    launches = fake_runtime(monkeypatch, exit_code=1)
    plan = grouped.GroupedNativeTrials(tmp_path, tmp_path / 'run', trials=requests(),
        quota=TrialQuota(tmp_path / 'quota', max_trials=4), group_size=2)
    with pytest.raises(RuntimeError, match='Native group failed'):
        plan.run(0)
    with pytest.raises(ReconciliationRequired):
        plan.run(1)
    assert len(launches) == 1


def test_missing_member_prevents_group_and_case_completion(tmp_path, monkeypatch):
    launches = fake_runtime(monkeypatch, omit_instance=1)
    plan = grouped.GroupedNativeTrials(tmp_path, tmp_path / 'run', trials=requests((0, 1)),
        quota=TrialQuota(tmp_path / 'quota', max_trials=2), group_size=2)
    with pytest.raises(FileNotFoundError):
        plan.run(0)
    assert len(launches) == 1
    assert not (plan.output / 'trials').exists()
    with pytest.raises(ReconciliationRequired):
        plan.reconcile()


def test_case_reader_rejects_rehashed_incomplete_group_completion(tmp_path, monkeypatch):
    fake_runtime(monkeypatch)
    plan = grouped.GroupedNativeTrials(tmp_path, tmp_path / 'run', trials=requests((0, 1)),
        quota=TrialQuota(tmp_path / 'quota', max_trials=2), group_size=2)
    plan.run(0)
    folder = plan.output / 'trials' / plan.plan['trials'][0]['id']
    shared = folder / 'group-completion.json'
    value = read_json(shared)
    value['results'].pop(plan.plan['trials'][1]['id'])
    atomic_json(shared, value)
    receipt = read_json(folder / 'receipt.json')
    receipt['evidence'][shared.name] = file_digest(shared)
    atomic_json(folder / 'receipt.json', receipt)
    with pytest.raises(ValueError, match='Incomplete group'):
        native.read_receipt(folder)


def test_candidate_groups_use_same_ordered_case_partitions(tmp_path):
    values = requests(candidate='parent') + requests((0, 1, 2, 3), candidate='candidate')
    plan = grouped.GroupedNativeTrials(tmp_path, tmp_path / 'run', trials=values,
        quota=TrialQuota(tmp_path / 'quota', max_trials=8), group_size=3)
    layouts = {}
    for group in plan.plan['groups']:
        spec = group['specification']
        layouts.setdefault(spec['system2']['candidate_id'], []).append(
            [m['case']['instance_id'] for m in spec['members']])
    assert layouts == {'parent': [[0, 1, 2], [3]], 'candidate': [[0, 1, 2], [3]]}


@pytest.mark.parametrize('child_group_size', [2, 3])
def test_core_comparison_requires_identical_grouping(tmp_path, monkeypatch, child_group_size):
    from datetime import datetime, timedelta, timezone
    from PhysicalRSI_baselines.behavior_baseline.bridge import selection_result
    from PhysicalRSI_core.self_harness.selection import select_survivor

    fake_runtime(monkeypatch)
    now = datetime.now(timezone.utc)
    comparison = dict(schema_version=1, round_id='group-fixture',
        frozen_at=(now - timedelta(seconds=2)).isoformat(), parent_id='parent',
        candidates={'parent': 'a' * 64, 'child': 'a' * 64},
        profile=dict(tasks={'radio': dict(weight=1, episodes=4, score_range=[0, 1],
                     maximum_regression=0)}, minimum_gain=.01, tie_tolerance=0))
    cases = [dict(task='radio', mode='train', instance_index=i, instance_id=i, rollout_id=0)
             for i in range(4)]
    cohort = dict(comparison_sha256=digest(comparison), split='validation',
        generated_at=(now - timedelta(seconds=1)).isoformat(),
        layouts={'radio': [digest(c) for c in cases]})
    quota = TrialQuota(tmp_path / 'quota', max_trials=8)
    results = []
    for candidate, size in [('parent', 2), ('child', child_group_size)]:
        origin = System2Trial(Revision('fixture', '1'), candidate, 'a' * 64, 'validation',
            comparison_sha256=digest(comparison), cohort_sha256=digest(cohort))
        plan = grouped.GroupedNativeTrials(tmp_path, tmp_path / candidate,
            trials=[dict(task='radio', instance_index=i, system2=origin) for i in range(4)],
            quota=quota, group_size=size)
        for i in range(4):
            plan.run(i)
        results.append(selection_result([
            dict(folder=plan.output / 'trials' / row['id'], specification=row['specification'])
            for row in plan.plan['trials']], candidate_id=candidate, comparison=comparison,
            cohort=cohort, evaluator_revision='fixture', evidence_root=tmp_path))
    if child_group_size == 2:
        assert results[0]['experiment_protocol_sha256'] == results[1]['experiment_protocol_sha256']
        assert select_survivor(comparison, cohort, results, evidence_root=tmp_path)['decision'] == 'retain_parent'
    else:
        assert results[0]['experiment_protocol_sha256'] != results[1]['experiment_protocol_sha256']
        with pytest.raises(ValueError, match='protocol|conditions'):
            select_survivor(comparison, cohort, results, evidence_root=tmp_path)
