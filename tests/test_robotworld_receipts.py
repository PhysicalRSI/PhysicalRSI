import json
import pytest
from PhysicalRSI_baselines.robotworld.receipts import robocasa_receipt


@pytest.fixture
def native(tmp_path):
    folder = tmp_path / 'CloseDrawer' / 'episode-0000'
    folder.mkdir(parents=True)
    config = dict(task='CloseDrawer', seed=7, horizon=450, official_horizon=450,
                  horizon_override=None, probe_only=False)
    result = dict(seed=7, steps=450, success=False, termination='horizon', probe_only=False)
    def write():
        (folder / 'episode.json').write_text(json.dumps(result))
        (folder / 'evaluation-config.json').write_text(json.dumps(config))
    def read():
        write()
        return robocasa_receipt(tmp_path, task='CloseDrawer', seed=7, horizon=450)
    return config, result, read


def test_native_failure_is_scored(native):
    _, _, read = native
    receipt = read()
    assert receipt['outcome'] == 'failure' and receipt['score'] == 0
    assert len(receipt['evidence']) == 2
    assert receipt['qualification'] is None


@pytest.mark.parametrize('patch', [dict(probe_only=True), dict(termination='wall_timeout'),
    dict(termination='infrastructure_error'), dict(steps=8), dict(success=1), dict(seed=8),
    dict(termination='success', success=False)])
def test_invalid_results_never_become_failures(native, patch):
    _, result, read = native
    result.update(patch)
    with pytest.raises(ValueError):read()


def test_success_uses_native_terminal_result(native):
    _, result, read = native
    result.update(success=True, termination='success', steps=120)
    assert read()['score'] == 1


def test_horizon_override_requires_separate_protocol(native):
    config, _, read = native
    config['horizon_override'] = 450
    with pytest.raises(ValueError):read()
