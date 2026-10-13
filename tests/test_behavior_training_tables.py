import numpy as np
import pytest

from PhysicalRSI_baselines.behavior_baseline.training_tables import episodes, normalization_batches


def tables():
    action = np.zeros((5, 23), np.float32)
    action[:, 0] = [0, 1, 2, 10, 11]
    state = np.zeros((5, 61), np.float32)
    state[:, 0] = [1, 2, 3, 4, 5]
    rows = dict(index=np.arange(5), episode_index=np.array([0, 0, 0, 1, 1]),
                frame_index=np.array([0, 1, 2, 0, 1]), task_index=np.array([7, 7, 7, 9, 9]),
                action=action, **{'observation.state': state})
    return [{k: v[a:b] for k, v in rows.items()} for a, b in [(0, 2), (2, 4), (4, 5)]]


EXPECTED = [dict(length=3, task_index=7), dict(length=2, task_index=9)]


def test_cross_file_episodes_and_cross_episode_batches():
    source = list(episodes(tables(), expected=EXPECTED))
    originals = [x['actions'].copy() for x in source]

    def delta(data):
        data['actions'] -= data['state'][:, None, :]
        return data

    result = list(normalization_batches(source, extract_state=lambda x: x[:, :23],
                                       transform=delta, horizon=3, batch_size=4, drop_last=False))
    assert [len(x['state']) for x in result] == [4, 1]
    actual = np.concatenate([x['actions'][:, :, 0] for x in result])
    np.testing.assert_array_equal(actual, [[-1, 0, 1], [-1, 0, 0], [-1, -1, -1],
                                          [6, 7, 7], [6, 6, 6]])
    for before, after in zip(originals, source):
        np.testing.assert_array_equal(before, after['actions'])
    dropped = list(normalization_batches(iter(source), extract_state=lambda x: x[:, :23],
                                        transform=delta, horizon=3, batch_size=4))
    assert len(dropped) == 1
    np.testing.assert_array_equal(dropped[0]['actions'], result[0]['actions'])


@pytest.mark.parametrize('column,value', [('index', 99), ('frame_index', 0),
                                         ('episode_index', 1), ('task_index', 9)])
def test_rejects_row_aliases_and_wrong_task(column, value):
    batches = tables()
    batches[0][column][1] = value
    with pytest.raises(ValueError, match='coverage or order'):
        list(episodes(batches, expected=EXPECTED))


def test_incomplete_and_nonfinite_cohort_fail():
    with pytest.raises(ValueError, match='Incomplete'):
        list(episodes(tables()[:-1], expected=EXPECTED))
    batches = tables()
    batches[0]['action'][0, 0] = np.nan
    with pytest.raises(ValueError, match='Malformed'):
        list(episodes(batches, expected=EXPECTED))
