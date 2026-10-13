"""Training-only table traversal for normalization without video decoding.

Callers verify publisher file hashes and supply episode metadata before reading.
The functions preserve episode-clipped action horizons and global batch order;
they neither compute task scores nor admit incomplete normalization artifacts.
"""
import numpy as np


def episodes(batches, *, expected):
    """Join contiguous table batches using declared episode lengths and tasks.

``expected`` is ordered metadata containing ``length`` and ``task_index``.
Every source row must have its original global index and episode/frame index.
The iterator must be exhausted before claiming complete-cohort coverage.
    """
    if not expected or any(type(row['length']) is not int or row['length'] < 1
                           or type(row['task_index']) is not int or row['task_index'] < 0
                           for row in expected):
        raise ValueError('Require positive episode lengths and task identities')
    episode, frame, index = 0, 0, 0
    states, actions = [], []
    keys = {'index', 'episode_index', 'frame_index', 'task_index', 'observation.state', 'action'}
    for batch in batches:
        if set(batch) != keys:
            raise ValueError('Unexpected training table columns')
        batch = {key: np.asarray(value) for key, value in batch.items()}
        count = len(batch['index'])
        for key in ('index', 'episode_index', 'frame_index', 'task_index'):
            if batch[key].shape != (count,) or not np.issubdtype(batch[key].dtype, np.integer):
                raise ValueError('Require integer row identities')
        for key, width in (('observation.state', 61), ('action', 23)):
            if (batch[key].shape != (count, width)
                    or not np.issubdtype(batch[key].dtype, np.floating)
                    or not np.isfinite(batch[key]).all()):
                raise ValueError('Malformed state or action table')
        offset = 0
        while offset < count:
            if episode >= len(expected):
                raise ValueError('Rows exceed declared episode cohort')
            metadata = expected[episode]
            size = min(count-offset, metadata['length']-frame)
            part = slice(offset, offset+size)
            if (not np.array_equal(batch['index'][part], np.arange(index, index+size))
                    or not np.array_equal(batch['frame_index'][part], np.arange(frame, frame+size))
                    or not np.all(batch['episode_index'][part] == episode)
                    or not np.all(batch['task_index'][part] == metadata['task_index'])):
                raise ValueError('Table rows differ from declared episode coverage or order')
            states.append(batch['observation.state'][part].copy())
            actions.append(batch['action'][part].copy())
            offset, frame, index = offset+size, frame+size, index+size
            if frame == metadata['length']:
                yield dict(episode_index=episode, task_index=metadata['task_index'],
                           state=np.concatenate(states), actions=np.concatenate(actions))
                episode, frame, states, actions = episode+1, 0, [], []
    if episode != len(expected) or frame:
        raise ValueError('Incomplete training episode cohort')


def normalization_batches(source, *, extract_state, transform, horizon=32,
                          batch_size=64, drop_last=True):
    """Apply supplied upstream transforms to globally ordered training batches.

Action horizons repeat the last action at each episode boundary. Batches span
episodes, matching the official sequential normalization loader's batching.
The upstream transform may mutate its action input; gathered actions are copies.
    """
    if type(horizon) is not int or horizon < 1 or type(batch_size) is not int or batch_size < 1:
        raise ValueError('Require positive horizon and batch size')
    parts, pending = [], 0
    for episode in source:
        count = len(episode['actions'])
        for offset in range(0, count, batch_size):
            indices = np.arange(offset, min(offset+batch_size, count))
            future = np.minimum(indices[:, None]+np.arange(horizon), count-1)
            data = transform(dict(state=extract_state(episode['state'][indices]),
                                  actions=episode['actions'][future].copy()))
            if set(data) != {'state', 'actions'} or any(not np.isfinite(v).all() for v in data.values()):
                raise ValueError('Unexpected normalization transform output')
            start = 0
            while start < len(indices):
                size = min(batch_size-pending, len(indices)-start)
                parts.append({key: value[start:start+size] for key, value in data.items()})
                start, pending = start+size, pending+size
                if pending == batch_size:
                    yield {key: np.concatenate([part[key] for part in parts]) for key in data}
                    parts, pending = [], 0
    if pending and not drop_last:
        yield {key: np.concatenate([part[key] for part in parts]) for key in ('state', 'actions')}
