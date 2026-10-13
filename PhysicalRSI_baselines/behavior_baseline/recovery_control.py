"""Experimental observation-triggered local recovery around a fixed primitive.

Proposed from development RGB showing approach followed by a stationary hand.
No native improvement has been established. All quantities are onboard joint
positions, base velocity and wrist RGB; no object identity or reset is encoded.
"""
import numpy as np


def initialize(task):
    return {'steps': 0}


def _views(observation):
    return np.stack([np.asarray(observation['rgb'][name])[:, ::8, ::8, :].astype(np.float32) / 255
                     for name in ('left_wrist', 'right_wrist')], axis=1)


def step(observation, state, primitive):
    action = np.asarray(primitive(observation)).copy()
    arms = np.stack([observation['proprio'][name] for name in
                     ('left_arm_qpos', 'right_arm_qpos')], axis=1)
    speed = np.linalg.norm(observation['proprio']['base_qvel'], axis=1)
    targets = np.stack([action[:, 7:14], action[:, 15:22]], axis=1)
    error = np.sqrt(np.mean((targets - arms) ** 2, axis=-1))
    batch = arms.shape[0]
    state['steps'] += 1
    if 'previous' not in state:
        state.update(previous=arms.copy(), reference=arms.copy(), images=_views(observation),
            travel=np.zeros((batch, 2)), quiet=np.zeros((batch, 2), dtype=int),
            recovery=np.full(batch, -1, dtype=int), arm=np.zeros(batch, dtype=int),
            attempts=np.zeros(batch, dtype=int), cooldown=np.zeros(batch, dtype=int))
        return action
    if state['previous'].shape != arms.shape:
        raise ValueError('Reset controller state when the episode batch changes')
    state['travel'] += np.linalg.norm(arms - state['previous'], axis=-1)
    state['previous'] = arms.copy()
    state['cooldown'] = np.maximum(state['cooldown'] - 1, 0)
    if state['steps'] % 32 == 0:
        images = _views(observation)
        motion = np.sqrt(np.mean((arms - state['reference']) ** 2, axis=-1))
        image_change = np.mean(np.abs(images - state['images']), axis=(2, 3, 4))
        quiet = ((motion < .003) & (image_change < .008) & (error < .025)
                 & (speed[:, None] < .01) & (state['travel'] > .25))
        state['quiet'] = np.where(quiet, state['quiet'] + 1, 0)
        state['reference'], state['images'] = arms.copy(), images
        if state['steps'] >= 192:
            for index, arm in enumerate(np.argmax(state['travel'], axis=1)):
                if (state['quiet'][index, arm] >= 2 and state['recovery'][index] < 0
                        and state['cooldown'][index] == 0 and state['attempts'][index] < 3):
                    state['recovery'][index], state['arm'][index] = 0, arm
                    state['attempts'][index] += 1
                    state['quiet'][index] = 0
    for index in range(batch):
        tick = state['recovery'][index]
        if tick < 0:
            continue
        arm = state['arm'][index]
        if speed[index] > .02 or error[index, arm] > .06:
            state['recovery'][index], state['cooldown'][index] = -1, 96
            continue
        # Native arm commands are absolute joint positions. Offsets remain
        # within 0.03 rad of the primitive target and fade to zero at both ends.
        phase = 2 * np.pi * tick / 48
        faster = 2 * np.pi * tick / 32
        envelope = np.sin(np.pi * tick / 95) ** 2
        offset = envelope * np.array([0, .018*np.sin(phase), 0, .025*np.cos(phase),
                                      .03*np.sin(faster), -.02*np.sin(phase), .015*np.cos(faster)])
        start = 7 if arm == 0 else 15
        action[index, start:start+7] += offset.astype(action.dtype)
        state['recovery'][index] += 1
        if state['recovery'][index] >= 96:
            state['recovery'][index], state['cooldown'][index] = -1, 96
    return action
