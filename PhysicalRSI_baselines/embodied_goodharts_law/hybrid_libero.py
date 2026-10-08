"""Admission boundary for LIBERO providers in an observation-gated skill plan.

This does not load a checkpoint or convert joint control to operational-space
control. A reviewed backend must perform its documented image preprocessing,
normalization and gripper conversion before returning native actions.
"""
import math
from copy import deepcopy

from PhysicalRSI_core.contracts import Contract, Operation
from PhysicalRSI_core.infra.storage import digest
from PhysicalRSI_core.skill_execution import RELEASE, REQUEST, STOPPED
from PhysicalRSI_core.timing import action_chunk, finite_seconds


def action_contract(controller):
    if controller not in {'OSC_POSE', 'JOINT_POSITION'}:
        raise ValueError('Unsupported LIBERO controller')
    dimensions = 7 if controller == 'OSC_POSE' else 8
    return Contract('libero.native-action-chunk/v1',
                    unit=f'normalized-{dimensions}d', frame=controller, embodiment='libero-panda')


def validate_actions(actions, *, controller):
    action_contract(controller)
    dimensions = 7 if controller == 'OSC_POSE' else 8
    if not isinstance(actions, list) or not actions:
        raise ValueError('Nonempty native action list required')
    for action in actions:
        if (not isinstance(action, list) or len(action) != dimensions
                or any(type(x) not in (int, float) or not math.isfinite(x) or abs(x) > 1
                       for x in action)):
            raise ValueError('Wrong action dimension, finite value or normalized range')


def learned_provider(name, *, controller, period_seconds, pins, predict, reset):
    """Wrap an explicitly pinned, externally bounded inference backend.

    predict(public_request, context) returns native action lists. It must not
    actuate. reset(context) discards provider caches/action queues and returns
    exactly True on confirmed completion. Model pinning is a caller attestation;
    the backend loader remains responsible for checking the actual model bytes.
    """
    required = {'checkpoint', 'implementation', 'preprocessing', 'normalization'}
    if (set(pins) != required or any(not isinstance(v, str) or len(v) != 64
            or any(c not in '0123456789abcdef' for c in v) for v in pins.values())):
        raise ValueError('Pin checkpoint, implementation, preprocessing and normalization SHA256')
    if not callable(predict) or not callable(reset):
        raise ValueError('Explicit inference and reset callbacks required')
    finite_seconds(period_seconds)
    contract = action_contract(controller)
    revision = digest(dict(name=name, pins=pins, controller=controller, period_seconds=period_seconds))

    def propose(request, context):
        context.check()
        # Pass the complete relation-bearing goal and only caller-admitted public observations.
        actions = predict(deepcopy(request), context)
        validate_actions(actions, controller=controller)
        context.check()
        return action_chunk(request['observation'], actions, period_seconds=period_seconds)

    def release(request, context):
        return {'stopped': reset(context) is True}

    return (Operation(name, revision, REQUEST, contract, propose),
            Operation(name + '.release', revision, RELEASE, STOPPED, release))
