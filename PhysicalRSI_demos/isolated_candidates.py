"""Reviewed stdlib-only programs for the isolated MuJoCo campaign fixture.

These are deterministic candidate programs. No model-generated repair quality
or physical qualification is established by executing them inside a jail.
"""

from PhysicalRSI_core.infra.storage import canonical


_POLICY = '''
import hashlib
import json

def begin_episode(task, case, observation):
    return dict(target=case['target'])

def act(observation, state):
    sample = observation['payload']
    kp, kd, limit = configuration['gains']
    effort = (max(-limit, min(limit, kp * (state['target'] - sample['qpos'][0]) - kd * sample['qvel'][0]))
              if configuration['mode'] == 'feedback' else 0.)
    reference = {key: observation[key] for key in ('episode', 'clock_domain', 'sequence')}
    encoded = json.dumps(observation, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()
    reference['sha256'] = hashlib.sha256(encoded).hexdigest()
    action = dict(schema='physicalrsi.action-chunk/v1', observation=reference,
                  period_seconds=configuration['period_seconds'], actions=[[effort], [effort]])
    return dict(action=action, state=state)
'''


def joint_policy(*, mode, gains, period_seconds):
    configuration = canonical(dict(mode=mode, gains=gains, period_seconds=period_seconds)).decode()
    return "import json\nconfiguration = json.loads(" + repr(configuration) + ")\n" + _POLICY


PROPOSAL = '''
import json

def generate(request, limits):
    current = request['editable']['control.json']
    outcomes = [episode['outcome'] for episode in request['development']['episodes']]
    candidates = []
    if 'failure' in outcomes and current['value']['mode'] != 'feedback':
        candidates = [dict(rationale='Evaluate the reviewed feedback mode after development failure',
            edits=[dict(path='control.json', before_sha256=current['sha256'], value=dict(mode='feedback'))])]
    return dict(text=json.dumps(dict(parent_sha256=request['parent_sha256'], candidates=candidates)),
                usage=None, provider=None, tool_calls=[])
'''
