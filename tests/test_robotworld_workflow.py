import pytest
from PhysicalRSI_baselines.robotworld.workflow import campaign
from PhysicalRSI_core.infra.storage import digest


class UnavailablePort:
    def identity(self):
        return {'kind': 'test-port-not-a-native-runtime'}


def inventory():
    body = dict(schema='physicalrsi.robotworld-catalog/v1', code_control=False,
                upstream_revision='pinned-test-fixture', tasks=[dict(
                    benchmark='robocasa', task='CloseDrawer', default_steps=450,
                    code_control_default=False, scoring_profile='native')])
    return dict(body=body, revision=digest(body))


def test_campaign_preserves_native_protocol_without_running_ports(tmp_path):
    c = campaign(tmp_path, catalog=inventory(), benchmark='robocasa',
                 tasks=['CloseDrawer'], proposer=UnavailablePort(),
                 evaluator=UnavailablePort(), rounds=2)
    loop = c.build_loop(c.root / 'rounds' / 'round-0001')
    assert loop.config['protocol']['tasks']['CloseDrawer']['default_steps'] == 450
    assert loop.config['protocol']['code_control'] is False
    assert loop.config['profile']['tasks']['CloseDrawer']['episodes'] == 3
    assert loop.config['scope'] == 'robotworld_native_task_adaptation'
    assert not (tmp_path / 'state' / 'current.json').exists()


@pytest.mark.parametrize('change', ['tamper', 'wrong-task', 'code-control'])
def test_campaign_rejects_protocol_changes(tmp_path, change):
    cat = inventory()
    if change == 'tamper':cat['body']['tasks'][0]['default_steps'] = 1
    if change == 'code-control':
        cat['body']['code_control'] = True
        cat['revision'] = digest(cat['body'])
    with pytest.raises(ValueError):
        campaign(tmp_path, catalog=cat, benchmark='robocasa',
                 tasks=['Other' if change == 'wrong-task' else 'CloseDrawer'],
                 proposer=UnavailablePort(), evaluator=UnavailablePort(), rounds=2)
