import pytest

from PhysicalRSI_baselines.robotworld.isolation import empty_proc_command, minimal_proc_command


def test_empty_proc_preserves_other_isolation_controls():
    original = ['bwrap', '--unshare-user', '--unshare-pid', '--cap-drop', 'ALL',
                '--proc', '/proc', '--ro-bind', '/public', '/observations',
                '--bind', '/scratch', '/workspace', '/runtime/codex-app-server']
    transformed = empty_proc_command(original)
    expected = original.copy()
    expected[expected.index('--proc')] = '--dir'
    assert transformed == expected
    assert '--proc' in original


@pytest.mark.parametrize('command', [
    ['sh', '--proc', '/proc'],
    ['bwrap', '--unshare-user', '--unshare-pid'],
    ['bwrap', '--unshare-user', '--unshare-pid', '--proc', '/other'],
    ['bwrap', '--unshare-user', '--proc', '/proc'],
    ['bwrap', '--unshare-user', '--unshare-pid', '--proc', '/proc', '--proc', '/x'],
])
def test_unexpected_upstream_contract_fails_closed(command):
    with pytest.raises(ValueError):
        empty_proc_command(command)


def test_minimal_proc_only_names_the_bound_app_server():
    original = ['bwrap', '--unshare-user', '--unshare-pid', '--proc', '/proc',
                '--ro-bind', '/verified/binary', '/runtime/codex-app-server',
                '/runtime/codex-app-server', '--listen', 'stdio://']
    result = minimal_proc_command(original)
    assert result[3:10] == ['--dir', '/proc', '--dir', '/proc/self',
                             '--symlink', '/runtime/codex-app-server', '/proc/self/exe']
    assert result[10:] == original[5:]
    assert '--proc' in original
