"""Explicit deployment variant for hosts that reject a new procfs mount.

Transform only a trusted, pinned upstream bubblewrap command. This helper is
not an admission check for arbitrary commands or an automatic fallback after
an isolation error. The evaluator must record this variant in its identity.
"""
from pathlib import Path


def empty_proc_command(command):
    """Preserve the upstream sandbox, replacing its procfs with an empty dir.

    No host /proc is exposed. Programs requiring procfs may fail, so successful
    command construction alone cannot qualify a runtime or benchmark result.
    """
    args = list(command)
    if not args or Path(args[0]).name != 'bwrap':
        raise ValueError('Expected a pinned upstream bubblewrap command')
    if args.count('--proc') != 1:
        raise ValueError('Expected exactly one procfs mount')
    index = args.index('--proc')
    if index + 1 >= len(args) or args[index + 1] != '/proc':
        raise ValueError('Unexpected procfs destination')
    if '--unshare-user' not in args or '--unshare-pid' not in args:
        raise ValueError('Preserve upstream user and PID namespaces')
    args[index] = '--dir'
    return args


def minimal_proc_command(command):
    """Add only the pinned app-server's executable link, without host procfs.

    This fixed link supports Codex startup; it is not general procfs emulation
    and subprocesses cannot use it to discover their own executable correctly.
    The caller must separately validate runtime/tool compatibility.
    """
    args = empty_proc_command(command)
    executable = '/runtime/codex-app-server'
    if executable not in args:
        raise ValueError('Expected the upstream app-server executable path')
    index = next(i for i in range(len(args) - 1)
                 if args[i:i + 2] == ['--dir', '/proc']) + 2
    args[index:index] = ['--dir', '/proc/self', '--symlink', executable, '/proc/self/exe']
    return args
