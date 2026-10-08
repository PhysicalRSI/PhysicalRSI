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
