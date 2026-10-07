"""Record actual tracked source bytes, local patches and submodule identities.

Assets, ignored build outputs and symlink targets require separate inventories.
A commit identifier alone is not an attestation of the bytes currently executed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args])


def capture(root):
    root = Path(git(root, 'rev-parse', '--show-toplevel').decode().strip()).resolve()
    rows = []
    issues = []
    for record in git(root, 'ls-files', '--stage', '-z').split(b'\0'):
        if not record:
            continue
        header, name = record.split(b'\t', 1)
        mode, blob, stage = header.decode().split()
        name = name.decode()
        path = root / name
        row = {'path': name, 'git_mode': mode, 'index_object': blob, 'stage': stage}
        if stage != '0':
            issues.append({'path': name, 'problem': 'unmerged-index'})
        if mode == '160000':
            try:
                checkout_root = Path(git(path, 'rev-parse', '--show-toplevel').decode().strip()).resolve()
                # git walks up from an uninitialized submodule directory and
                # can otherwise report the parent's commit as the checkout.
                initialized = checkout_root == path.resolve()
                row['checkout_commit'] = git(path, 'rev-parse', 'HEAD').decode().strip() if initialized else None
            except subprocess.CalledProcessError:
                row['checkout_commit'] = None
            if row['checkout_commit'] is None:
                issues.append({'path': name, 'problem': 'submodule-not-initialized'})
            elif row['checkout_commit'] != blob:
                issues.append({'path': name, 'problem': 'submodule-commit-differs-from-index'})
            row['contents_inventoried'] = False
        elif path.is_symlink():
            row['symlink_target'] = str(path.readlink())
            row['target_contents_inventoried'] = False
        elif path.is_file():
            before = path.stat()
            h = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                    h.update(block)
            after = path.stat()
            if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                    after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise RuntimeError('Source changed while hashing: ' + name)
            row.update(sha256=h.hexdigest(), bytes=after.st_size)
        else:
            issues.append({'path': name, 'problem': 'missing-tracked-file'})
        rows.append(row)
    return {'scope': 'tracked-source-byte-inventory', 'qualification': None,
            'complete_execution_closure_attested': False, 'root': str(root),
            'commit': git(root, 'rev-parse', 'HEAD').decode().strip(),
            'status_porcelain': git(root, 'status', '--porcelain', '--untracked-files=normal').decode(),
            'tracked': rows, 'issues': issues,
            'limits': ['Ignored build outputs, untracked files, submodule contents and symlink targets are not captured',
                       'Freeze source trees and reverify before execution admission']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Use a fresh output')
    result = capture(args.root)
    with args.output.open('x') as stream:
        json.dump(result, stream, sort_keys=True)
    print(json.dumps({'root': result['root'], 'commit': result['commit'],
                      'tracked_entries': len(result['tracked']), 'issues': len(result['issues'])}))


if __name__ == '__main__':
    main()
