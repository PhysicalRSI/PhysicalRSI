"""Freeze the upstream task and budget inventory without importing simulator code."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from PhysicalRSI_core.infra.storage import atomic_json, digest, locked

REVISION = '229251b4c3e81cb9839bec9cc0a98fa288d3750a'
SOURCE = 'https://github.com/robotworldai/robotworld'
FILES = ('docs/HANDOFF_TASKS.json', 'environment/evaluation/suites.json',
         'environment/evaluation/native17.json', 'sources.lock.json')


def snapshot(upstream):
    """Read pinned Git objects, never mutable working files or provider code."""
    root = Path(upstream).resolve(strict=True)
    files, content = {}, {}
    for name in FILES:
        raw = subprocess.check_output(['git', '-C', str(root), 'show', f'{REVISION}:{name}'])
        files[name] = hashlib.sha256(raw).hexdigest()
        content[name] = json.loads(raw)
    tasks = content['docs/HANDOFF_TASKS.json']
    identities = [(row['benchmark'], row['task']) for row in tasks]
    if not tasks or len(set(identities)) != len(identities):
        raise ValueError('Task inventory must be nonempty and unique')
    for row in tasks:
        if type(row['default_steps']) is not int or row['default_steps'] <= 0:
            raise ValueError('Every task needs a positive native or declared step budget')
        if row['code_control_default'] is not False:
            raise ValueError('Unexpected upstream control permissions; review protocol')
    body = dict(schema='physicalrsi.robotworld-catalog/v1', source=SOURCE,
                upstream_revision=REVISION, files=files, tasks=tasks,
                task_count=len(tasks), benchmark_count=len({x[0] for x in identities}),
                default_rollouts=3, code_control=False, scope='source_inventory',
                qualification=None)
    return dict(revision=digest(body), body=body)


def freeze(upstream, workspace):
    result = snapshot(upstream)
    target = Path(workspace) / 'catalog.json'
    with locked(target.parent / '.catalog.lock'):
        if target.exists():
            if json.loads(target.read_text()) != result:
                raise ValueError('Inventory changed; use a new workspace')
        else:
            atomic_json(target, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', required=True)
    parser.add_argument('--workspace', required=True)
    args = parser.parse_args()
    result = freeze(args.upstream, args.workspace)
    print(json.dumps({k: result['body'][k] for k in
                      ('task_count', 'benchmark_count', 'upstream_revision', 'scope', 'qualification')}, indent=2))


if __name__ == '__main__':
    main()
