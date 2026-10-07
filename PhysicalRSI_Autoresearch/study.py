"""Bounded research trials with immutable evidence and resumable selection.

System 2 supplies declarative hypotheses. Reviewed System 1 code executes them.
Neither this runner nor its ideal audit qualifies a physical laboratory result.
"""
import csv
import io
import json
import math
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from PhysicalRSI_core.infra.storage import atomic_json, digest, file_digest, locked
from PhysicalRSI_baselines.opentrons_liquid_handling.policy import Case, plan, policy_settings, protocol_source
from PhysicalRSI_baselines.opentrons_liquid_handling.audit import audit

DEFAULT_PROPOSALS = [
    {'name': 'mixed', 'hypothesis': 'Explicit mixing establishes a valid reference protocol.',
     'settings': {'mix_repetitions': 3, 'batch_water': False}},
    {'name': 'unmixed', 'hypothesis': 'Negative control: legal transfers without mixing must fail the independent audit.',
     'settings': {'mix_repetitions': 0, 'batch_water': False}},
    {'name': 'batched', 'hypothesis': 'Batching water before stock addition reduces tips without changing dilution.',
     'settings': {'mix_repetitions': 3, 'batch_water': True}},
    {'name': 'reuse-mix-tip', 'hypothesis': 'Mixing with the just-dispensed transfer tip avoids a fresh tip at the same well.',
     'settings': {'mix_repetitions': 3, 'batch_water': True, 'reuse_mix_tip': True}},
]
DEVELOPMENT = [asdict(Case(4, 100)), asdict(Case(6, 100))]
VALIDATION = [asdict(Case(8, 80)), asdict(Case(12, 120))]


def proposals_checked(proposals):
    if not isinstance(proposals, list) or not 1 <= len(proposals) <= 100:
        raise ValueError('Declare 1–100 proposals')
    result = []
    for p in proposals:
        if not isinstance(p, dict) or set(p) != {'name', 'hypothesis', 'settings'}:
            raise ValueError('A proposal needs name, hypothesis and settings')
        if any(not isinstance(p[k], str) or not p[k].strip() or len(p[k]) > 2000 for k in ('name', 'hypothesis')):
            raise ValueError('Invalid proposal description')
        result.append(dict(p, settings=policy_settings(p['settings'])))
    if len({p['name'] for p in result}) != len(result):
        raise ValueError('Proposal names must be unique')
    return result


class OpentronsSimulator:
    def __init__(self, python):
        # Preserve virtual-environment symlink identity.
        self.python = str(Path(python).absolute())
        p = subprocess.run([self.python, '-c',
            "import importlib.metadata; print(importlib.metadata.version('opentrons'))"],
            capture_output=True, text=True, timeout=30, check=True)
        if p.stdout.strip() != '8.8.2':
            raise ValueError('Expected opentrons==8.8.2')
        freeze = subprocess.run([self.python, '-m', 'pip', 'freeze'], capture_output=True,
                                text=True, timeout=60, check=True)
        self.identity = {'kind': 'opentrons', 'version': '8.8.2', 'python': self.python,
                         'installed_packages': freeze.stdout.splitlines()}

    def __call__(self, protocol, output, timeout):
        repo = Path(__file__).resolve().parent.parent
        command = [self.python, '-m', 'PhysicalRSI_baselines.opentrons_liquid_handling.simulate',
                   str(protocol), str(output / 'simulation.json')]
        try:
            with (output / 'simulator.log').open('w') as log:
                p = subprocess.run(command, cwd=repo, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
            result = json.loads((output / 'simulation.json').read_text()) if p.returncode == 0 else {}
            return {'passed': p.returncode == 0 and result.get('opentrons_version') == '8.8.2'
                    and result.get('scope') == 'opentrons-software-simulation', 'returncode': p.returncode}
        except subprocess.TimeoutExpired:
            with (output / 'simulator.log').open('a') as log:
                log.write('\nSimulator exceeded trial budget.\n')
            return {'passed': False, 'error': 'timeout'}


def _sources():
    repo = Path(__file__).resolve().parent.parent
    paths = list((repo / 'PhysicalRSI_Autoresearch').glob('*.py'))
    paths += list((repo / 'PhysicalRSI_baselines/opentrons_liquid_handling').glob('*.py'))
    paths += [repo / 'PhysicalRSI_core/infra/storage.py', Path(__file__).with_name('literature.json'),
              Path(__file__).with_name('program.md')]
    return {str(p.relative_to(repo)): file_digest(p) for p in sorted(paths)}


def _verify(root, state):
    for ref in state['evidence']:
        p = root / ref['path']
        if file_digest(p) != ref['sha256']:
            raise ValueError('Recorded evidence changed: ' + ref['path'])
        receipt = json.loads(p.read_text())
        for name, sha in receipt['artifacts'].items():
            if file_digest(p.parent / name) != sha:
                raise ValueError('Trial artifact changed: ' + name)
    if state.get('memory_revision'):
        memory = json.loads((root / 'memory' / (state['memory_revision'] + '.json')).read_text())
        if digest(memory) != state['memory_revision']:
            raise ValueError('Memory revision changed')


def _reports(root, state):
    out = io.StringIO()
    writer = csv.writer(out, delimiter='\t')
    writer.writerow(['round', 'candidate', 'status', 'tips', 'commands', 'parent', 'receipt'])
    for row in state['rounds']:
        writer.writerow([row[k] for k in ('round', 'candidate', 'status', 'tips', 'commands', 'parent', 'receipt')])
    (root / 'results.tsv').write_text(out.getvalue())
    lines = ['# Research report', '', 'Scope: software protocol simulation and ideal liquid audit.',
             'Physical experiments: 0. Physical qualification: none.', '',
             f"State: {state['status']}. Selected candidate: {state['selected']}.", '',
             '| Round | Candidate | Decision | Tips | Commands |', '|---|---|---|---|---|']
    for row in state['rounds']:
        # Names are untrusted prose; JSON escaping does not escape Markdown.
        name = row['candidate'].replace('|', '\\|').replace('\n', ' ')
        lines.append(f"| {row['round']} | {name} | {row['status']} | {row['tips']} | {row['commands']} |")
    lines += ['', 'Selection uses development cases only. Ties retain the incumbent.',
              'Validation is run after the proposal budget is exhausted and never selects a survivor.',
              'These declared validation cases are public, not a secret held-out benchmark.',
              'Mixing and contamination checks use an ideal model, not physical measurements.']
    (root / 'report.md').write_text('\n'.join(lines) + '\n')


def run_study(workspace, simulator, *, proposals=None, resume=False, max_rounds=None, trial_seconds=120):
    """Run or resume a frozen proposal batch; interruption preserves completed rounds.

    max_rounds bounds work in this invocation. Proposal/evaluator changes require
    a new workspace. A partial trial is retained and retried in a new directory.
    """
    if not math.isfinite(trial_seconds) or not 0 < trial_seconds <= 3600:
        raise ValueError('Expected a trial budget of 0–3600 seconds')
    if max_rounds is not None and (type(max_rounds) is not int or max_rounds < 1):
        raise ValueError('Expected a positive round budget')
    proposals = proposals_checked(DEFAULT_PROPOSALS if proposals is None else proposals)
    root = Path(workspace).resolve()
    manifest = {'schema': 'physicalrsi.autoresearch/v2', 'scope': 'software-protocol-research',
                'question': 'Can serial dilution consume fewer tips while preserving the declared liquid audit?',
                'proposals': proposals, 'development': DEVELOPMENT, 'validation': VALIDATION,
                'sources': _sources(), 'simulator': simulator.identity, 'trial_seconds': trial_seconds,
                'selection': 'development validity, then total tips, then commands; incumbent wins ties',
                'qualification': None, 'self_harness_admitted': False}
    if not resume:
        root.mkdir(parents=True, exist_ok=False)
    elif not (root / 'state.json').is_file():
        raise ValueError('No resumable study state')
    with locked(root / '.lock'):
        if resume:
            if json.loads((root / 'study.json').read_text()) != manifest:
                raise ValueError('Study configuration or source changed; start a new workspace')
            for name, sha in manifest['sources'].items():
                if file_digest(root / 'sources' / name) != sha:
                    raise ValueError('Source snapshot changed: ' + name)
            state = json.loads((root / 'state.json').read_text())
            _verify(root, state)
        else:
            atomic_json(root / 'study.json', manifest)
            repo = Path(__file__).resolve().parent.parent
            for name, sha in manifest['sources'].items():
                source = (repo / name).read_bytes()
                target = root / 'sources' / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source)
                if file_digest(target) != sha:
                    raise ValueError('Source changed during snapshot')
            state = {'status': 'running', 'scope': manifest['scope'],
                     'study_sha256': file_digest(root / 'study.json'),
                     'selected': None, 'best_score': None, 'rounds': [],
                     'evidence': [], 'qualification': None, 'physical_experiments': 0,
                     'self_harness_admitted': False, 'validation_passed': None}
            atomic_json(root / 'state.json', state)
        if state['status'] == 'completed':
            _reports(root, state)
            return state

        def evaluate(proposal, cases, split, index):
            base = root / split / f'{index:04d}'
            base.mkdir(parents=True, exist_ok=True)
            attempts = [p for p in base.iterdir() if p.is_dir()]
            out = base / f'attempt-{len(attempts):04d}'
            out.mkdir()
            rows = []
            started = time.monotonic()
            for i, spec in enumerate(cases):
                trial = out / str(i); trial.mkdir()
                actions = plan(Case(**spec), proposal['settings'])
                atomic_json(trial / 'actions.json', actions)
                protocol = trial / 'protocol.py'; protocol.write_text(protocol_source(actions))
                result = audit(Case(**spec), actions)
                try:
                    simulation = simulator(protocol, trial, trial_seconds)
                    if type(simulation.get('passed')) is not bool:
                        raise ValueError('Simulator must return a boolean passed field')
                except Exception as exc:
                    simulation = {'passed': False, 'error': type(exc).__name__ + ': ' + str(exc)}
                rows.append({'case': spec, 'audit': result, 'simulation': simulation,
                             'passed': result['passed'] and simulation['passed']})
            artifacts = {str(p.relative_to(out)): file_digest(p) for p in out.rglob('*') if p.is_file()}
            receipt = {'proposal': proposal, 'split': split, 'rows': rows, 'artifacts': artifacts,
                       'elapsed_seconds': time.monotonic() - started, 'qualification': None}
            atomic_json(out / 'receipt.json', receipt)
            ref = {'path': str((out / 'receipt.json').relative_to(root)), 'sha256': file_digest(out / 'receipt.json')}
            return receipt, ref

        completed = 0
        for index in range(len(state['rounds']), len(proposals)):
            if max_rounds is not None and completed >= max_rounds:
                break
            proposal = proposals[index]
            receipt, ref = evaluate(proposal, DEVELOPMENT, 'development', index)
            rows = receipt['rows']
            valid = all(r['passed'] for r in rows)
            score = [sum(r['audit'][k] for r in rows) for k in ('tips', 'primitive_commands')]
            status = 'keep' if valid and (state['best_score'] is None or score < state['best_score']) else 'discard'
            if any(not r['simulation']['passed'] for r in rows):
                status = 'error'
            parent = state['selected']
            if status == 'keep':
                state.update(selected=proposal['name'], best_score=score)
            state['rounds'].append({'round': index, 'candidate': proposal['name'], 'parent': parent,
                                    'status': status, 'tips': score[0], 'commands': score[1], 'receipt': ref['path']})
            state['evidence'].append(ref)
            atomic_json(root / 'state.json', state)
            _reports(root, state)
            completed += 1
        if len(state['rounds']) == len(proposals):
            if state['selected'] is not None:
                selected = next(p for p in proposals if p['name'] == state['selected'])
                receipt, ref = evaluate(selected, VALIDATION, 'validation', 0)
                state['evidence'].append(ref)
                state['validation_passed'] = all(r['passed'] for r in receipt['rows'])
            else:
                state['validation_passed'] = False
            state['status'] = 'completed'
            memory = {'selected': state['selected'], 'validation_passed': state['validation_passed'],
                      'evidence': state['evidence'], 'lesson': 'Retain only development improvements; validation is reported separately.',
                      'qualification': None, 'parent': None}
            state['memory_revision'] = digest(memory)
            atomic_json(root / 'memory' / (state['memory_revision'] + '.json'), memory)
            atomic_json(root / 'state.json', state)
            atomic_json(root / 'result.json', state)
        _reports(root, state)
        return state
