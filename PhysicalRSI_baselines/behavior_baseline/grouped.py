"""Explicit grouped native launches with one quota reservation per logical case.

This transport changes execution grouping, not the official task budget or score.
Development diagnostics remain single-instance. A group is frozen before effects;
case receipts describe the actual shared command rather than a fictitious launch.
"""
from dataclasses import asdict
import os
from pathlib import Path
import shutil
import signal
import subprocess

from PhysicalRSI_core.contracts import ReconciliationRequired
from PhysicalRSI_core.embodiment import System2Trial
from PhysicalRSI_core.infra.storage import atomic_json, digest, file_digest, locked, read_json
from . import native
from .trials import NativeTrials


def verify_case_binding(folder, receipt, specification):
    """Verify the case's membership in its bound completed shared execution."""
    folder = Path(folder)
    required = {'group-specification.json', 'group-completion.json'}
    if not required <= set(receipt['evidence']):
        raise ValueError('Grouped case lacks shared execution evidence')
    group = read_json(folder / 'group-specification.json')
    completion = read_json(folder / 'group-completion.json')
    membership = specification.get('group') or {}
    if (membership.get('specification_sha256') != digest(group)
            or completion.get('state') != 'completed'
            or completion.get('specification_sha256') != digest(group)):
        raise ValueError('Grouped execution identity differs from case receipt')
    slot = membership.get('slot')
    if type(slot) is not int or not 0 <= slot < len(group['members']):
        raise ValueError('Invalid grouped case slot')
    member = group['members'][slot]
    if member['case'] != specification['case']:
        raise ValueError('Grouped case differs from its frozen slot')
    for field in ('argv', 'source_revision', 'policy_revision', 'system2', 'seconds',
                  'graphics_manifests', 'temporary_directory', 'temporary_directory_explicit', 'implementation_sha256'):
        if specification[field] != group[field]:
            raise ValueError('Grouped case execution fields changed')
    if set(completion['results']) != {row['id'] for row in group['members']}:
        raise ValueError('Incomplete group completion')
    result = completion['results'][member['id']]
    if (result['native_result'] != receipt['native_result']
            or result['native_result_sha256'] != receipt['evidence'][receipt['native_result_file']]):
        raise ValueError('Grouped native result differs from shared completion')


def _read_group(folder, expected):
    receipt = read_json(folder / 'receipt.json')
    if receipt.get('state') != 'completed':
        raise ReconciliationRequired(f'Reconcile shared native execution before retry: {folder}')
    if read_json(folder / 'specification.json') != expected or receipt['specification_sha256'] != digest(expected):
        raise ValueError('Frozen native group changed')
    for relative, sha in receipt['evidence'].items():
        path = (folder / relative).resolve()
        if not path.is_relative_to(folder.resolve()) or file_digest(path) != sha:
            raise ValueError('Native group evidence changed')
    if set(receipt['results']) != {m['id'] for m in expected['members']}:
        raise ValueError('Incomplete native group')
    for member in expected['members']:
        row = receipt['results'][member['id']]
        result = native.parse_result(folder / row['native_result_file'],
            task=member['case']['task'], instance_id=member['case']['instance_id'])
        if result != row['native_result'] or receipt['evidence'].get(row['native_result_file']) != row['native_result_sha256']:
            raise ValueError('Native group result changed')
    return receipt


class GroupedNativeTrials(NativeTrials):
    """Full-budget validation/test groups; default NativeTrials stays single-case."""
    def __init__(self, runtime, output, *, trials, quota, group_size):
        if type(group_size) is not int or group_size < 2:
            raise ValueError('Declare an integer native group size of at least two')
        self.runtime, self.output = Path(runtime).resolve(), Path(output).resolve()
        self.quota = quota
        if quota.root.is_relative_to(self.output):
            raise ValueError('Keep mutable trial quota outside experiment evidence')
        if not isinstance(trials, list) or not trials:
            raise ValueError('Declare a complete nonempty trial plan')
        rows, buckets, seen = [], {}, set()
        for request in trials:
            inputs = dict(request)
            origin = inputs.get('system2')
            if (not isinstance(origin, System2Trial) or origin.split not in {'validation', 'test'}
                    or inputs.get('max_steps') is not None):
                raise ValueError('Grouped trials require full-budget validation or test attribution')
            inputs.update(policy='websocket', policy_revision=origin.candidate_sha256)
            identity = dict(inputs, system2=asdict(origin))
            key = digest(identity)
            spec = native.specification(self.runtime, self.output / 'trials' / key, **inputs)
            semantic = digest(dict(origin=asdict(origin), case=spec['case']))
            if semantic in seen:
                raise ValueError('Repeated trial in native plan')
            seen.add(semantic)
            row = dict(id=key, inputs=identity, specification=spec)
            rows.append(row)
            compatibility = {k: v for k, v in identity.items() if k != 'instance_index'}
            buckets.setdefault(digest(compatibility), []).append(row)
        groups = []
        for bucket in buckets.values():
            bucket.sort(key=lambda row: row['specification']['case']['instance_index'])
            for offset in range(0, len(bucket), group_size):
                members = bucket[offset:offset + group_size]
                group_id = digest(dict(members=[row['inputs'] for row in members], group_size=group_size))
                folder = self.output / 'groups' / group_id
                common = dict(members[0]['specification'])
                common.pop('case')
                common.update(schema='physicalrsi.behavior-native-group/v1',
                    implementation_sha256=file_digest(Path(__file__)),
                    temporary_directory_explicit=members[0]['inputs'].get('temporary_directory') is not None,
                    members=[dict(id=row['id'], case=row['specification']['case']) for row in members])
                argv = list(common['argv'])
                i = argv.index('--instance-indices') + 1
                argv[i:i + 1] = [str(row['case']['instance_index']) for row in common['members']]
                argv[argv.index('--num-envs') + 1] = str(len(members))
                argv[argv.index('--output-dir') + 1] = str(folder / 'native')
                common['argv'] = argv
                groups.append(dict(id=group_id, specification=common))
                for slot, row in enumerate(members):
                    case = row['specification']['case']
                    row['specification'] = {k: v for k, v in common.items() if k != 'members'}
                    row['specification'].update(schema='physicalrsi.behavior-native-group-case/v1', case=case,
                        group=dict(id=group_id, slot=slot, specification_sha256=digest(common)))
        self.plan = dict(schema='physicalrsi.behavior-native-group-plan/v1', runtime=str(self.runtime),
            output=str(self.output), quota=quota.identity(), trials=rows, groups=groups,
            group_size=group_size, implementation_sha256=file_digest(Path(__file__)), qualification=None)
        with locked(self.output / '.plan.lock'):
            path = self.output / 'plan.json'
            if path.exists() and read_json(path) != self.plan:
                raise ValueError('Frozen native trial plan changed')
            reservation = quota.reserve(self._requests())
            quota_path = self.output / 'quota.json'
            if quota_path.exists() and read_json(quota_path) != reservation:
                raise ValueError('Native plan quota receipt changed')
            if not path.exists():
                atomic_json(path, self.plan)
            atomic_json(quota_path, reservation)

    def _publish(self, group, receipt):
        """Recover missing case projections only from complete immutable group evidence."""
        source = self.output / 'groups' / group['id']
        for row in self.plan['trials']:
            if row['specification']['group']['id'] != group['id']:
                continue
            folder = self.output / 'trials' / row['id']
            if (folder / 'receipt.json').exists():
                native.read_receipt(folder)
                if read_json(folder / 'specification.json') != row['specification']:
                    raise ValueError('Case projection differs from group plan')
                continue
            folder.mkdir(parents=True, exist_ok=True)
            result = receipt['results'][row['id']]
            files = ['policy_service.json', 'evaluator.log', result['native_result_file']]
            case = row['specification']['case']
            video = f"native/videos/{case['task']}_{case['instance_id']}_0.mp4"
            if video in receipt['evidence']:
                files.append(video)
            for relative in files:
                target = folder / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    if file_digest(target) != receipt['evidence'][relative]:
                        raise ValueError('Interrupted case projection changed')
                else:
                    # Completed evidence is immutable by contract. Share large
                    # videos on the same filesystem; every read still hashes it.
                    try:
                        os.link(source / relative, target)
                    except OSError:
                        shutil.copyfile(source / relative, target)
                if file_digest(target) != receipt['evidence'][relative]:
                    raise ValueError('Case evidence changed during projection')
            atomic_json(folder / 'specification.json', row['specification'])
            atomic_json(folder / 'group-specification.json', group['specification'])
            atomic_json(folder / 'group-completion.json', receipt)
            evidence = {str(p.relative_to(folder)): file_digest(p) for p in folder.rglob('*') if p.is_file()}
            value = dict(schema='physicalrsi.behavior-native-group-case-receipt/v1', state='completed',
                specification_sha256=digest(row['specification']), native_result=result['native_result'],
                native_result_file=result['native_result_file'], evidence=evidence,
                outcome='success' if result['native_result']['success'] else 'failure',
                purpose='policy_evaluation', qualification=None)
            atomic_json(folder / 'receipt.json', value)
            native.read_receipt(folder)

    def _reconcile(self):
        if file_digest(Path(__file__)) != self.plan['implementation_sha256'] or read_json(self.output / 'plan.json') != self.plan:
            raise ValueError('Frozen native group executor or plan changed')
        if self.quota.identity() != self.plan['quota'] or self.quota.reserve(self._requests()) != read_json(self.output / 'quota.json'):
            raise ValueError('Native group quota changed')
        for group in self.plan['groups']:
            folder = self.output / 'groups' / group['id']
            if (folder / 'receipt.json').exists():
                self._publish(group, _read_group(folder, group['specification']))
            elif folder.exists() and any(p.name != '.lock' for p in folder.iterdir()):
                raise ReconciliationRequired(f'Incomplete native group directory: {folder}')
            else:
                for member in group['specification']['members']:
                    if (self.output / 'trials' / member['id']).exists():
                        raise ValueError('Case evidence exists without a completed native group')

    def run(self, index):
        if type(index) is not int or not 0 <= index < len(self.plan['trials']):
            raise ValueError('Trial index outside frozen plan')
        with locked(self.output / '.execution.lock'):
            self._reconcile()
            row = self.plan['trials'][index]
            group = next(g for g in self.plan['groups'] if g['id'] == row['specification']['group']['id'])
            folder = self.output / 'groups' / group['id']
            if not (folder / 'receipt.json').exists():
                self._run_group(folder, group['specification'])
            self._publish(group, _read_group(folder, group['specification']))
            return native.read_receipt(self.output / 'trials' / row['id'])

    def _run_group(self, folder, specification):
        source = self.runtime / 'sources/BEHAVIOR-1K'
        native.verify_source(source)
        temporary = specification['temporary_directory'] if specification['temporary_directory_explicit'] else None
        env, graphics = native.runtime_environment(self.runtime, temporary_directory=temporary)
        if graphics != specification['graphics_manifests'] or env['TMPDIR'] != specification['temporary_directory']:
            raise ValueError('Native group graphics changed')
        argv = specification['argv']
        metadata = native.probe_policy(argv[argv.index('--host') + 1], int(argv[argv.index('--port') + 1]), specification['policy_revision'])
        if metadata.get('task') != specification['members'][0]['case']['task'] or metadata.get('policy_identity', {}).get('diagnostics') is not None:
            raise ValueError('Grouped policy task differs or single-case diagnostics are enabled')
        atomic_json(folder / 'policy_service.json', metadata)
        atomic_json(folder / 'specification.json', specification)
        receipt = dict(schema='physicalrsi.behavior-native-group-receipt/v1', state='started',
            specification_sha256=digest(specification), qualification=None)
        atomic_json(folder / 'receipt.json', receipt)
        process = None
        try:
            with (folder / 'evaluator.log').open('x') as log:
                process = subprocess.Popen(argv, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                atomic_json(folder / 'receipt.json', dict(receipt, pid=process.pid))
                if process.wait(timeout=specification['seconds']) != 0:
                    raise RuntimeError('Native group failed; inspect evaluator.log')
            native.verify_source(source)
            results = {}
            for member in specification['members']:
                case = member['case']
                relative = f"native/json/{case['task']}_{case['instance_id']}_0.json"
                results[member['id']] = dict(native_result_file=relative,
                    native_result=native.parse_result(folder / relative, task=case['task'], instance_id=case['instance_id']),
                    native_result_sha256=file_digest(folder / relative))
            evidence = {str(p.relative_to(folder)): file_digest(p) for p in folder.rglob('*')
                        if p.is_file() and p.name not in {'receipt.json', '.lock'}}
            atomic_json(folder / 'receipt.json', dict(receipt, state='completed', results=results, evidence=evidence))
        except BaseException as error:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
            atomic_json(folder / 'receipt.json', dict(receipt, state='unresolved', error_type=type(error).__name__,
                reason=str(error), pid=process.pid if process else None))
            raise
