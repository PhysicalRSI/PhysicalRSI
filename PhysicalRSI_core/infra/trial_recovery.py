"""Explicit replacement of a stopped Trial that never reached reset readiness.

Authorization is an operator attestation, not an automatic crash detector. A
replacement has a new identity, an additional reservation, and owned copies of
the failed attempt and stopping evidence. Completed outcomes are never replaced.
"""
from contextlib import contextmanager
from copy import deepcopy
import fcntl
from pathlib import Path
import shutil

from ..contracts import ReconciliationRequired
from .evidence import EvidenceReader
from .storage import atomic_json, digest, file_digest, identifier, locked, read_json, relative_path


@contextmanager
def inactive(path):
    """Require a stopped local owner instead of waiting for one to finish."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ReconciliationRequired('Cannot replace a live execution: ' + str(path)) from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _inventory(root):
    root = Path(root)
    if root != root.resolve():
        raise ValueError('Recovery requires an owned physical directory')
    result = {}
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError('Recovery cannot snapshot symlinks')
        if path.is_file() and not path.name.endswith('.lock'):
            result[str(path.relative_to(root))] = file_digest(path)
    return result


def _copy(source, destination, expected):
    if file_digest(source) != expected:
        raise ValueError('Recovery source evidence changed')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if file_digest(destination) != expected:
            raise ReconciliationRequired('Incomplete recovery snapshot changed: ' + str(destination))
    else:
        shutil.copy2(source, destination)
    if file_digest(destination) != expected or file_digest(source) != expected:
        raise ValueError('Recovery evidence changed during capture')


def base_specification(specification):
    return {k: v for k, v in specification.items() if k != 'replacement'}


def reservations(directory, plan):
    return {digest(dict(directory=str(directory), run_id=row['run_id'])):
            dict(plan_sha256=digest(plan), run_id=row['run_id']) for row in plan['trials']}


def _matches_plan(specification, original, plan):
    rows = [row for row in plan['trials'] if row['run_id'] == original]
    if len(rows) != 1 or specification['scope'] != plan['scope']:
        raise ValueError('Replacement lacks its original planned Trial')
    if any(specification[k] != rows[0][k] for k in ('task', 'case', 'budget')):
        raise ValueError('Failed Trial conditions differ from the frozen plan')
    origin = specification.get('system2')
    if origin is not None and any(origin[k] != plan[k] for k in
            ('candidate_id', 'candidate_sha256', 'split', 'comparison_sha256', 'cohort_sha256')):
        raise ValueError('Failed Trial attribution differs from the frozen plan')


def audit_bundle(reader, run_id, specification, *, prefix=''):
    """Read only owned evidence, including when an archive outlives its source."""
    reference = specification['replacement']
    if not isinstance(reference, dict) or set(reference) != {'authorization_sha256'}:
        raise ValueError('Invalid replacement reference')
    base = prefix + 'recovery/'
    name = base + 'authorization.json'
    record = reader.document(name, reference['authorization_sha256'])
    if (record.get('schema') != 'physicalrsi.trial-replacement/v1'
            or record['run_id'] != run_id or record['predecessor_id'] == run_id
            or type(record['index']) is not int or type(record['maximum']) is not int
            or not 1 <= record['maximum'] <= 10 or not 1 <= record['index'] <= record['maximum']
            or record['quiescent'] is not True or not record['operator'] or not record['reason']
            or record['base_experiment_sha256'] != digest(base_specification(specification))):
        raise ValueError('Replacement differs from its authorized execution')
    for key in ('run_id', 'predecessor_id', 'original_id'):
        identifier(record[key])
    files = record['files']
    if not isinstance(files, dict) or not {'source/receipt.json', 'source/experiment.json'} <= set(files):
        raise ValueError('Replacement lacks its failed predecessor')
    for relative in files:
        if str(relative_path(relative)) != relative or relative == 'authorization.json':
            raise ValueError('Invalid recovery evidence path')
    reader.verify({base + p: sha for p, sha in files.items()})
    previous = reader.document(base + 'source/receipt.json', files['source/receipt.json'])
    previous_spec = reader.document(base + 'source/experiment.json', files['source/experiment.json'])
    if (previous['state'] not in {'started', 'needs_reconciliation'} or 'outcome' in previous
            or previous.get('execution_phase') != 'reset'
            or previous['id'] != record['predecessor_id']
            or previous['experiment_sha256'] != digest(previous_spec)
            or base_specification(previous_spec) != base_specification(specification)
            or {'source/reset.json', 'source/pending-action.json'}.intersection(files)):
        raise ValueError('Only an interrupted attempt before reset readiness can be replaced')
    if digest(record['plan']) != record['plan_sha256']:
        raise ValueError('Replacement plan identity changed')
    _matches_plan(previous_spec, record['original_id'], record['plan'])
    if 'source/trajectory.json' in files and reader.document(base + 'source/trajectory.json'):
        raise ValueError('An observed action cannot be discarded by replacement')
    if not record['proofs'] or any(files.get(row['owned']) != row['sha256'] for row in record['proofs'].values()):
        raise ValueError('Replacement lacks stopping evidence')
    if 'source/lease.json' in files:
        lease = reader.document(base + 'source/lease.json')
        resolved = reader.document(base + 'device-reconciliation.json', files.get('device-reconciliation.json'))
        if (resolved['state'] != 'reconciled' or resolved.get('owner_active') is not False
                or any(resolved[k] != lease[k] for k in ('token', 'owner', 'resources', 'registry'))
                or lease['registry'] != previous_spec['device_registry']):
            raise ValueError('Replacement lacks reconciled device ownership')
    if previous_spec.get('replacement'):
        previous_record, _ = audit_bundle(reader, record['predecessor_id'], previous_spec, prefix=base + 'source/')
        if previous_record['original_id'] != record['original_id'] or previous_record['index'] + 1 != record['index']:
            raise ValueError('Replacement ancestry changed')
    elif record['index'] != 1 or record['predecessor_id'] != record['original_id']:
        raise ValueError('Replacement has no original attempt')
    request, reservation = record['quota_request'], record['quota_receipt']
    body = {key: value for key, value in record.items() if key not in {'quota_request', 'quota_receipt'}}
    expected_request = {digest(dict(directory=record['directory'], run_id=run_id)):
        dict(plan_sha256=record['plan_sha256'], run_id=run_id, authorization_inputs_sha256=digest(body))}
    if (reservation.get('schema') != 'physicalrsi.trial-reservations/v1'
            or request != expected_request
            or reservation['trials'] != {key: digest(value) for key, value in request.items()}
            or len(request) != 1):
        raise ValueError('Replacement quota evidence changed')
    original_quota = record['original_quota_receipt']
    if (original_quota.get('schema') != 'physicalrsi.trial-reservations/v1'
            or original_quota['quota'] != reservation['quota']
            or original_quota['trials'] != {key: digest(value) for key, value in reservations(record['directory'], record['plan']).items()}):
        raise ValueError('The original failed attempt lacks its quota evidence')
    return record, {name, *(base + p for p in files)}


def before_execution(folder, run_id, specification, device_registry):
    """Check live authorization and predecessor bytes before starting effects."""
    from .trial_quota import TrialQuota
    reader = EvidenceReader(folder)
    record, names = audit_bundle(reader, run_id, specification)
    source = Path(record['predecessor_root'])
    with inactive(source / '.lock'):
        expected = {name.removeprefix('source/'): sha for name, sha in record['files'].items() if name.startswith('source/')}
        if _inventory(source) != expected:
            raise ValueError('Failed Trial changed since replacement authorization')
        if (source / 'lease.json').exists():
            lease = read_json(source / 'lease.json')
            if device_registry is None or device_registry.identity() != lease['registry']:
                raise ValueError('Replacement device registry changed')
            current = device_registry.inspect(lease['token'])
            if current['state'] != 'reconciled' or current['owner_active']:
                raise ReconciliationRequired('Replacement device owner is not reconciled')
    identity = record['quota_receipt']['quota']
    quota = TrialQuota(identity['root'], max_trials=identity['max_trials'])
    if quota.verify(record['quota_request']) != record['quota_receipt']:
        raise ValueError('Replacement has no exact additional quota reservation')
    return names


class TrialRecovery:
    def __init__(self, directory, plan, *, quota, device_registry=None, maximum=0):
        if type(maximum) is not int or not 0 <= maximum <= 10:
            raise ValueError('Declare zero to ten replacement attempts per planned Trial')
        self.directory, self.plan = Path(directory).resolve(), deepcopy(plan)
        self.root = self.directory / 'trials'
        self.quota, self.device_registry, self.maximum = quota, device_registry, maximum
        self.path = self.directory / 'replacements.json'

    def _ledger(self):
        expected = dict(schema='physicalrsi.trial-replacements/v1', plan_sha256=digest(self.plan), maximum=self.maximum)
        if not self.path.exists():
            return dict(expected, attempts={})
        ledger = read_json(self.path)
        if any(ledger.get(k) != v for k, v in expected.items()) or not isinstance(ledger['attempts'], dict):
            raise ValueError('Frozen replacement allowance or plan changed')
        if set(ledger['attempts']) - {r['run_id'] for r in self.plan['trials']}:
            raise ValueError('Replacement is outside the declared plan')
        return ledger

    def _chain(self, original, ledger):
        entries = ledger['attempts'].get(original, [])
        if not isinstance(entries, list) or len(entries) > self.maximum:
            raise ValueError('Replacement allowance exceeded')
        previous, result = original, []
        for index, entry in enumerate(entries, 1):
            folder = self.root / identifier(entry['run_id'])
            reader = EvidenceReader(folder)
            record = reader.document('recovery/authorization.json', entry['authorization_sha256'])
            previous_spec = reader.document('recovery/source/experiment.json')
            spec = dict(base_specification(previous_spec), replacement={'authorization_sha256': entry['authorization_sha256']})
            checked, _ = audit_bundle(reader, entry['run_id'], spec)
            if (checked != record or record['predecessor_id'] != previous or record['original_id'] != original
                    or record['plan_sha256'] != digest(self.plan) or record['maximum'] != self.maximum
                    or record['index'] != index
                    or self.quota.verify(record['quota_request']) != record['quota_receipt']):
                raise ValueError('Replacement no longer belongs to its frozen plan')
            previous = entry['run_id']
            result.append(record)
        return result

    def resolve(self, original):
        if original not in {r['run_id'] for r in self.plan['trials']}:
            raise ValueError('Unknown planned Trial')
        ledger = self._ledger()
        chain = self._chain(original, ledger)
        if not chain:
            return dict(run_id=original, replacement=None, attempts=0)
        entry = ledger['attempts'][original][-1]
        return dict(run_id=entry['run_id'], replacement={'authorization_sha256': entry['authorization_sha256']}, attempts=len(chain))

    def authorize(self, original, failed_id, *, operator, reason, evidence, quiescent):
        if (not self.maximum or quiescent is not True or not isinstance(operator, str) or not operator.strip()
                or not isinstance(reason, str) or not reason.strip() or not isinstance(evidence, dict) or not evidence):
            raise ValueError('Replacement requires a frozen allowance, operator, reason and stopping evidence')
        identifier(failed_id)
        proofs = {}
        for i, (name, sha) in enumerate(sorted(evidence.items())):
            source = Path(name).resolve()
            if file_digest(source) != sha:
                raise ValueError('Stopping evidence changed')
            proofs[str(source)] = dict(owned='proof/' + str(i), sha256=sha)
        with locked(self.directory / '.recovery.lock'):
            ledger = self._ledger()
            chain = self._chain(original, ledger)
            for record in chain:
                if record['predecessor_id'] == failed_id:
                    if any(record[k] != v for k, v in dict(operator=operator, reason=reason, proofs=proofs).items()):
                        raise ValueError('Replacement authorization changed')
                    return record['run_id']
            current = self.resolve(original)
            if current['run_id'] != failed_id or current['attempts'] >= self.maximum:
                raise ReconciliationRequired('Replacement target is stale or its allowance is exhausted')
            source = self.root / failed_id
            with inactive(source / '.lock'):
                files = _inventory(source)
                receipt, spec = read_json(source / 'receipt.json'), read_json(source / 'experiment.json')
                if (receipt['state'] not in {'started', 'needs_reconciliation'} or 'outcome' in receipt
                        or receipt.get('execution_phase') != 'reset'
                        or receipt['id'] != failed_id or receipt['experiment_sha256'] != digest(spec)
                        or {'reset.json', 'pending-action.json'}.intersection(files)
                        or ('trajectory.json' in files and read_json(source / 'trajectory.json'))):
                    raise ValueError('Only an interrupted attempt before reset readiness can be replaced')
                _matches_plan(spec, original, self.plan)
                original_quota = self.quota.verify(reservations(self.directory, self.plan))
                index = current['attempts'] + 1
                run_id = 'replacement-' + digest(dict(directory=str(self.directory), original=original, index=index))
                destination = self.root / run_id / 'recovery'
                if (destination.parent / 'receipt.json').exists():
                    raise ReconciliationRequired('Unpublished replacement has execution evidence')
                captured = {'source/' + name: sha for name, sha in files.items()}
                for name, sha in files.items():
                    _copy(source / name, destination / 'source' / name, sha)
                for name, row in proofs.items():
                    _copy(Path(name), destination / row['owned'], row['sha256'])
                    captured[row['owned']] = row['sha256']
                if 'lease.json' in files:
                    lease = read_json(source / 'lease.json')
                    if self.device_registry is None or self.device_registry.identity() != lease['registry']:
                        raise ValueError('Declare the original device registry for recovery')
                    resolved = self.device_registry.inspect(lease['token'])
                    if resolved['state'] != 'reconciled' or resolved['owner_active']:
                        raise ReconciliationRequired('Reconcile the stopped device lease before replacement')
                    atomic_json(destination / 'device-reconciliation.json', resolved)
                    captured['device-reconciliation.json'] = file_digest(destination / 'device-reconciliation.json')
                body = dict(schema='physicalrsi.trial-replacement/v1', original_id=original, predecessor_id=failed_id,
                    predecessor_root=str(source), run_id=run_id, index=index, maximum=self.maximum,
                    directory=str(self.directory), plan=self.plan, original_quota_receipt=original_quota,
                    plan_sha256=digest(self.plan), base_experiment_sha256=digest(base_specification(spec)),
                    operator=operator, reason=reason, quiescent=True, proofs=proofs, files=captured)
                request = {digest(dict(directory=str(self.directory), run_id=run_id)):
                           dict(plan_sha256=digest(self.plan), run_id=run_id, authorization_inputs_sha256=digest(body))}
                reserved = self.quota.reserve(request)
                record = dict(body, quota_request=request, quota_receipt=reserved)
                path = destination / 'authorization.json'
                if path.exists() and read_json(path) != record:
                    raise ReconciliationRequired('Prepared replacement authorization changed')
                atomic_json(path, record)
                if _inventory(source) != files:
                    raise ValueError('Failed Trial changed while authorizing replacement')
                audit_bundle(EvidenceReader(destination.parent), run_id,
                             dict(base_specification(spec), replacement={'authorization_sha256': file_digest(path)}))
                ledger['attempts'].setdefault(original, []).append(dict(run_id=run_id, authorization_sha256=file_digest(path)))
                atomic_json(self.path, ledger)
                return run_id
