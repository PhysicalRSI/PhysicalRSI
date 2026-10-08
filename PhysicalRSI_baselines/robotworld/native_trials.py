"""Durable candidate-bound RoboCasa launches for the native evaluator port.

The worker is trusted deployment code, responsible for actually loading the
candidate, enforcing isolation and preserving the upstream task protocol.
Hashes and a binding receipt identify that execution; they do not independently
prove an arbitrary worker is honest. Admission remains a separate gate.
"""
from copy import deepcopy
import math
from pathlib import Path
import sys
import time

from PhysicalRSI_core.contracts import ReconciliationRequired
from PhysicalRSI_core.infra import processes, storage
from PhysicalRSI_core.infra.storage import atomic_json, digest, file_digest, locked, read_json
from PhysicalRSI_core.self_harness import artifacts
from PhysicalRSI_core.self_harness.artifacts import verify_harness
from . import receipts


def validate_case(case):
    required = {'task', 'launcher_seed', 'reset_seed', 'task_index', 'horizon',
                'task_set', 'split', 'trial_slot'}
    if not isinstance(case, dict) or set(case) != required:
        raise ValueError('Declare the complete native reset and repeated-trial identity')
    if not isinstance(case['task'], str) or not case['task']:
        raise ValueError('Declare the native task')
    for key in ('launcher_seed', 'reset_seed', 'task_index', 'horizon', 'trial_slot'):
        if type(case[key]) is not int or case[key] < 0:
            raise ValueError('Native reset and budget fields must be nonnegative integers')
    if (case['horizon'] < 1 or case['reset_seed'] != case['launcher_seed'] + case['task_index']
            or case['task_set'] != 'all_tasks' or case['split'] != 'pretrain'):
        raise ValueError('Expected upstream one-episode default reset protocol')


class RoboCasaTrials:
    def __init__(self, *, worker, deployment, python=sys.executable):
        self.worker = Path(worker).resolve(strict=True)
        self.deployment = Path(deployment).resolve(strict=True)
        self.python = Path(python).resolve(strict=True)
        self._identity = deepcopy(self.identity())

    def identity(self):
        return dict(kind='robotworld-native-robocasa-trials/v1',
                    episode_count=1, episode_start=0, code_control=False,
                    files={str(p): file_digest(p) for p in (
                        self.worker, self.deployment, self.python, Path(__file__),
                        Path(receipts.__file__), Path(artifacts.__file__),
                        Path(processes.__file__), Path(storage.__file__))})

    def _stable(self):
        if self.identity() != self._identity:
            raise ValueError('Native trial implementation changed')

    def _result(self, root, request):
        native = root / 'native'
        binding = native / 'execution-binding.json'
        if native.is_symlink() or binding.is_symlink() or read_json(binding) != {'request_sha256': digest(request)}:
            raise ValueError('Native worker did not bind the requested candidate and case')
        case = request['case']
        result = receipts.robocasa_receipt(native, task=case['task'], seed=case['reset_seed'],
                                           horizon=case['horizon'])
        config_path = next(native.glob('*/episode-*/evaluation-config.json'))
        config = read_json(config_path)
        expected = dict(task_set=case['task_set'], split=case['split'],
                        task_index=case['task_index'], global_episode_index=case['task_index'])
        if any(config.get(key) != value or type(config.get(key)) is not type(value)
               for key, value in expected.items()):
            raise ValueError('Native registry/split differs from the launch plan')
        evidence = {'native/' + name: sha for name, sha in result['evidence'].items()}
        evidence['native/execution-binding.json'] = file_digest(binding)
        return dict(result, candidate_sha256=request['candidate_sha256'],
                    case_sha256=digest(case), evidence=evidence)

    def run(self, candidate, case, output, *, seconds):
        self._stable()
        validate_case(case)
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('Declare a finite positive launch deadline')
        sha = verify_harness(candidate)
        root = Path(output).resolve()
        request = dict(schema='physicalrsi.robotworld-native-trial/v1',
                       candidate=deepcopy(candidate), candidate_sha256=sha,
                       case=deepcopy(case), runner=self._identity, seconds=seconds)
        with locked(root / '.trial.lock'):
            journal = root / 'trial.json'
            if journal.exists():
                record = read_json(journal)
                if record['request_sha256'] != digest(request) or read_json(root / 'request.json') != request:
                    raise ValueError('Native launch inputs changed')
                if record['state'] != 'completed':
                    raise ReconciliationRequired('Reconcile the prior native launch; automatic replay is forbidden')
                result = self._result(root, request)
                if digest(record['result']) != record['result_sha256'] or result != record['result']:
                    raise ValueError('Completed native evidence changed')
                return result
            if any(root.iterdir()):
                # The lock is the only file this invocation may have created.
                if any(p.name != '.trial.lock' for p in root.iterdir()):
                    raise ValueError('Native launch output is not empty')
            atomic_json(root / 'request.json', request)
            record = dict(state='started', request_sha256=digest(request), started_at=time.time())
            atomic_json(journal, record)
            worker = processes.ManagedProcess('robotworld-native-trial', [
                str(self.python), str(self.worker), '--deployment', str(self.deployment),
                '--request', str(root / 'request.json'), '--output', str(root / 'native')],
                cwd=root, log_path=root / 'worker.log', env_overrides={
                    'PYTHONPATH': str(Path(__file__).resolve().parents[2]), 'PYTHONDONTWRITEBYTECODE': '1'})
            started = time.monotonic()
            try:
                worker.start()
                record['pid'] = worker.pid
                atomic_json(journal, record)
                while worker.poll() is None:
                    remaining = seconds - (time.monotonic() - started)
                    if remaining <= 0:
                        raise TimeoutError('Native launch deadline reached')
                    time.sleep(min(.05, remaining))
                record.update(state='exited', returncode=worker.poll())
                atomic_json(journal, record)
                if time.monotonic() - started >= seconds:
                    raise TimeoutError('Native launch deadline reached')
                if worker.poll() != 0:
                    raise RuntimeError('Native worker exited without a scored episode')
                worker.stop(timeout=1)
                self._stable()
                if verify_harness(candidate) != sha:
                    raise ValueError('Candidate changed during native execution')
                result = self._result(root, request)
                record.update(state='completed', result=result, result_sha256=digest(result),
                              ended_at=time.time())
                atomic_json(journal, record)
                return result
            except Exception as error:
                record.update(state='unresolved', error_type=type(error).__name__, ended_at=time.time())
                atomic_json(journal, record)
                raise ReconciliationRequired('Native trial has no validated result; inspect its journal and logs') from None
            finally:
                worker.stop(timeout=1)
