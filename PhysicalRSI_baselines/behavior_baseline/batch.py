"""Supervise local OpenPI services for an already reserved native trial plan.

This connects transport lifetimes, not benchmark admission or survivor selection.
The caller must admit candidates and preregister complete comparison cohorts.
"""
from pathlib import Path

from PhysicalRSI_core.infra.storage import atomic_json, digest, file_digest, locked, read_json
from PhysicalRSI_core.self_harness.artifacts import verify_harness
from . import native, service
from .trials import NativeTrials
from .openpi_backend import supported_tasks


def execute(trials, *, candidates, python, startup_seconds=1800, episode_seconds=3600):
    """Run all missing trials, preserving completed evidence and interrupted services.

    Candidate paths are keyed by their attributed candidate IDs. A service is
    started once per candidate/task/endpoint group. Interrupted launches require
    explicit reconciliation; this function never invents a retry identity.
    """
    if not isinstance(trials, NativeTrials):
        raise TypeError("A complete quota-reserved NativeTrials plan is required")
    expected = {row['inputs']['system2']['candidate_id'] for row in trials.plan['trials']}
    if set(candidates) != expected:
        raise ValueError("Candidate registry differs from the complete native plan")
    bound, groups = {}, {}
    for name, path in candidates.items():
        path = Path(path).resolve()
        candidate = read_json(path)
        if candidate['id'] != name:
            raise ValueError("Candidate registry ID differs from its artifact")
        bound[name] = dict(path=str(path), revision=verify_harness(candidate))
    for index, row in enumerate(trials.plan['trials']):
        inputs = row['inputs']
        origin = inputs['system2']
        name, task = origin['candidate_id'], inputs['task']
        if origin['candidate_sha256'] != bound[name]['revision']:
            raise ValueError("Candidate revision differs from reserved trial")
        host, port = inputs.get('host', '127.0.0.1'), inputs.get('port', 8000)
        if host != '127.0.0.1':
            raise ValueError("Local service orchestration requires a loopback endpoint")
        candidate = read_json(bound[name]['path'])
        backend = read_json(Path(candidate['root']) / 'backend.json')
        if task not in supported_tasks(backend):
            raise ValueError("Candidate backend does not cover the planned task")
        groups.setdefault((name, task, port), []).append(index)
    identity = dict(plan_sha256=digest(trials.plan), candidates=bound,
                    python=str(Path(python).absolute()), startup_seconds=startup_seconds,
                    episode_seconds=episode_seconds,
                    implementations={p.name: file_digest(p) for p in (
                        Path(__file__), Path(service.__file__), Path(native.__file__))})
    with locked(trials.output / '.batch.lock'):
        trials.reconcile()
        path = trials.output / 'batch.json'
        if path.exists() and read_json(path) != identity:
            raise ValueError("Frozen native batch orchestration changed")
        if not path.exists():
            atomic_json(path, identity)
        for (name, task, port), indices in groups.items():
            pending = [i for i in indices if not (
                trials.output / 'trials' / trials.plan['trials'][i]['id'] / 'receipt.json').exists()]
            if not pending:
                continue
            trials.reconcile()
            launch = trials.output / 'services' / digest(dict(candidate=name, task=task, port=port))
            with service.policy_service(trials.runtime, bound[name]['path'], launch,
                    python=python, task=task, port=port, startup_seconds=startup_seconds,
                    episode_seconds=episode_seconds) as endpoint:
                if endpoint != dict(host='127.0.0.1', port=port, policy_revision=bound[name]['revision']):
                    raise ValueError("Launched service differs from frozen trial endpoint")
                for index in pending:
                    trials.run(index)
        trials.reconcile()
        return [native.read_receipt(trials.output / 'trials' / row['id'])
                for row in trials.plan['trials']]
