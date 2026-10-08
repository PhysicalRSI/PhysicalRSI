"""Native RoboCasa receipts for later Self-Harness evaluation; no agent scoring."""
import json
from pathlib import Path

from PhysicalRSI_core.infra.storage import file_digest


def robocasa_receipt(artifacts, *, task, seed, horizon):
    """Read one complete native episode and its independently recorded config.

    A returned failure is a completed task failure, never a missing experiment.
    Probe, timeout and malformed records fail closed before selection. This
    validates artifact structure, not the trustworthiness of an arbitrary
    producer; the evaluator must separately bind a pinned launcher/candidate.
    """
    root = Path(artifacts).resolve(strict=True)
    paths = list(root.glob('*/episode-*/episode.json'))
    if len(paths) != 1:
        raise ValueError('Expected exactly one native RoboCasa episode')
    result_path = paths[0]
    config_path = result_path.with_name('evaluation-config.json')
    for p in (result_path, config_path):
        if p.is_symlink() or not p.resolve().is_relative_to(root):
            raise ValueError('Native receipt escapes its artifact root')
    config = json.loads(config_path.read_text())
    result = json.loads(result_path.read_text())
    if type(seed) is not int or type(horizon) is not int or horizon < 1:
        raise ValueError('Declare an integer reset seed and positive native horizon')
    if (config.get('task') != task or config.get('seed') != seed
            or result.get('seed') != seed or config.get('horizon') != horizon
            or config.get('official_horizon') != horizon
            or config.get('horizon_override') is not None):
        raise ValueError('Native reset identity or horizon differs from evaluation')
    if config.get('probe_only') is not False or result.get('probe_only') is not False:
        raise ValueError('Probe or undeclared evaluation cannot qualify a policy')
    if any(result.get(k) for k in ('diagnostic_only', 'not_model_performance')):
        raise ValueError('Diagnostic result cannot qualify a policy')
    success, steps = result.get('success'), result.get('steps')
    if type(success) is not bool or type(steps) is not int or not 0 <= steps <= horizon:
        raise ValueError('Invalid native success or executed step count')
    termination = result.get('termination')
    if termination not in {'success', 'done', 'truncated', 'horizon'}:
        raise ValueError('Unscored or incomplete native termination')
    if termination == 'success' and not success:
        raise ValueError('Contradictory native success termination')
    if termination == 'horizon' and steps != horizon:
        raise ValueError('Native horizon was not reached')
    return dict(scope='native_simulator_receipt', qualification=None, task=task,
                seed=seed, horizon=horizon, steps=steps,
                outcome='success' if success else 'failure', score=int(success),
                termination=termination,
                evidence={str(p.relative_to(root)): file_digest(p)
                          for p in (result_path, config_path)})
