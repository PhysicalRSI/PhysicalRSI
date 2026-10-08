"""Native interpreter entry point; use upstream reset, controls and evaluator.

Only stdlib imports precede the upstream environment imports so the simulator
venv does not need the physicalRSI application's dependencies.
"""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('request', 'output', 'manifest', 'policy-inputs'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    case = request['case']
    context = json.loads((args.policy_inputs / 'context.json').read_text())
    if context['candidate_sha256'] != request['candidate_sha256']:
        raise ValueError('Policy input mount differs from the requested candidate')
    from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
    from robocasa.utils.dataset_registry_utils import get_task_horizon
    tasks = list(TASK_SET_REGISTRY[case['task_set']])
    if (tasks.index(case['task']) != case['task_index']
            or get_task_horizon(case['task']) != case['horizon']
            or case['reset_seed'] != case['launcher_seed'] + case['task_index']):
        raise ValueError('Task registry, native reset or horizon differs from the frozen request')
    from environment.benchmarks.robocasa import policy
    policy.BASE += ('\nFrozen physicalRSI inputs are read-only in /policy/context.json. '
                    'That file names reusable Python skills and the initial memory snapshot. '
                    'Use them for local computation and reusable knowledge; all robot motion '
                    'and fresh observations still require the existing robot tools. '
                    'Keep working notes in /workspace. Supplemental policy guidance:\n' + context['guidance'])
    from environment.integrations import robocasa_eval
    sys.argv = ['robocasa_eval', '--output', str(args.output), '--manifest', str(args.manifest),
                '--task-name', case['task'], '--task-set', case['task_set'], '--split', case['split'],
                '--num-trials', '1', '--episode-start', '0', '--seed', str(case['launcher_seed'])]
    robocasa_eval.main()


if __name__ == '__main__':
    main()
