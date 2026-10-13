"""Read-only timing summaries of verified, completed Trials.

Summed worker time is not campaign wall time: jobs may overlap, and queue waits,
terminal receipt publication and later archival work are outside Trial timers.
No rollout, selection, quota reservation or recovery is performed here.
"""
from math import fsum, isclose
from statistics import mean

from .infra.storage import digest
from .timing import finite_seconds
from .trials import Evaluation


def profile_trials(sources):
    """Summarize (ExperimentRuntime, run_id) pairs, auditing their evidence.

    Legacy receipts without phase timing still contribute elapsed time. Their
    missing breakdown is explicit, never replaced with zero-duration phases.
    Duplicate evidence is rejected rather than counted as independent execution.
    Interrupted Trials must be reconciled separately before using this API.
    """
    rows, seen, executions = [], set(), set()
    for runtime, run_id in sources:
        value = Evaluation.read(runtime, run_id).to_dict()
        receipt = value['receipt']
        key = (value['trial_sha256'], digest(receipt))
        execution_id = receipt.get('execution_id')
        if key in seen or (execution_id is not None and execution_id in executions):
            raise ValueError('Timing profile requires distinct Trial executions')
        seen.add(key)
        if execution_id is not None:
            executions.add(execution_id)
        elapsed = finite_seconds(receipt['elapsed_seconds'], positive=False)
        timing = receipt.get('timing')
        phases = None
        if 'timing' in receipt:
            if (not isinstance(timing, dict) or timing.get('schema') != 'physicalrsi.trial-timing/v1'
                    or not isinstance(timing.get('phase_seconds'), dict)
                    or set(timing['phase_seconds']) != {
                        'preparation', 'reset', 'rollout', 'verification', 'finalization', 'cleanup'}):
                raise ValueError('Unknown or incomplete Trial timing breakdown')
            phases = {name: finite_seconds(seconds, positive=False)
                      for name, seconds in timing['phase_seconds'].items()}
            # Phase accumulation and elapsed subtraction round separately.
            if not isclose(fsum(phases.values()), elapsed, rel_tol=1e-12, abs_tol=1e-9):
                raise ValueError('Trial timing phases disagree with elapsed time')
        rows.append(dict(runtime_root=str(runtime.root), run_id=run_id, task=value['trial']['task'],
            scope=value['trial']['scope'], outcome=value['outcome'], trial_sha256=key[0],
            receipt_sha256=key[1], execution_id=execution_id, elapsed_seconds=elapsed, phase_seconds=phases))
    if not rows:
        raise ValueError('Timing profile requires completed Trials')

    def aggregate(selected):
        timed = [row for row in selected if row['phase_seconds'] is not None]
        phases = sorted({phase for row in timed for phase in row['phase_seconds']})
        return dict(trials=len(selected), phase_timed_trials=len(timed),
            missing_phase_trials=len(selected)-len(timed),
            elapsed_seconds=dict(total=fsum(row['elapsed_seconds'] for row in selected),
                mean=float(mean(row['elapsed_seconds'] for row in selected)),
                maximum=max(row['elapsed_seconds'] for row in selected)),
            phase_seconds={phase: dict(observed_trials=sum(phase in row['phase_seconds'] for row in timed),
                total=fsum(row['phase_seconds'][phase] for row in timed if phase in row['phase_seconds']))
                for phase in phases})

    return dict(schema='physicalrsi.trial-profile/v1', trials=rows, summary=aggregate(rows),
                by_task={task: aggregate([row for row in rows if row['task'] == task])
                         for task in sorted({row['task'] for row in rows})},
                interpretation='summed_trial_time_not_campaign_wall_time', qualification=None)
