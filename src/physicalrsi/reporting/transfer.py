"""Task-independent paired transfer metrics. Exposure is declared, never inferred."""
from math import comb, isfinite


def evaluate_transfer(parent, candidate, *, training_tasks, selection_tasks,
                      alpha=0.05, minimum_gain=0.0):
    """Compare mappings of (task, case_id) to boolean success.

    Case IDs must include scenario and seed. Call once per frozen candidate and
    prespecified task family. Exact one-sided paired tests use discordant pairs;
    Bonferroni correction covers all tasks in this call. Post-hoc results remain
    descriptive, regardless of the gate. Missing/error trials must be resolved
    explicitly before calling; never encode infrastructure errors as failures.
    """
    if not 0 < alpha < 1 or not isfinite(minimum_gain) or not 0 <= minimum_gain <= 1:
        raise ValueError('Invalid alpha or minimum_gain')
    if not parent or parent.keys() != candidate.keys():
        raise ValueError('Nonempty, identical paired case sets required')
    for arm in (parent, candidate):
        for key, value in arm.items():
            if (not isinstance(key, tuple) or len(key) != 2
                    or not all(isinstance(x, str) and x for x in key)
                    or type(value) is not bool):
                raise ValueError('Expected (task, case_id) strings and boolean outcomes')
    training, selection = set(training_tasks), set(selection_tasks)
    tasks = sorted({k[0] for k in parent})
    rows = {}
    for task in tasks:
        keys = [k for k in parent if k[0] == task]
        n = len(keys)
        wins = sum(not parent[k] and candidate[k] for k in keys)
        losses = sum(parent[k] and not candidate[k] for k in keys)
        discordant = wins + losses
        p = sum(comb(discordant, i) for i in range(wins, discordant + 1)) / 2**discordant
        adjusted = min(1.0, p * len(tasks))
        gain = (wins - losses) / n
        rows[task] = dict(pairs=n, parent_success=sum(parent[k] for k in keys),
                          candidate_success=sum(candidate[k] for k in keys),
                          gain=gain, wins=wins, losses=losses, ties=n-discordant,
                          improvement_p=p, adjusted_p=adjusted,
                          benefit_supported=gain > minimum_gain and adjusted <= alpha,
                          heldout_task=task not in training | selection)
    unseen = [r for r in rows.values() if r['heldout_task']]
    gains = [r['gain'] for r in rows.values()]
    return dict(schema='physicalrsi.transfer/v1', tasks=rows,
                macro_gain=sum(gains)/len(tasks), worst_task_gain=min(gains),
                positive_task_fraction=sum(g > 0 for g in gains)/len(tasks),
                regression_task_fraction=sum(g < 0 for g in gains)/len(tasks),
                all_tasks_benefit=all(r['benefit_supported'] for r in rows.values()),
                heldout_task_count=len(unseen),
                heldout_transfer_supported=(all(r['benefit_supported'] for r in unseen) if unseen else None),
                training_tasks=sorted(training), selection_tasks=sorted(selection),
                alpha=alpha, minimum_gain=minimum_gain,
                inference='Exact paired one-sided tests; Bonferroni across tasks. Requires independent cases; post-hoc use is descriptive.')
