"""Stop repeating an interaction whose observable effect already holds.

Adapters supply a current boolean predicate. Safe release/withdrawal remains
the caller's responsibility; this decision does not certify task completion.
"""


def effect_decision(*, effect_observed, execution_ok):
    if type(effect_observed) is not bool or type(execution_ok) is not bool:
        raise ValueError('Effect and execution status must be booleans')
    if not execution_ok:
        return dict(decision='abort', reason='execution_failed')
    return dict(decision='finish_safe' if effect_observed else 'continue',
                reason='effect_observed' if effect_observed else 'effect_pending')


def effect_guard():
    from pathlib import Path
    from PhysicalRSI_core.infra.storage import file_digest
    from .protocol import CallableSkill, SkillDescriptor
    return CallableSkill(SkillDescriptor('effect_guard', file_digest(Path(__file__)),
        dict(name='observed_effect/v1'), dict(name='effect_guard_decision/v1'),
        metadata=dict(qualification='Current effect predicate only; caller must complete safe release and verify persistence')),
        lambda value, context: effect_decision(**value))
