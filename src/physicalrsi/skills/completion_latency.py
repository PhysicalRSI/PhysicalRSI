"""Empirical latency to an observed effect, calibrated outside the rollout.

Samples must measure completed transitions on a compatible embodiment and
action route. Censored or absent effects must not be encoded as durations.
"""
import math
import statistics


def completion_latency(*, observed_seconds):
    from pathlib import Path
    from PhysicalRSI_core.infra.storage import digest, file_digest
    from .protocol import CallableSkill, SkillDescriptor
    if (not isinstance(observed_seconds, (list, tuple)) or not observed_seconds or
            any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0
                for x in observed_seconds)):
        raise ValueError('Expected nonempty positive finite completed-effect durations')
    samples = tuple(float(x) for x in observed_seconds)
    estimate = statistics.median(samples)
    metadata = dict(parameters=dict(observed_seconds=list(samples)), estimator='median',
                    qualification='Empirical completed-effect latency; not a deadline guarantee or cross-embodiment calibration')
    revision = digest(dict(source=file_digest(Path(__file__)), metadata=metadata))
    return CallableSkill(SkillDescriptor('completion_latency', revision,
        dict(name='completion_latency_request/v1'),
        dict(name='completion_latency_estimate/v1', unit='SI'), metadata=metadata),
        lambda value, context: dict(seconds=estimate, samples=len(samples)))
