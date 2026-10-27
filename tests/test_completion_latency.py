import pytest
from PhysicalRSI_core.contracts import Context
from physicalrsi.skills.completion_latency import completion_latency


def test_robust_latency_uses_completed_durations_and_freezes_input():
    samples = [1., 1.2, 1.4, 20.]
    skill = completion_latency(observed_seconds=samples)
    samples[0] = 100.
    result = skill.execute({}, Context('latency'))
    assert result['seconds'] == pytest.approx(1.3)
    assert result['samples'] == 4
    assert skill.revision != completion_latency(observed_seconds=[1., 1.2, 1.4]).revision


@pytest.mark.parametrize('samples', [[], [0], [-1], [True], [float('inf')], [float('nan')], [None]])
def test_absent_censored_or_invalid_samples_are_not_durations(samples):
    with pytest.raises(ValueError):
        completion_latency(observed_seconds=samples)
