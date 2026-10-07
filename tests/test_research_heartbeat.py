import pytest

from PhysicalRSI_Autoresearch.heartbeat import ResearchHeartbeat
from PhysicalRSI_core.infra.storage import file_digest


def test_heartbeat_cadence_pending_coalescing_and_evidenced_review(tmp_path):
    timer = ResearchHeartbeat(tmp_path / "heartbeat")
    first = timer.tick(now=100)
    assert first["pending"]
    assert timer.tick(now=2000)["pending"] == first["pending"]
    assert len(list((timer.root / "requests").glob("*.json"))) == 1
    evidence = tmp_path / "audit.json"
    evidence.write_text('{}')
    review = dict(evidence={str(evidence): file_digest(evidence)}, reflection="No measured gain.",
        alternatives="Test public temporal evidence instead of another threshold change.",
        decision="Reject the regressing candidate.", next_experiment="Fresh paired layouts.")
    result = timer.complete(first["pending"], review, now=2000)
    assert result["qualification"] is None
    assert timer.tick(now=3699)["pending"] is None
    assert timer.tick(now=3700)["pending"]
    with pytest.raises(ValueError, match="pending"):
        timer.complete(first["pending"], review, now=3800)


def test_no_empty_or_corrupted_review_and_no_silent_interval_change(tmp_path):
    timer = ResearchHeartbeat(tmp_path / "heartbeat")
    request = timer.tick(now=100)["pending"]
    with pytest.raises(ValueError, match="Review needs"):
        timer.complete(request, {}, now=200)
    with pytest.raises(ValueError, match="interval changed"):
        ResearchHeartbeat(timer.root, interval_seconds=60).tick(now=200)
    assert timer.tick(now=4000)["pending"] == request
