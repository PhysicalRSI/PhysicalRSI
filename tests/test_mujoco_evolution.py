from pathlib import Path

import pytest

pytest.importorskip("mujoco")

from PhysicalRSI_core.infra.storage import digest, file_digest, read_json
from PhysicalRSI_core.lineage import HarnessState
from PhysicalRSI_demos.mujoco_evolution import run


@pytest.mark.parametrize("proposal_strategy", ["reviewed", "enumerated"])
def test_native_campaign_commits_measured_improvement_then_reuses_receipts(tmp_path, proposal_strategy):
    result = run(tmp_path, proposal_strategy=proposal_strategy)
    assert result["status"] == "completed" and result["completed_rounds"] == 2
    assert result["lineage_committed"] is True and result["qualification"] is None
    assert result["harness"] == "feedback" if proposal_strategy == "reviewed" else result["harness"].startswith("proposal-")
    assert result["selection"]["metrics"][result["harness"]]["success_rate"] == {"slider": 1, "hinge": 1}
    assert result["selection"]["metrics"]["zero-effort"]["success_rate"] == {"slider": 0, "hinge": 0}
    assert [r["outcome"] for r in result["rounds"]] == ["inherited", "retained"]
    assert result["quota"] == dict(max_trials=12, reserved_trials=12, remaining_trials=0)
    state = HarnessState(tmp_path / "state")
    child = state.resolve()
    parent = state.read(child["previous"])
    assert child["harness"]["components"]["foundation"] == parent["harness"]["components"]["foundation"]
    if proposal_strategy == "reviewed":
        assert child["harness"]["costs"]["training_steps"] == 0
    else:
        assert child["harness"]["costs"]["reserved_calls"] == 1
        call = read_json(tmp_path / "campaign/rounds/round-0001/proposal/call.json")
        assert call["binding"]["proposer"]["strategy"]["kind"] == "enumerated-json-edits/v1"
    assert read_json(Path(child["harness"]["root"]) / "control.json") == dict(mode="feedback")
    round_root = tmp_path / "campaign/rounds/round-0001"
    development = read_json(round_root / "develop.json")["output"]
    cohort = read_json(round_root / "validation.json")["output"]
    assert not {digest(row["case"]) for row in development["episodes"]}.intersection(
        sha for cases in cohort["layouts"].values() for sha in cases)
    for path in (round_root / "experiments/validation").rglob("experiment.json"):
        trial = read_json(path)
        assert trial["system2"]["split"] == "validation"
        assert trial["binding"]["system1"]["revision"] == trial["system2"]["candidate_sha256"]
        assert trial["binding"]["embodiment"]["mode"] == "simulation"
    requests = {str(p): file_digest(p) for p in (tmp_path / "controllers").glob("*/requests/*.json")}
    authorities = {str(p): file_digest(p) for p in (tmp_path / "controllers").glob("*/authority.json")}
    assert len(authorities) == 2
    assert all(read_json(path)["current"] is None and read_json(path)["generation"] == 6 for path in authorities)
    assert requests
    assert run(tmp_path, proposal_strategy=proposal_strategy) == result
    assert {str(p): file_digest(p) for p in (tmp_path / "controllers").glob("*/requests/*.json")} == requests
    assert {str(p): file_digest(p) for p in (tmp_path / "controllers").glob("*/authority.json")} == authorities


def test_isolated_system1_and_system2_complete_native_campaign_with_separate_evaluation(tmp_path, isolation_runtime):
    result = run(tmp_path, proposal_strategy="isolated", isolation_runtime=isolation_runtime)
    assert result["status"] == "completed" and result["lineage_committed"] and result["qualification"] is None
    assert [row["outcome"] for row in result["rounds"]] == ["inherited", "retained"]
    assert result["selection"]["metrics"][result["harness"]]["success_rate"] == {"slider": 1, "hinge": 1}
    assert result["selection"]["metrics"]["zero-effort"]["success_rate"] == {"slider": 0, "hinge": 0}
    assert result["system1_isolation"]["kind"] == "linux-python-data-isolation/v1"
    from test_isolated_policy import assert_stopped
    assert_stopped(tmp_path / "isolated-system1")
    assert_stopped(tmp_path / "isolated-system2")
    cleanup = list((tmp_path / "campaign").rglob("policy.json"))
    assert len(cleanup) == 12
    for path in cleanup:
        record = read_json(path)
        assert record["stopped"] and record["worker_error"] is None
        receipt = read_json(path.parent / "receipt.json")
        assert receipt["evidence"]["policy.json"] == file_digest(path)
    calls = list((tmp_path / "isolated-system2").glob("*/result.json"))
    assert len(calls) == 2
    before = {str(p): file_digest(p) for p in (tmp_path / "controllers").glob("*/requests/*.json")}
    assert run(tmp_path, proposal_strategy="isolated", isolation_runtime=isolation_runtime) == result
    assert before == {str(p): file_digest(p) for p in (tmp_path / "controllers").glob("*/requests/*.json")}
    assert list((tmp_path / "isolated-system2").glob("*/result.json")) == calls
