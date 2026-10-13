import pytest

from PhysicalRSI_baselines.behavior_baseline.native import make_command, parse_result, read_receipt
from PhysicalRSI_core.contracts import ReconciliationRequired
from PhysicalRSI_core.infra.storage import atomic_json, digest, file_digest


@pytest.mark.parametrize("split,diagnostic_index,accepted", [
    ("development", 1, True), ("development", 2, False), ("validation", 1, False), (None, 1, False)])
def test_diagnostics_are_bound_to_development_before_native_effects(tmp_path, monkeypatch, split, diagnostic_index, accepted):
    from PhysicalRSI_baselines.behavior_baseline import native
    from PhysicalRSI_core.embodiment import Revision, System2Trial
    attribution = (System2Trial(Revision("fixture", "b" * 64), "candidate", "a" * 64, split,
                                comparison_sha256="c" * 64, cohort_sha256="d" * 64)
                   if split else None)
    metadata = dict(task="radio", policy_identity=dict(diagnostics=dict(
        split="development", candidate_sha256="a" * 64, task="radio",
        case_id=digest(["radio", {"instance_id": diagnostic_index}]))))
    monkeypatch.setattr(native, "verify_source", lambda *_: None)
    monkeypatch.setattr(native, "probe_policy", lambda *_, **__: metadata)
    def no_native_effect(*args, **kwargs):
        assert accepted, "Invalid diagnostic attribution reached native launch"
        raise RuntimeError("fixture reached launch")
    monkeypatch.setattr(native.subprocess, "Popen", no_native_effect)
    with pytest.raises(RuntimeError if accepted else ValueError,
                       match="fixture reached launch" if accepted else "diagnostics"):
        native.run(tmp_path, tmp_path / "trial", task="radio", instance_index=1,
                   policy="websocket", policy_revision="a" * 64, system2=attribution)


def test_preregistered_budget_change_is_rejected_before_external_effects(tmp_path, monkeypatch):
    from PhysicalRSI_baselines.behavior_baseline import native

    def forbidden(*args, **kwargs):
        pytest.fail("Mismatched plan must fail before source checks, connection or process launch")

    kwargs = dict(task="radio", instance_index=201, policy="websocket", policy_revision="a" * 64)
    folder = tmp_path / "trial"
    planned = native.specification(tmp_path, folder, seconds=3600, **kwargs)
    assert planned["case"] == dict(task="radio", mode="train", instance_index=201,
                                    instance_id=201, rollout_id=0)
    monkeypatch.setattr(native, "verify_source", forbidden)
    monkeypatch.setattr(native, "probe_policy", forbidden)
    monkeypatch.setattr(native.subprocess, "Popen", forbidden)
    with pytest.raises(ValueError, match="preregistered"):
        native.run(tmp_path, folder, seconds=10, expected_specification=planned, **kwargs)
    assert not folder.exists()


def test_explicit_temporary_storage_is_frozen_before_effects(tmp_path, monkeypatch):
    from PhysicalRSI_baselines.behavior_baseline import native
    scratch = tmp_path / "local-scratch"
    scratch.mkdir()
    env, _ = native.runtime_environment(tmp_path, temporary_directory=scratch)
    assert env["TMPDIR"] == str(scratch)
    assert env["XDG_CACHE_HOME"] == str(tmp_path / "cache")
    plan = native.specification(tmp_path, tmp_path / "run", task="radio", temporary_directory=scratch)
    assert plan["temporary_directory"] == str(scratch)
    monkeypatch.setattr(native, "verify_source", lambda *a: pytest.fail("Storage mismatch must precede effects"))
    with pytest.raises(ValueError, match="preregistered"):
        native.run(tmp_path, tmp_path / "run", task="radio", expected_specification=plan)
    with pytest.raises(ValueError, match="already exist"):
        native.runtime_environment(tmp_path, temporary_directory=tmp_path / "absent")


def test_handshake_budget_bounds_a_connected_but_silent_policy():
    import threading
    import time
    from websockets.sync.server import serve
    from PhysicalRSI_baselines.behavior_baseline.native import probe_policy

    release = threading.Event()
    with serve(lambda connection: release.wait(5), "127.0.0.1", 0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.socket.getsockname()[1]
        started = time.monotonic()
        try:
            with pytest.raises(TimeoutError):
                probe_policy("127.0.0.1", port, "fixture", timeout=.3)
            assert time.monotonic() - started < 1.5
        finally:
            release.set()
            server.shutdown()
            thread.join(timeout=2)


def test_command_uses_native_entrypoint_without_hidden_or_injected_arguments(tmp_path):
    kwargs = dict(task="turning_on_radio", mode="train", instance_index=1, policy="local", host="127.0.0.1", port=8000)
    command = make_command(tmp_path, tmp_path / "result", **kwargs)
    assert command[1:3] == ["-m", "omnigibson.eval.eval"]
    assert "--max-steps" not in command
    with pytest.raises(ValueError):
        make_command(tmp_path, tmp_path, **dict(kwargs, mode="hidden_test"))
    with pytest.raises(ValueError):
        make_command(tmp_path, tmp_path, **dict(kwargs, task="--help"))


def test_native_result_must_match_case_and_have_real_score(tmp_path):
    path = tmp_path / "result.json"
    result = dict(task="radio", instance_id=301, rollout_id=0, success=False, q_score={"final": .5})
    atomic_json(path, result)
    assert parse_result(path, task="radio", instance_id=301) == result
    with pytest.raises(ValueError, match="another episode"):
        parse_result(path, task="radio", instance_id=302)
    atomic_json(path, dict(result, q_score={"final": 2}))
    with pytest.raises(ValueError, match="Q-score"):
        parse_result(path, task="radio", instance_id=301)


def test_receipt_rejects_unresolved_jobs_and_tampered_native_evidence(tmp_path):
    atomic_json(tmp_path / "receipt.json", {"state": "started", "pid": 123})
    with pytest.raises(ReconciliationRequired):
        read_receipt(tmp_path)
    spec = {"task": "radio"}
    atomic_json(tmp_path / "specification.json", spec)
    atomic_json(tmp_path / "native.json", {"success": False})
    receipt = dict(state="completed", specification_sha256=digest(spec),
                   evidence={"native.json": file_digest(tmp_path / "native.json")},
                   native_result_file="native.json", native_result={"success": False}, outcome="failure")
    atomic_json(tmp_path / "receipt.json", receipt)
    assert read_receipt(tmp_path) == receipt
    atomic_json(tmp_path / "native.json", {"success": True})
    with pytest.raises(ValueError, match="evidence changed"):
        read_receipt(tmp_path)
