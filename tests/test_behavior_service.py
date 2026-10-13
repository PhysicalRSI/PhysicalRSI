import os
import sys

import pytest

from PhysicalRSI_baselines.behavior_baseline import service
from PhysicalRSI_core.infra.storage import atomic_json, file_digest, read_json
from PhysicalRSI_core.self_harness.artifacts import CLOSURE


@pytest.mark.parametrize("wrong_owner", [False, True])
@pytest.mark.parametrize("diagnostics_enabled", [False, True])
def test_owned_process_is_reaped_on_success_and_handshake_rejection(tmp_path, monkeypatch, wrong_owner, diagnostics_enabled):
    # Real child process; handshake fixture does not establish model readiness.
    root = tmp_path / "candidate"
    root.mkdir()
    atomic_json(root / "backend.json", {"kind": "openpi-b1k/v1", "task": "radio"})
    candidate = dict(id="fixture", root=str(root), components={
        key: {"backend.json": file_digest(root / "backend.json")} for key in CLOSURE})
    atomic_json(root / "candidate.json", candidate)
    monkeypatch.setattr(service, "runtime_environment", lambda _: (os.environ.copy(), {}))
    monkeypatch.setattr(service, "command", lambda *a, **k: [sys.executable, "-c", "import time; time.sleep(60)"])
    folder = tmp_path / "run"
    binding = None
    if diagnostics_enabled:
        from PhysicalRSI_core.self_harness.artifacts import verify_harness
        from PhysicalRSI_core.self_harness.protocol import CaseProtocol
        from PhysicalRSI_core.infra.storage import digest
        def case(i):
            return dict(task="radio", mode="train", instance_id=i, instance_index=i, rollout_id=0)
        protocol = CaseProtocol(tmp_path / "cases", identity_keys=("instance_id",), splits={
            split: {"radio": [case(i)]} for split, i in [("development", 1), ("validation", 200), ("test", 301)]})
        binding = dict(schema="physicalrsi.behavior-development-binding/v1", split="development",
                       task="radio", candidate_sha256=verify_harness(candidate), case=case(1),
                       protocol_path=str(protocol.root / "protocol.json"), protocol_sha256=protocol.revision)

    def handshake(*args, **kwargs):
        assert 0 < kwargs["timeout"] <= 5
        state = read_json(folder / "service.json")
        result = dict(service_pid=state["pid"] + int(wrong_owner),
                      launch_nonce=state["launch_nonce"], task="radio")
        if binding is not None:
            result["policy_identity"] = dict(diagnostics=dict(
                schema="physicalrsi.behavior-development-trace/v1", split="development",
                task="radio", candidate_sha256=binding["candidate_sha256"],
                case_id=digest(["radio", {"instance_id": 1}]), path=str(folder / "development.jsonl")))
        return result

    monkeypatch.setattr(service, "probe_policy", handshake)
    manager = service.policy_service(tmp_path, root / "candidate.json", folder,
                                     python=sys.executable, task="radio", startup_seconds=5,
                                     development_binding=binding)
    if wrong_owner:
        with pytest.raises(ValueError, match="not this launched"):
            with manager:
                pytest.fail("An unrelated endpoint must never be accepted")
    else:
        with manager as endpoint:
            assert endpoint["host"] == "127.0.0.1"
            assert read_json(folder / "service.json")["state"] == "ready"
    state = read_json(folder / "service.json")
    assert state["state"] == "stopped"
    assert state["returncode"] is not None
    with pytest.raises(ProcessLookupError):
        os.kill(state["pid"], 0)
    with pytest.raises(FileExistsError, match="previous service launch"):
        with service.policy_service(tmp_path, root / "candidate.json", folder,
                                    python=sys.executable, task="radio"):
            pytest.fail("A launch evidence folder must not be overwritten")
