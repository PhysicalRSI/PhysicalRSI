import pytest
from pathlib import Path
import shutil
import sys

pytest.importorskip("mujoco")

from PhysicalRSI_core.embodiment import System1
from PhysicalRSI_core.experiments import Budget, ExperimentRuntime
from PhysicalRSI_core.infra.actuation_rpc import ActuationClient
from PhysicalRSI_core.infra.clock_sync import ClockSyncPolicy
from PhysicalRSI_core.infra.controller import GatewayEnvironment
from PhysicalRSI_core.infra.controller_rpc import controller_transport
from PhysicalRSI_core.infra.devices import DeviceRegistry
from PhysicalRSI_core.infra.services import ServiceSpec, Services
from PhysicalRSI_core.infra.storage import file_digest, read_json
from PhysicalRSI_core.timing import action_chunk
from PhysicalRSI_demos.mujoco_control import JointVerifier
from PhysicalRSI_demos.mujoco_watchdog import MODULE, SCOPE, TIMING, context, make_backend, make_controller, run


@pytest.mark.parametrize("model", ["slider", "hinge"])
def test_independent_actuator_survives_controller_death_and_measures_stop(tmp_path, model):
    result = run(tmp_path, model=model)
    assert result["controller_process_killed"] and result["backend_process_survived"]
    assert result["continued_simulation"]
    assert result["observed_motion"]["quiescence"]["effort"] == [1.]
    assert result["observed_motion"]["quiescence"]["measured_speed"] > .02
    assert result["after"]["quiescence"]["effort"] == [0.]
    assert result["after"]["quiescence"]["measured_speed"] <= .02
    assert result["after"]["ownership"]["phase"] == "recovery_required"
    assert result["competing_controller"]["state"] == "rejected"
    assert result["trial_outcome"] == "uncertain" and result["qualification"] is None
    assert read_json(tmp_path / "controller/requests/killed-command.json")["state"] == "started"
    with pytest.raises(ValueError, match="fresh workspace"):
        run(tmp_path, model=model)


@pytest.mark.parametrize("clock_offset_seconds", [0, -120])
def test_normal_experiment_waits_for_measured_settling_and_releases_both_authorities(tmp_path, clock_offset_seconds):
    probe = make_backend(tmp_path / "backend", "slider", clock_offset_seconds=clock_offset_seconds)
    expected = probe.identity()
    probe.close()
    service = ServiceSpec("backend", expected, command=(sys.executable, "-m", MODULE,
        "--serve", "backend", "--workspace", str(tmp_path), "--model", "slider", "--clock-offset-seconds", str(clock_offset_seconds)))
    with Services(tmp_path / "services", client_factory=controller_transport) as services:
        rpc = services.start([service])["backend"]
        client = ActuationClient(rpc, expected=expected, clock_policy=ClockSyncPolicy() if clock_offset_seconds else None)
        gateway = make_controller(tmp_path / "controller", "slider", client)
        try:
            spec = gateway.embodiment

            class Pulse:
                def identity(self):
                    return dict(name="one-bounded-pulse", source=file_digest(Path(__file__)))

                def describe(self):
                    return System1("pulse", file_digest(Path(__file__)), spec.observation, spec.action, timing=TIMING)

                def begin_episode(self, *args):
                    pass

                def act(self, observation, context):
                    return action_chunk(observation, [[.5]], period_seconds=TIMING.period_seconds)

            runtime = ExperimentRuntime(tmp_path / "trials", device_registry=DeviceRegistry(tmp_path / "devices"))
            receipt = runtime.run("pulse", task="slider", case=dict(initial=0., target=.3,
                position_tolerance=.02, velocity_tolerance=.05), scope=SCOPE,
                environment=GatewayEnvironment(gateway), policy=Pulse(), verifier=JointVerifier(), budget=Budget(1, 10))
            # A completed low-level motion is not a successful target-reaching
            # trial. The independent verifier keeps that distinction visible.
            assert receipt["outcome"] == "failure"
            assert gateway.ownership()["phase"] == "free"
            stopped = client.inspect(context())
            assert stopped["quiescence"]["quiescent"] and stopped["ownership"]["phase"] == "free"
            assert stopped["quiescence"]["effort"] == [0.]
            assert stopped["quiescence"]["measured_speed"] <= .02
            assert runtime.read("pulse") == receipt
        finally:
            gateway.close()


@pytest.mark.parametrize("clock_offset_seconds", [0, 120])
def test_explicit_operator_recovery_restarts_controller_without_reclassifying_unknown_trial(tmp_path, clock_offset_seconds):
    result = run(tmp_path, recover_after_fault=True, clock_offset_seconds=clock_offset_seconds, clock_sync=bool(clock_offset_seconds))
    recovery = result["operator_recovery"]
    assert recovery["backend"]["state"] == recovery["controller"]["state"] == "completed"
    assert recovery["new_session"]["grant"]["claim"]["generation"] == 2
    assert recovery["new_session"]["action"]["control"]["executed_steps"] == 1
    assert recovery["new_session"]["stop"]["quiescent"] is True
    assert recovery["final_backend"]["ownership"]["phase"] == recovery["final_controller"]["phase"] == "free"
    assert recovery["final_backend"]["quiescence"]["quiescent"] is True
    assert recovery["original_trial_outcome"] == result["trial_outcome"] == "uncertain"
    assert recovery["qualification"] is result["qualification"] is None
    assert read_json(tmp_path / "result.json")["trial_outcome"] == "uncertain"
    assert read_json(tmp_path / "controller/requests/killed-command.json")["state"] == "started"
    assert read_json(tmp_path / "recovery-result.json") == recovery
    if clock_offset_seconds:
        observation = result["initial_observation"]
        assert observation["capture_uncertainty_seconds"] > 0
        assert observation["timing_evidence"]["source"]["captured_at"] > observation["captured_at"] + 119
        motion = recovery["new_session"]["action"]
        evidence = motion["driver"]["actuation"]["timing_evidence"]
        assert evidence["local_deadline"] == motion["control"]["deadline"]
        assert evidence["local_deadline"] + 119 < evidence["remote_deadline"] <= evidence["local_deadline"] + 120


@pytest.mark.skipif(not shutil.which("openssl"), reason="OpenSSL CLI required for ephemeral simulation TLS")
def test_mutual_tls_both_rpc_hops_preserve_clock_mapping_watchdog_and_local_recovery(tmp_path, monkeypatch):
    from PhysicalRSI_demos import mujoco_watchdog
    original, directories = mujoco_watchdog._tls_fixture, []

    def ephemeral(directory):
        directories.append(directory)
        return original(directory)

    monkeypatch.setattr(mujoco_watchdog, "_tls_fixture", ephemeral)
    result = run(tmp_path, recover_after_fault=True, clock_offset_seconds=120, clock_sync=True, tls_fixture=True)
    assert result["qualification"] is None and result["trial_outcome"] == "uncertain"
    assert result["controller_process_killed"] and result["backend_process_survived"]
    recovery = result["operator_recovery"]
    assert recovery["new_session"]["grant"]["claim"]["generation"] == 2
    assert recovery["new_session"]["action"]["control"]["executed_steps"] == 1
    assert recovery["final_controller"]["phase"] == recovery["final_backend"]["ownership"]["phase"] == "free"
    assert recovery["final_backend"]["quiescence"]["quiescent"]
    controller, actuator = (result["transport_security"][key] for key in ("controller", "actuator"))
    assert controller["client_sha256"] != actuator["client_sha256"]
    assert controller["server_sha256"] != actuator["server_sha256"]
    for kind, security in (("controller", controller), ("backend", actuator)):
        announcements = list((tmp_path / (kind + "-service")).glob("*/" + kind + ".endpoint.json"))
        assert len(announcements) == 1 and read_json(announcements[0])["endpoint"].startswith("https://")
        audit = [read_json(path) for path in (tmp_path / kind / "transport/access").glob("*.json")]
        assert audit and all(row["peer_sha256"] == security["client_sha256"] and row["authorized"] for row in audit)
        assert all(row["principal"] == "simulation-" + kind + "-client" for row in audit)
    assert not list(tmp_path.rglob("*.key"))
    assert not list(tmp_path.rglob("*.pem"))
    assert directories and all(not directory.exists() for directory in directories)
    motion = recovery["new_session"]["action"]
    timing = motion["driver"]["actuation"]["timing_evidence"]
    assert timing["local_deadline"] == motion["control"]["deadline"]
    assert timing["local_deadline"] + 119 < timing["remote_deadline"] <= timing["local_deadline"] + 120
