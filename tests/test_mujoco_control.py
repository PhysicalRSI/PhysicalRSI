import pytest

pytest.importorskip("mujoco")

from PhysicalRSI_core.experiments import ExperimentRuntime
from PhysicalRSI_core.infra.storage import digest, read_json
from PhysicalRSI_core.timing import observation_reference
from PhysicalRSI_demos.mujoco_control import MuJoCoDriver, run


def test_two_embodiments_run_through_controller_processes_and_paired_selection(tmp_path):
    result = run(tmp_path)
    assert result["qualification"] is None and result["lineage_committed"] is False
    assert result["selection"]["survivor_id"] == "feedback"
    assert result["selection"]["metrics"]["feedback"]["success_rate"] == {"slider": 1, "hinge": 1}
    assert result["selection"]["metrics"]["zero-effort"]["success_rate"] == {"slider": 0, "hinge": 0}
    runtime = ExperimentRuntime(tmp_path / "trials")
    for mode in ("zero-effort", "feedback"):
        for model in ("slider", "hinge"):
            for index in range(2):
                name = f"{mode}-{model}-{index}"
                receipt = runtime.read(name)
                assert receipt["binding"]["embodiment"]["mode"] == "simulation"
                assert receipt["binding"]["embodiment"]["action"]["unit"] == ("N" if model == "slider" else "N*m")
                reset = read_json(tmp_path / "trials" / name / "reset.json")
                stop = read_json(tmp_path / "trials" / name / "quiescence.json")
                assert reset["control"]["authority"]["generation"] == stop["ownership_release"]["generation"]
                trace = read_json(tmp_path / "trials" / name / "trajectory.json")
                total_steps = 0
                for previous, row in zip(trace, trace[1:]):
                    total_steps += row["control"]["executed_steps"]
                    assert row["control"]["source"] == observation_reference(previous["observation"])
                    assert row["control"]["observation_age_seconds"] < 2
                    assert row["control"]["completed_at"] <= row["control"]["deadline"]
                assert trace[-1]["observation"]["payload"]["simulation_seconds"] == pytest.approx(total_steps * .02)
    before = digest(result)
    authorities = {model: read_json(tmp_path / "controllers" / model / "authority.json") for model in ("slider", "hinge")}
    assert all(record["current"] is None and record["generation"] == 4 for record in authorities.values())
    # Fresh controller processes resume verified receipts without new resets.
    assert digest(run(tmp_path)) == before
    assert {model: read_json(tmp_path / "controllers" / model / "authority.json") for model in authorities} == authorities


@pytest.mark.parametrize("model,limit", [("slider", 10), ("hinge", 3)])
def test_simulator_adapter_rejects_invalid_effort_before_actuation(model, limit):
    driver = MuJoCoDriver(model)
    for actions in ([[limit + 1]], [[float("nan")]], [[True]], [[0, 0]], [["1"]]):
        with pytest.raises(ValueError, match="actuator limits"):
            driver.validate(actions)
    assert float(driver.data.time) == 0
