import hashlib
import json
from pathlib import Path
import sys
import time

import pytest

from PhysicalRSI_baselines.embodied_goodharts_law import aspire_ik
from PhysicalRSI_baselines.embodied_goodharts_law.persistent_aspire_ik import PersistentAspireHandIK, IKNonConvergence


def solver(tmp_path, monkeypatch, worker):
    model = tmp_path / "model.urdf"
    model.write_text("protocol-test-model")
    monkeypatch.setattr(aspire_ik, "PANDA_SHA256", hashlib.sha256(model.read_bytes()).hexdigest())
    script = tmp_path / "worker.py"
    script.write_text(worker)

    class ProtocolWorker(PersistentAspireHandIK):
        def _command(self):
            return [sys.executable, str(script)]

    return ProtocolWorker(worker_directory=tmp_path / "owned-worker", interpreter=sys.executable,
        urdf=model, native_hand_translation=[0, 0, 0], native_hand_quaternion_wxyz=[1, 0, 0, 0])


def call(ik, seconds=5):
    return ik(pose=[0, 0, 0, 1, 0, 0, 0], joints=[0] * 7,
              gripper_fraction=1.01, deadline=time.monotonic() + seconds)


def test_one_worker_handles_multiple_requests_and_confirms_clean_exit(tmp_path, monkeypatch):
    ik = solver(tmp_path, monkeypatch, '''import sys,json
for line in sys.stdin:
 r=json.loads(line)
 print(json.dumps({'id':r['id'],'status':'solved','value':{'joints':r['joints']}}),flush=True)
''')
    first = call(ik)
    pid = ik._process.pid
    assert call(ik) == first and ik._process.pid == pid
    assert first["gripper_seed"] == {"measured_fraction": 1.01, "model_fraction": 1., "clipped": True}
    receipt = ik.close()
    assert receipt["stopped"] and receipt["returncode"] == 0 and receipt["requests"] == 2
    assert not Path("/proc", str(pid)).exists()


def test_deadline_kills_and_reaps_worker_before_returning(tmp_path, monkeypatch):
    ik = solver(tmp_path, monkeypatch, "import time\ntime.sleep(60)\n")
    with pytest.raises(TimeoutError):
        call(ik, .1)
    receipt = json.loads((ik.directory / "process.json").read_text())
    assert receipt["stopped"] and receipt["forced"] and receipt["returncode"] < 0
    assert not Path("/proc", str(receipt["pid"])).exists()
    closed = ik.close()
    assert closed["forced"]
    assert json.loads((ik.directory / "process.json").read_text()) == receipt


def test_nonconvergence_is_a_completed_solver_reply_not_a_worker_crash(tmp_path, monkeypatch):
    ik = solver(tmp_path, monkeypatch, '''import sys,json
for line in sys.stdin:
 r=json.loads(line)
 print(json.dumps({'id':r['id'],'status':'nonconverged'}),flush=True)
''')
    try:
        with pytest.raises(IKNonConvergence):
            call(ik)
        assert ik._process.poll() is None
        with pytest.raises(IKNonConvergence):
            call(ik)
        assert ik._index == 2
    finally:
        assert ik.close()["returncode"] == 0


def test_wrong_reply_identity_stops_worker(tmp_path, monkeypatch):
    ik = solver(tmp_path, monkeypatch, '''import sys,json,time
sys.stdin.readline()
print(json.dumps({'id':99,'status':'solved','value':{}}),flush=True)
time.sleep(60)
''')
    with pytest.raises(ValueError, match="sequence"):
        call(ik)
    assert ik._process.poll() is not None


@pytest.mark.parametrize("changed", [False, True])
def test_only_solver_misses_before_native_effects_are_recoverable(monkeypatch, changed):
    from PhysicalRSI_baselines.embodied_goodharts_law.persistent_libero_primitives import PersistentIKLiberoPrimitives
    from PhysicalRSI_baselines.embodied_goodharts_law.located_libero_primitives import LocatedLiberoPrimitives
    api = object.__new__(PersistentIKLiberoPrimitives)
    api._steps = 7
    api.get_robot_state = lambda: {"public": "proprioception"}

    def failed(self, pose):
        if changed:
            self._steps += 1
        raise IKNonConvergence("IK target did not converge")

    monkeypatch.setattr(LocatedLiberoPrimitives, "try_move_to_pose", failed)
    if changed:
        with pytest.raises(RuntimeError, match="after native effects"):
            api.try_move_to_pose([0] * 7)
    else:
        reply = api.try_move_to_pose([0] * 7)
        assert not reply["reached"] and not reply["native_effects"]
        assert reply["physics_steps"] == 7
