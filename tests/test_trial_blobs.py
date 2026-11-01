import hashlib
import time

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("PIL")

from PhysicalRSI_core.experiments import Budget, ExperimentRuntime
from PhysicalRSI_core.infra.blobs import BlobStore


class CameraEnvironment:
    def __init__(self, steps=100):
        self.steps = steps

    def identity(self):
        return {"name": "fake-camera", "revision": "1"}

    def reset(self, case, context):
        self.step_count = 0
        self.rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        self.depth = np.ones((480, 640), dtype=np.float32)
        self.rgb_ref = context.blobs.put_png(self.rgb)
        self.depth_ref = context.blobs.put_array(self.depth)
        return {"ready": True, "observation": self.observation(), "terminated": False}

    def observation(self):
        return {"rgb": self.rgb_ref, "depth": self.depth_ref}

    def step(self, action, context):
        self.step_count += 1
        return {"observation": self.observation(), "terminated": self.step_count >= self.steps}


class CameraPolicy:
    def identity(self):
        return {"name": "camera-reader", "revision": "1"}

    def act(self, observation, context):
        rgb = context.blobs.get_array(observation["rgb"])
        depth = context.blobs.get_array(observation["depth"])
        assert rgb.shape == (480, 640, 3) and depth.shape == (480, 640)
        return int(depth[0, 0] > 0)


class CameraVerifier:
    def identity(self):
        return {"name": "camera-verifier", "revision": "1"}

    def verify(self, case, trace, context):
        assert context.blobs.get_array(trace[-1]["observation"]["depth"]).dtype == np.dtype("float32")
        return {"outcome": "success", "reason": "camera trace complete", "measurements": {"steps": len(trace) - 1}}


def run_camera(tmp_path, *, steps=100):
    runtime = ExperimentRuntime(tmp_path / "trials")
    started = time.perf_counter()
    receipt = runtime.run(
        "camera",
        task="camera",
        case={},
        scope="software",
        environment=CameraEnvironment(steps),
        policy=CameraPolicy(),
        verifier=CameraVerifier(),
        budget=Budget(steps, 30),
    )
    elapsed = time.perf_counter() - started
    print(f"blob trial overhead: {elapsed / steps * 1000:.3f} ms/step")
    return runtime, receipt, elapsed


def test_camera_trial_is_compact_and_covers_blobs(tmp_path):
    runtime, receipt, elapsed = run_camera(tmp_path)
    assert elapsed / 100 < 0.020
    folder = runtime.root / "camera"
    assert (folder / "blobs").is_dir()
    assert (folder / "trajectory.json").stat().st_size < 200 * 1024
    blob_entries = {name: sha for name, sha in receipt["evidence"].items() if name.startswith("blobs/")}
    assert blob_entries
    assert all(len(sha) == 64 for sha in blob_entries.values())
    assert runtime.read_evidence("camera")["receipt"] == receipt


def test_tampering_with_referenced_blob_is_detected(tmp_path):
    runtime, receipt, _ = run_camera(tmp_path)
    name = next(name for name in receipt["evidence"] if name.startswith("blobs/"))
    path = runtime.root / "camera" / name
    path.write_bytes(b"tampered" * ((path.stat().st_size // 8) + 1))
    with pytest.raises(ValueError, match="evidence changed|digest|size"):
        runtime.read_evidence("camera")


def test_missing_blob_during_step_fails_trial(tmp_path):
    class Missing(CameraEnvironment):
        def step(self, action, context):
            missing = hashlib.sha256(b"missing").hexdigest()
            ref = {"$blob": missing, "media_type": "application/x-npy", "shape": [1],
                   "dtype": "uint8", "bytes": 7}
            return {"observation": {"frame": ref}, "terminated": True}

    runtime = ExperimentRuntime(tmp_path / "trials")
    with pytest.raises(FileNotFoundError):
        runtime.run("missing", task="camera", case={}, scope="software", environment=Missing(1),
                     policy=CameraPolicy(), verifier=CameraVerifier(), budget=Budget(1, 30))


def test_identical_arrays_are_deduplicated(tmp_path):
    store = BlobStore(tmp_path / "blobs")
    array = np.arange(32, dtype=np.uint8)
    first = store.put_array(array)
    second = store.put_array(array.copy())
    assert first == second
    assert len(store.files()) == 1
