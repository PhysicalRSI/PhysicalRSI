"""Owned, deadline-bounded CPU IK process with model/JIT reuse across calls."""
import json
import math
import os
from pathlib import Path
import selectors
import signal
import subprocess
import threading
import time

from PhysicalRSI_core.infra.storage import atomic_json, file_digest
from .aspire_ik import AspireHandIK, gripper_seed_fraction


class IKNonConvergence(RuntimeError):
    """The pure solver returned normally without a sufficiently accurate pose."""


class PersistentAspireHandIK(AspireHandIK):
    def __init__(self, *, worker_directory, **kwargs):
        super().__init__(**kwargs)
        self.directory = Path(worker_directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self._process = None
        self._log = None
        self._closed = False
        self._close_result = None
        self._index = 0
        self._lock = threading.Lock()

    def identity(self):
        return {**super().identity(), "persistent_client_sha256": file_digest(Path(__file__)),
                "persistent_worker_sha256": file_digest(Path(__file__).with_name("aspire_ik_worker.py")),
                "execution": "owned CPU process; cached model and compiled solver"}

    def _command(self):
        return [self.interpreter, "-m", "PhysicalRSI_baselines.embodied_goodharts_law.aspire_ik_worker"]

    def _start(self):
        env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONDONTWRITEBYTECODE="1")
        env["PHYSICALRSI_IK_OWNER_PID"] = str(os.getpid())
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2]) + os.pathsep + env.get("PYTHONPATH", "")
        self._log = (self.directory / "worker.log").open("xb")
        self._process = subprocess.Popen(self._command(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self._log, env=env, start_new_session=True, bufsize=0)
        atomic_json(self.directory / "process.json", {"pid": self._process.pid,
                    "state": "running", "identity": self.identity()})

    def _stop(self, *, force=False):
        if self._close_result is not None:
            return dict(self._close_result)
        self._closed = True
        process = self._process
        if process is None:
            if self._log is not None:
                self._log.close()
            self._close_result = {"started": False, "stopped": True}
            return dict(self._close_result)
        if process.poll() is None:
            if force:
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.stdin.close()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
                force = True
        process.stdin.close()
        process.stdout.close()
        self._log.close()
        result = {"started": True, "stopped": True, "pid": process.pid,
                  "returncode": process.returncode, "forced": force, "requests": self._index}
        atomic_json(self.directory / "process.json", {**result, "state": "stopped", "identity": self.identity()})
        self._close_result = result
        return dict(result)

    def close(self):
        if not self._lock.acquire(timeout=5):
            raise RuntimeError("IK call has not relinquished worker ownership")
        try:
            return self._stop()
        finally:
            self._lock.release()

    def __call__(self, *, pose, joints, gripper_fraction, deadline):
        remaining = min(self.timeout_seconds, deadline - time.monotonic())
        if not math.isfinite(remaining) or remaining <= 0:
            raise TimeoutError("IK deadline reached")
        end = time.monotonic() + remaining
        if not self._lock.acquire(timeout=remaining):
            raise TimeoutError("IK ownership deadline reached")
        try:
            if self._closed:
                raise RuntimeError("IK worker is closed")
            end = min(deadline, end)
            fraction = gripper_seed_fraction(gripper_fraction)
            request = {"id": self._index, "urdf": self.urdf, "pose": list(map(float, pose)),
                       "joints": list(map(float, joints)), "gripper_fraction": fraction,
                       "translation": list(map(float, self.translation)),
                       "quaternion": list(map(float, self.quaternion))}
            payload = (json.dumps(request, allow_nan=False) + "\n").encode()
            if len(payload) > 4096:
                raise ValueError("IK request exceeds allowance")
            if self._process is None:
                self._start()
            if time.monotonic() >= end:
                raise TimeoutError("IK deadline reached")
            # One outstanding bounded request; no queued writes may fill stdin.
            if self._process.stdin.write(payload) != len(payload):
                raise RuntimeError("Incomplete IK request write")
            buffer = bytearray()
            with selectors.DefaultSelector() as selector:
                selector.register(self._process.stdout, selectors.EVENT_READ)
                while not buffer.endswith(b"\n"):
                    remaining = end - time.monotonic()
                    if remaining <= 0 or not selector.select(remaining):
                        raise TimeoutError("IK deadline reached")
                    data = os.read(self._process.stdout.fileno(), 65537 - len(buffer))
                    if not data:
                        raise RuntimeError("IK worker exited without a response")
                    buffer.extend(data)
                    if len(buffer) > 65536:
                        raise ValueError("IK response exceeds allowance")
            if time.monotonic() >= end:
                raise TimeoutError("IK deadline reached")
            response = json.loads(buffer)
            if response.get("id") != self._index:
                raise ValueError("IK response sequence mismatch")
            self._index += 1
            if response.get("status") == "nonconverged":
                raise IKNonConvergence("IK target did not converge")
            if response.get("status") != "solved" or not isinstance(response.get("value"), dict):
                raise ValueError("Invalid IK response")
            value = response["value"]
            value["gripper_seed"] = {"measured_fraction": float(gripper_fraction),
                                     "model_fraction": fraction,
                                     "clipped": fraction != float(gripper_fraction)}
            return value
        except IKNonConvergence:
            raise
        except BaseException:
            self._stop(force=True)
            raise
        finally:
            self._lock.release()
