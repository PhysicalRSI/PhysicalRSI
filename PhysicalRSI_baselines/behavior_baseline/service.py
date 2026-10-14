"""Bounded local policy-service lifecycle for native experiments.

The server runs reviewed code in its own process group. This is lifecycle
supervision, not a security sandbox or proof of benchmark admission.
"""
from contextlib import contextmanager
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid

from PhysicalRSI_core.infra.storage import atomic_json, locked, read_json
from PhysicalRSI_core.self_harness.artifacts import verify_harness
from .native import probe_policy, runtime_environment
from .diagnostics import validate_binding
from .openpi_backend import supported_tasks


def command(python, candidate_path, *, port, episode_seconds, development_binding=None, development_trace=None, task=None):
    argv = [str(python), "-m", "PhysicalRSI_baselines.behavior_baseline.serve", str(candidate_path),
            "--host", "127.0.0.1", "--port", str(port), "--episode-seconds", str(episode_seconds)]
    if task is not None:
        argv += ["--task", task]
    if development_binding is not None:
        argv += ["--development-binding", str(development_binding), "--development-trace", str(development_trace)]
    return argv


def stop(process):
    """Reap our own process and stop any model workers in its process group."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)
    # A parent can exit before its worker children do.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


@contextmanager
def policy_service(runtime, candidate_path, folder, *, python, task, port=8000,
                   startup_seconds=1800, episode_seconds=3600, development_binding=None):
    """Start one frozen OpenPI candidate; yield its verified local endpoint.

    Use a new evidence folder per launch. An interrupted folder must be inspected
    rather than reused, even if the old service appears to have disappeared.
    """
    for seconds in (startup_seconds, episode_seconds):
        if isinstance(seconds, bool) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Service deadlines must be finite and positive")
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("Invalid service port")
    candidate_path = Path(candidate_path).resolve()
    candidate = read_json(candidate_path)
    revision = verify_harness(candidate)
    backend = read_json(Path(candidate["root"]) / "backend.json")
    if task not in supported_tasks(backend):
        raise ValueError("Service must match the candidate's OpenPI task binding")
    folder = Path(folder).resolve()
    diagnostic_case = (validate_binding(development_binding, candidate_sha256=revision, task=task)
                       if development_binding is not None else None)
    env, graphics = runtime_environment(runtime)
    repository = Path(__file__).resolve().parents[2]
    env["PYTHONPATH"] = str(repository) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PHYSICALRSI_SERVICE_NONCE"] = uuid.uuid4().hex
    # Leave GPU memory for the native renderer in the colocated configuration.
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    diagnostic_args = (dict(development_binding=folder / "development-binding.json",
                            development_trace=folder / "development.jsonl")
                       if diagnostic_case is not None else {})
    argv = command(python, candidate_path, port=port, episode_seconds=episode_seconds, task=task, **diagnostic_args)
    with locked(folder / ".lock"):
        state_path = folder / "service.json"
        if state_path.exists():
            raise FileExistsError("Inspect the previous service launch; use a new evidence folder")
        if diagnostic_case is not None:
            atomic_json(folder / "development-binding.json", development_binding)
        state = dict(state="starting", candidate_sha256=revision, task=task, argv=argv,
                     startup_seconds=startup_seconds, episode_seconds=episode_seconds,
                     launch_nonce=env["PHYSICALRSI_SERVICE_NONCE"], graphics_manifests=graphics,
                     qualification=None)
        atomic_json(state_path, state)
        process = None
        try:
            with (folder / "server.log").open("x") as log:
                process = subprocess.Popen(argv, cwd=repository, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                state["pid"] = process.pid
                atomic_json(state_path, state)
                deadline = time.monotonic() + startup_seconds
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f"Policy server exited {process.returncode}; inspect server.log")
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Policy service startup deadline exceeded")
                    try:
                        metadata = probe_policy("127.0.0.1", port, revision,
                                                timeout=min(25, max(.001, deadline-time.monotonic())))
                    except (OSError, TimeoutError):
                        time.sleep(min(1, max(0, deadline-time.monotonic())))
                        continue
                    if (metadata.get("service_pid") != process.pid
                            or metadata.get("launch_nonce") != state["launch_nonce"]
                            or metadata.get("task") != task):
                        raise ValueError("Endpoint is not this launched candidate service")
                    diagnostics = metadata.get("policy_identity", {}).get("diagnostics")
                    if diagnostic_case is not None:
                        if diagnostics != dict(schema="physicalrsi.behavior-development-trace/v1",
                                               split="development", candidate_sha256=revision,
                                               task=task, case_id=diagnostic_case,
                                               path=str(folder / "development.jsonl")):
                            raise ValueError("Service diagnostic attribution differs from launch")
                    elif diagnostics is not None:
                        raise ValueError("Unexpected diagnostic recorder in policy service")
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Policy service startup deadline exceeded")
                    if process.poll() is not None:
                        raise RuntimeError("Policy server exited during handshake")
                    break
                if verify_harness(candidate) != revision:
                    raise ValueError("Candidate changed during service startup")
                atomic_json(folder / "handshake.json", metadata)
                state["state"] = "ready"
                atomic_json(state_path, state)
                yield dict(host="127.0.0.1", port=port, policy_revision=revision)
        except BaseException as error:
            state.update(error_type=type(error).__name__, reason=str(error))
            raise
        finally:
            if process is not None:
                stop(process)
                state["returncode"] = process.returncode
            state["state"] = "stopped"
            atomic_json(state_path, state)
