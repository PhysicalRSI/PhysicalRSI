"""Durable launch of the unmodified upstream evaluator, outside the policy process.

This is the native evaluation transport, not a System 1 controller. A websocket
endpoint's declared identity still needs to be attested by the policy service.
"""
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import signal
import subprocess

from PhysicalRSI_core.contracts import ReconciliationRequired
from PhysicalRSI_core.embodiment import System2Trial
from PhysicalRSI_core.infra.storage import atomic_json, digest, file_digest, identifier, locked, read_json
from .doctor import BEHAVIOR_REVISION


def runtime_environment(root, *, temporary_directory=None):
    """Keep caches on the runtime disk and use its optional graphics manifests."""
    root = Path(root).resolve()
    temporary = root / "tmp" if temporary_directory is None else Path(temporary_directory).resolve()
    if temporary_directory is not None and not temporary.is_dir():
        raise ValueError("Explicit temporary directory must already exist")
    env = dict(os.environ, TMPDIR=str(temporary), HF_HOME=str(root / "cache/huggingface"),
               OPENPI_DATA_HOME=str(root / "cache/openpi"),
               XDG_CACHE_HOME=str(root / "cache"), CUDA_CACHE_PATH=str(root / "cache/cuda"),
               __GL_SHADER_DISK_CACHE_PATH=str(root / "cache/gl-shaders"),
               OMNIGIBSON_APPDATA_PATH=str(root / "appdata"),
               OMNIGIBSON_DATA_PATH=str(root / "data"),
               OMNI_KIT_ACCEPT_EULA="YES", OMNI_KIT_ALLOW_ROOT="1", PYTHONUNBUFFERED="1")
    manifests = {"VK_ICD_FILENAMES": root / "config/nvidia_icd.json",
                 "__EGL_VENDOR_LIBRARY_FILENAMES": root / "config/10_nvidia.json"}
    present = [p.is_file() for p in manifests.values()]
    if any(present) and not all(present):
        raise ValueError("Provide both Vulkan and EGL runtime manifests")
    graphics = {}
    if all(present):
        for name, path in manifests.items():
            env[name] = str(path)
            graphics[name] = {"path": str(path), "sha256": file_digest(path)}
        xdg = root / "xdg-runtime"
        xdg.mkdir(mode=0o700, parents=True, exist_ok=True)
        env["XDG_RUNTIME_DIR"] = str(xdg)
    return env, graphics


def probe_policy(host, port, expected_revision, *, timeout=25):
    """Bind a cooperative local service's handshake to this candidate.

    This does not infer, reset a robot, or prove remote process isolation.
    """
    import msgpack
    import websockets.sync.client
    if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("A finite positive handshake timeout is required")
    # Budget connection, metadata receive and close separately. A silent peer
    # must not extend the supervising service's remaining startup allowance.
    stage_timeout = timeout / 3
    with websockets.sync.client.connect(f"ws://{host}:{port}", open_timeout=stage_timeout,
                                        close_timeout=stage_timeout, max_size=16 * 1024 * 1024) as connection:
        payload = connection.recv(timeout=stage_timeout)
        metadata = msgpack.unpackb(payload, raw=False)
    if (metadata.get("schema") != "physicalrsi.behavior-policy-service/v1"
            or metadata.get("candidate_sha256") != expected_revision
            or metadata.get("policy_identity_sha256") != digest(metadata.get("policy_identity"))):
        raise ValueError("Policy service handshake differs from the frozen candidate")
    return metadata


def make_command(runtime, output, *, task, mode, instance_index, policy, host, port, max_steps=None):
    identifier(task)
    if mode not in {"train", "public_test"}:
        raise ValueError("Hidden evaluation is not a development input")
    if type(instance_index) is not int or instance_index < 0 or (mode == "public_test" and instance_index >= 20):
        raise ValueError("Invalid instance index")
    if policy not in {"local", "websocket"}:
        raise ValueError("Unknown native policy transport")
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("Invalid policy port")
    if max_steps is not None and (type(max_steps) is not int or max_steps < 1):
        raise ValueError("Invalid step limit")
    argv = [str(Path(runtime) / "env/bin/python"), "-m", "omnigibson.eval.eval",
            "--task-name", task, "--mode", mode, "--instance-indices", str(instance_index),
            "--num-envs", "1", "--num-rollouts", "1", "--policy", policy,
            "--host", host, "--port", str(port), "--output-dir", str(output), "--write-video", "--headless"]
    if max_steps is not None:
        argv += ["--max-steps", str(max_steps)]
    return argv


def parse_result(path, *, task, instance_id):
    result = read_json(path)
    if (result.get("task"), result.get("instance_id"), result.get("rollout_id")) != (task, instance_id, 0):
        raise ValueError("Native result belongs to another episode")
    if type(result.get("success")) is not bool:
        raise ValueError("Native evaluator must report success explicitly")
    score = result["q_score"]["final"]
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Invalid native final Q-score")
    return result


def verify_source(source):
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True, timeout=30).strip()
    if revision != BEHAVIOR_REVISION:
        raise ValueError("Native evaluator revision differs from the pinned release")
    subprocess.run(["git", "-C", str(source), "diff", "--quiet", "HEAD", "--"], check=True, timeout=120)
    return revision


def read_receipt(folder):
    folder = Path(folder)
    receipt = read_json(folder / "receipt.json")
    if receipt["state"] != "completed":
        raise ReconciliationRequired(f"Inspect native evaluator process and outputs before retry: {folder}")
    for name, expected in receipt["evidence"].items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder.resolve()) or file_digest(path) != expected:
            raise ValueError("Native evaluation evidence changed")
    if digest(read_json(folder / "specification.json")) != receipt["specification_sha256"]:
        raise ValueError("Native evaluation specification changed")
    native_file = receipt["native_result_file"]
    if native_file not in receipt["evidence"]:
        raise ValueError("Native result is not bound to evidence")
    native = read_json(folder / native_file)
    if receipt["native_result"] != native or receipt["outcome"] != ("success" if native["success"] else "failure"):
        raise ValueError("Native result differs from receipt")
    specification = read_json(folder / "specification.json")
    if specification.get("schema") == "physicalrsi.behavior-native-group-case/v1":
        from .grouped import verify_case_binding
        if receipt.get("schema") != "physicalrsi.behavior-native-group-case-receipt/v1":
            raise ValueError("Grouped case receipt schema differs from its specification")
        verify_case_binding(folder, receipt, specification)
    return receipt


def specification(runtime, folder, *, task, mode="train", instance_index=0, policy="local",
        host="127.0.0.1", port=8000, max_steps=None, seconds=3600, policy_revision=None,
        system2: System2Trial | None = None, temporary_directory=None):
    """Construct a preregistrable launch identity without contacting a policy.

    This reads implementation and optional graphics manifests but does not load
    a model or simulator, reserve trials, or certify admission.
    """
    if isinstance(seconds, bool) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("A finite positive process deadline is required")
    if policy == "websocket" and not policy_revision:
        raise ValueError("Declare the served policy revision")
    if system2 is not None:
        if not isinstance(system2, System2Trial) or system2.candidate_sha256 != policy_revision:
            raise ValueError("System 2 attribution must match the served candidate")
        if policy != "websocket" or (system2.split == "test") != (mode == "public_test"):
            raise ValueError("Native mode differs from the System 2 split")
    root, folder = Path(runtime).resolve(), Path(folder).resolve()
    argv = make_command(root, folder / "native", task=task, mode=mode, instance_index=instance_index,
                        policy=policy, host=host, port=port, max_steps=max_steps)
    env, graphics = runtime_environment(root, temporary_directory=temporary_directory)
    case = dict(task=task, mode=mode, instance_index=instance_index,
                instance_id=instance_index if mode == "train" else 301 + instance_index, rollout_id=0)
    return dict(schema="physicalrsi.behavior-native-run/v1", argv=argv, source_revision=BEHAVIOR_REVISION,
                         seconds=seconds, policy_revision=policy_revision,
                         purpose="smoke_only" if policy == "local" else "policy_evaluation",
                         implementation_sha256=file_digest(Path(__file__)), graphics_manifests=graphics,
                         temporary_directory=env["TMPDIR"],
                         case=case, system2=asdict(system2) if system2 else None)


def run(runtime, folder, *, task, mode="train", instance_index=0, policy="local",
        host="127.0.0.1", port=8000, max_steps=None, seconds=3600, policy_revision=None,
        system2: System2Trial | None = None, expected_specification=None, temporary_directory=None):
    planned = specification(runtime, folder, task=task, mode=mode, instance_index=instance_index,
                            policy=policy, host=host, port=port, max_steps=max_steps, seconds=seconds,
                            policy_revision=policy_revision, system2=system2,
                            temporary_directory=temporary_directory)
    if expected_specification is not None and planned != expected_specification:
        raise ValueError("Native launch differs from its preregistered specification")
    root, folder = Path(runtime).resolve(), Path(folder).resolve()
    source = root / "sources/BEHAVIOR-1K"
    env, graphics = runtime_environment(root, temporary_directory=temporary_directory)
    if graphics != planned["graphics_manifests"]:
        raise ValueError("Graphics manifests changed during native launch preparation")
    if env["TMPDIR"] != planned["temporary_directory"]:
        raise ValueError("Temporary directory changed during native launch preparation")
    argv = planned["argv"]
    with locked(folder / ".lock"):
        if (folder / "receipt.json").exists():
            if read_json(folder / "specification.json") != planned:
                raise ValueError("Run directory already belongs to a different experiment")
            return read_receipt(folder)
        verify_source(source)
        diagnostics = None
        if policy == "websocket":
            metadata = probe_policy(host, port, policy_revision)
            if metadata.get("task") != task:
                raise ValueError("Policy service is serving a different task")
            diagnostics = metadata.get("policy_identity", {}).get("diagnostics")
            if diagnostics is not None:
                if (not isinstance(diagnostics, dict) or system2 is None
                        or system2.split != "development" or mode != "train"
                        or diagnostics.get("split") != "development"
                        or diagnostics.get("candidate_sha256") != policy_revision
                        or diagnostics.get("task") != task
                        or diagnostics.get("case_id") != digest([task, {"instance_id": instance_index}])):
                    raise ValueError("Policy diagnostics do not match this development trial")
            atomic_json(folder / "policy_service.json", metadata)
        atomic_json(folder / "specification.json", planned)
        receipt = dict(schema="physicalrsi.behavior-native-receipt/v1", state="started",
                       specification_sha256=digest(planned), qualification=None)
        atomic_json(folder / "receipt.json", receipt)
        process = None
        try:
            with (folder / "evaluator.log").open("w") as log:
                process = subprocess.Popen(argv, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                atomic_json(folder / "receipt.json", dict(receipt, pid=process.pid))
                returncode = process.wait(timeout=seconds)
            if returncode != 0:
                raise RuntimeError(f"Native evaluator exited {returncode}; inspect evaluator.log")
            verify_source(source)
            instance_id = instance_index if mode == "train" else 301 + instance_index
            result_file = folder / "native/json" / f"{task}_{instance_id}_0.json"
            result = parse_result(result_file, task=task, instance_id=instance_id)
            if diagnostics is not None:
                from .diagnostics import seal_trace
                seal_trace(diagnostics["path"], folder / "development-trace.jsonl",
                           expected_identity=diagnostics, expected_steps=result.get("steps"))
            evidence = {str(p.relative_to(folder)): file_digest(p) for p in folder.rglob("*")
                        if p.is_file() and p.name not in {"receipt.json", ".lock"}}
            receipt.update(state="completed", native_result=result,
                           native_result_file=str(result_file.relative_to(folder)),
                           outcome="success" if result["success"] else "failure", evidence=evidence,
                           purpose=planned["purpose"])
            atomic_json(folder / "receipt.json", receipt)
            return read_receipt(folder)
        except BaseException as error:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
            atomic_json(folder / "receipt.json", dict(receipt, state="unresolved", error_type=type(error).__name__,
                                                       reason=str(error), pid=process.pid if process else None))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime")
    parser.add_argument("output")
    parser.add_argument("--task", required=True)
    parser.add_argument("--mode", default="train", choices=["train", "public_test"])
    parser.add_argument("--instance-index", type=int, default=0)
    parser.add_argument("--policy", default="local", choices=["local", "websocket"])
    parser.add_argument("--policy-revision")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--seconds", type=float, default=3600)
    parser.add_argument("--temporary-directory", help="Existing local scratch directory; bound to the run specification")
    args = vars(parser.parse_args())
    args["folder"] = args.pop("output")
    print(json.dumps(run(**args), indent=2))


if __name__ == "__main__":
    main()
