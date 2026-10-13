"""Launch-boundary checks without importing or emulating Isaac physics."""
import subprocess
import sys
import time
import venv

from PhysicalRSI_core.contracts import Context
from PhysicalRSI_demos.auto_engineering import trials


def test_worker_preserves_virtualenv_and_uses_owned_runtime_paths(tmp_path, monkeypatch):
    environment = tmp_path / "venv"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
    executable = environment / "bin/python"
    assert executable.is_symlink()
    actual_popen = subprocess.Popen
    launches = []
    def inspect_launcher(command, **kwargs):
        launches.append((command, kwargs["env"]))
        directory = command[command.index("--bridge") + 1]
        script = ("import json,pathlib,sys,time; "
                  "pathlib.Path(sys.argv[1], 'ready.json').write_text(json.dumps({'ready':True, 'observation':sys.prefix})); "
                  "time.sleep(30)")
        return actual_popen([command[0], "-c", script, directory], **kwargs)
    monkeypatch.setattr(trials.subprocess, "Popen", inspect_launcher)
    worker = trials.NativeWorker(executable, tmp_path / "native")
    try:
        result = worker.reset(dict(seed=17), Context("launch", deadline=time.monotonic()+10))
        assert result["observation"] == str(environment)
        command, env = launches[0]
        assert command[0] == str(executable)
        assert command[command.index("--runtime-root")+1] == str(tmp_path / "native/runtime/kit")
        assert env["TMPDIR"] == str(tmp_path / "native/runtime/tmp")
        assert env["CUDA_CACHE_PATH"] == str(tmp_path / "native/runtime/cuda-cache")
        assert env["__GL_SHADER_DISK_CACHE_PATH"] == str(tmp_path / "native/runtime/gl-cache")
    finally:
        worker.close()
    assert worker.process is None
