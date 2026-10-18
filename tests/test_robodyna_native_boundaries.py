"""CPU tests for native-worker cleanup and the scoring budget boundary."""
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import os
import time

import pytest

ROOT = Path(__file__).parents[1] / 'PhysicalRSI_baselines/robodyna'


def load(name):
    spec = importlib.util.spec_from_file_location('staged_' + name, ROOT / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


adapter = load('adapter')
worker = load('worker')


class Process:
    pid = 1234567

    def __init__(self):
        self.stdin = io.StringIO()
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout):
        self.returncode = 0
        return 0


def test_metadata_failure_closes_child_and_fds_even_if_gpu_query_fails(tmp_path, monkeypatch):
    process = Process()
    suite = SimpleNamespace(output=tmp_path, runtime_environment={}, python='unused', root=tmp_path)
    env = adapter.NativeEnvironment(suite, 'catch_ramp_ball')
    monkeypatch.setattr(adapter.subprocess, 'Popen', lambda *a, **k: process)
    def failed_metadata():
        raise RuntimeError('metadata unavailable')
    env.identity = failed_metadata
    commands = []
    env._rpc = lambda command, **kw: commands.append(command)
    def gpu_failure(*args, **kwargs):
        assert kwargs['timeout'] == 5
        raise RuntimeError('GPU query unavailable')
    monkeypatch.setattr(adapter.subprocess, 'check_output', gpu_failure)
    with pytest.raises(RuntimeError, match='metadata unavailable'):
        env.reset(dict(seed=1, mode='frozen'), None)
    assert process.returncode == 0
    assert process.stdin.closed and env.log.closed
    assert env.process is None
    assert commands == ['close']
    with pytest.raises(OSError):
        os.fstat(env.reply_fd)
    assert json.loads((env.folder/'gpu-final-error.json').read_text())['error_type'] == 'RuntimeError'


class Vector(list):
    def tolist(self):
        return list(self)


def make_worker(tmp_path, monkeypatch, success_step):
    # Construct without chdir/file ownership: only observer/observation contracts.
    native = worker.NativeWorker.__new__(worker.NativeWorker)
    native.sim_steps = 0
    native.max_sim_steps = 2
    native.success_snapshot = None
    native.budget_score_snapshot = None
    native.frames = 0
    native.task = 'catch_ramp_ball'
    native.rollout_started = time.monotonic()
    native.execution_seconds = 0.
    native.executor = SimpleNamespace(phase='place')
    native._sample = lambda: None
    native._record_frame = lambda: None
    native._event = lambda *args: None
    native._phase = lambda: 'flight'
    native.samples = []
    actor = SimpleNamespace(get_pose=lambda: SimpleNamespace(p=Vector([0, 0, 1])))
    scene = SimpleNamespace(step=lambda: None, get_timestep=lambda: 0.004)
    native.env = SimpleNamespace(scene=scene, ball=actor, cup=actor, table_top=.74,
        _surface_point=lambda *args: Vector([0, 0, 1]), _y_top=1, _y_edge=0,
        check_success=lambda: native.sim_steps >= success_step,
        get_score=lambda: float(native.sim_steps >= success_step),
        get_score_detail=lambda: {'score': float(native.sim_steps >= success_step)},
        get_fail_reason=lambda: '' if native.sim_steps >= success_step else 'not caught',
        async_inference_stats=lambda: {}, _episode_past_deadline=lambda: False)
    native._install_observer()
    return native


def test_motion_finishing_after_budget_cannot_turn_late_catch_into_success(tmp_path, monkeypatch):
    native = make_worker(tmp_path, monkeypatch, success_step=3)
    for _ in range(4):
        native.env.scene.step()
    result = native.observe()
    assert native.sim_steps == 4  # Physics was not pretended to hard-stop at 2.
    assert native.success_snapshot is None
    assert result['terminated']
    assert result['native']['success'] is False
    assert result['native']['progress'] == 0.
    assert result['native']['score_detail'] == {'score': 0.}
    assert result['native']['failure_reason'] == 'not caught'
    assert result['native']['stop_reason'] == 'adapter_simulation_budget'


def test_native_success_exactly_at_budget_is_preserved(tmp_path, monkeypatch):
    native = make_worker(tmp_path, monkeypatch, success_step=2)
    for _ in range(3):
        native.env.scene.step()
    result = native.observe()
    assert native.success_snapshot['sim_steps'] == 2
    assert result['native']['success'] is True
    assert result['native']['progress'] == 1.
