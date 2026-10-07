import os
from pathlib import Path
import signal
import subprocess
import sys
from threading import Timer
from time import monotonic, sleep

import pytest

from PhysicalRSI_core.contracts import Cancelled, Context, ReconciliationRequired
from PhysicalRSI_core.embodiment import System1
from PhysicalRSI_core.experiments import Budget, ExperimentRuntime
from PhysicalRSI_core.infra.isolated_policy import IsolatedPolicy
from PhysicalRSI_core.infra.isolated_program import PythonIsolation
from PhysicalRSI_core.infra.isolated_proposal import IsolatedProposal
from PhysicalRSI_core.infra.storage import atomic_json, digest, file_digest, read_json
from PhysicalRSI_demos.experiment import ACTION, OBSERVATION, CounterEnvironment, CounterVerifier
from test_embodied_experiments import LeasedCounter
from test_structured_proposals import setup


STATEFUL = '''
def begin_episode(task, case, observation):
    return 0

def act(observation, state):
    return dict(action=state + 1, state=state + 1)
'''


def policy(root, runtime, source=STATEFUL, **kwargs):
    return IsolatedPolicy(source, specification=System1("isolated-counter", digest(source), OBSERVATION, ACTION),
                          isolation=PythonIsolation(runtime), output=root, **kwargs)


def context(program, episode="one", seconds=15):
    return Context(episode, deadline=monotonic() + seconds, harness_revision=program.describe().revision)


def assert_stopped(root):
    records = list(Path(root).rglob("processes/isolated/process.json"))
    assert records
    for path in records:
        record = read_json(path)
        assert record["stopped"] is True and record["returncode"] is not None
        assert not Path("/proc", str(record["pid"])).exists()


def test_isolated_state_persists_only_within_episode_and_receipt_requires_stopped_worker(tmp_path, isolation_runtime):
    program = policy(tmp_path / "workers", isolation_runtime)
    runtime = ExperimentRuntime(tmp_path / "trials")
    options = dict(task="counter", case=dict(initial=0, target=3), scope="software",
                   environment=CounterEnvironment(), policy=program, verifier=CounterVerifier(), budget=Budget(3, 15))
    for episode in ("first", "second"):
        receipt = runtime.run(episode, **options)
        assert receipt["outcome"] == "success" and receipt["steps"] == 2 and receipt["qualification"] is None
        assert "policy.json" in receipt["evidence"]
        cleanup = read_json(runtime.root / episode / "policy.json")
        assert cleanup["stopped"] and cleanup["worker_error"] is None and cleanup["actions"] == 2
        assert cleanup["episode"] == episode
        trace = read_json(runtime.root / episode / "trajectory.json")
        assert [row["action"] for row in trace[1:]] == [1, 2]
    before = {str(p): file_digest(p) for p in (tmp_path / "workers").rglob("process.json")}
    assert runtime.run("second", **options) == receipt
    assert before == {str(p): file_digest(p) for p in (tmp_path / "workers").rglob("process.json")}
    assert_stopped(tmp_path)
    atomic_json(runtime.root / "second/policy.json", dict(stopped=False))
    with pytest.raises(ValueError, match="evidence changed"):
        runtime.read("second")


def test_old_cleanup_cannot_close_a_different_or_replayed_episode(tmp_path, isolation_runtime):
    program = policy(tmp_path / "workers", isolation_runtime)
    first, other = context(program, "first"), context(program, "other")
    program.begin_episode("counter", {}, {}, first)
    cleanup = program.end_episode(first)
    with pytest.raises(ValueError, match="different isolated policy episode"):
        program.end_episode(other)
    (tmp_path / "workers/other").mkdir()
    with pytest.raises(ReconciliationRequired, match="cannot be replayed"):
        program.begin_episode("counter", {}, {}, other)
    with pytest.raises(ValueError, match="different isolated policy episode"):
        program.end_episode(other)
    assert program.end_episode(first) == cleanup
    assert not (tmp_path / "workers/other/cleanup.json").exists()
    assert_stopped(tmp_path)


def test_child_cannot_read_host_credentials_devices_or_create_network_processes(tmp_path, isolation_runtime, monkeypatch):
    secret = tmp_path / "controller-private-key"
    secret.write_text("fixture-secret-never-for-child")
    monkeypatch.setenv("PHYSICALRSI_TEST_CONTROLLER_SECRET", "ambient-secret-never-for-child")
    inherited = os.open(secret, os.O_RDONLY)
    os.set_inheritable(inherited, True)
    checks = '''
import os, socket, sys
def begin_episode(task, case, observation):
    return case
def act(observation, state):
    result = dict(uid=os.getuid(), environment=sorted(os.environ))
    attempts = {
        'credential': lambda: open(state['secret']).read(),
        'parent_root': lambda: open('/proc/%d/root' % os.getppid() + state['secret']).read(),
        'device': lambda: open('/dev/null', 'rb'),
        'runtime_write': lambda: open(sys.executable, 'wb'),
        'inherited_fd': lambda: os.read(state['fd'], 1),
        'network': lambda: socket.socket(),
        'unix_socket': lambda: socket.socket(socket.AF_UNIX),
        'process': lambda: os.fork(),
        'signal': lambda: os.kill(os.getppid(), 0),
        'privilege': lambda: os.setuid(0),
    }
    for name, call in attempts.items():
        try:
            call()
            result[name] = 'unexpected-access'
        except OSError as error:
            result[name] = error.errno
    return dict(action=result, state=state)
'''
    program = policy(tmp_path / "workers", isolation_runtime, checks)
    ctx = context(program)
    try:
        program.begin_episode("security-fixture", dict(secret=str(secret), fd=inherited), {}, ctx)
        result = program.act({}, ctx)
        assert result.pop("uid") == 65534
        assert "PHYSICALRSI_TEST_CONTROLLER_SECRET" not in result.pop("environment")
        assert all(type(value) is int and value in {1, 2, 9, 13} for value in result.values())
        assert secret.read_text() == "fixture-secret-never-for-child"
    finally:
        program.end_episode(ctx)
        os.close(inherited)
    assert_stopped(tmp_path)


@pytest.mark.parametrize("mode", ["hang", "exit", "oversize", "forged_call", "duplicate", "cancel"])
def test_bad_or_stalled_policy_returns_no_action_and_is_reaped(tmp_path, isolation_runtime, mode):
    body = {
        "hang": "while True: pass",
        "exit": "__import__('os')._exit(3)",
        "oversize": "return dict(action='x' * (2 * 1024 * 1024), state=None)",
        "forged_call": "open('/output/result.json', 'w').write('{\"kind\":\"call\",\"id\":2,\"method\":\"control.execute\",\"args\":[],\"kwargs\":{}}')\n    while True: pass",
        "duplicate": "open('/output/result.json', 'w').write('{\"kind\":\"result\",\"value\":1,\"value\":2}')\n    while True: pass",
        "cancel": "while True: pass",
    }[mode]
    source = "def begin_episode(task, case, observation): return None\ndef act(observation, state):\n    " + body
    program = policy(tmp_path / "workers", isolation_runtime, source, call_seconds=2)
    ctx = context(program)
    program.begin_episode("counter", {}, {}, ctx)
    timer = Timer(.05, ctx.cancelled.set) if mode == "cancel" else None
    started = monotonic()
    try:
        if timer:
            timer.start()
        with pytest.raises((TimeoutError, RuntimeError, Cancelled)):
            program.act({}, ctx)
    finally:
        if timer:
            timer.cancel()
            timer.join()
        cleanup = program.end_episode(ctx)
    assert monotonic() - started < 5
    assert cleanup["stopped"] and cleanup["actions"] == 0
    assert_stopped(tmp_path)


def test_experiment_failure_reaps_worker_and_keeps_device_claim(tmp_path, isolation_runtime):
    from PhysicalRSI_core.infra.devices import DeviceRegistry
    class BrokenEnvironment(LeasedCounter):
        def step(self, action, context):
            super().step(action, context)
            raise RuntimeError("lost sensor after action")
    program = policy(tmp_path / "workers", isolation_runtime)
    runtime = ExperimentRuntime(tmp_path / "trials", device_registry=DeviceRegistry(tmp_path / "devices"))
    options = dict(task="counter", case=dict(initial=0, target=3), scope="software",
                   environment=BrokenEnvironment(), policy=program, verifier=CounterVerifier(), budget=Budget(3, 15))
    with pytest.raises(RuntimeError, match="lost sensor"):
        runtime.run("broken", **options)
    receipt = read_json(runtime.root / "broken/receipt.json")
    assert receipt["state"] == "needs_reconciliation" and runtime.device_registry.occupied()
    assert read_json(runtime.root / "broken/policy.json")["stopped"] is True
    with pytest.raises(ReconciliationRequired):
        runtime.run("broken", **options)
    assert_stopped(tmp_path)


def test_isolation_manifest_change_fails_before_starting_process(tmp_path, isolation_runtime):
    manifest = tmp_path / "runtime/manifest.json"
    atomic_json(manifest, read_json(isolation_runtime / "manifest.json"))
    profile = PythonIsolation(manifest.parent)
    atomic_json(manifest, dict(read_json(manifest), executable="/other"))
    with pytest.raises(ValueError, match="configuration changed"):
        profile.run("raise AssertionError('must not execute')", {}, output=tmp_path / "job", deadline=monotonic()+2)
    assert not (tmp_path / "job").exists()


def test_parent_process_death_kills_its_isolated_worker(tmp_path, isolation_runtime):
    script = '''
from pathlib import Path
import sys
from time import sleep
from PhysicalRSI_core.infra.isolated_program import PythonIsolation
from PhysicalRSI_core.infra.isolated_policy import IsolatedPolicy
from PhysicalRSI_core.contracts import Context, Contract
from PhysicalRSI_core.embodiment import System1
from time import monotonic
root=Path(sys.argv[2])
source="def begin_episode(task,case,observation): return None\\ndef act(observation,state): return dict(action=1,state=state)"
program=IsolatedPolicy(source,specification=System1('parent-death','1',Contract('in'),Contract('out')),
    isolation=PythonIsolation(sys.argv[1]),output=root/'worker')
program.begin_episode('fixture',{}, {}, Context('episode',deadline=monotonic()+30,harness_revision='1'))
(root/'ready').touch()
sleep(30)
'''
    process = subprocess.Popen([sys.executable, "-c", script, str(isolation_runtime), str(tmp_path)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    child = None

    def alive(pid):
        try:
            return Path("/proc", str(pid), "stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
        except FileNotFoundError:
            return False

    try:
        deadline = monotonic() + 10
        while not (tmp_path / "ready").exists():
            assert process.poll() is None, process.stderr.read().decode()
            assert monotonic() < deadline, "Isolated parent did not become ready"
            sleep(.02)
        record = read_json(tmp_path / "worker/episode/processes/isolated/process.json")
        child = record["pid"]
        assert alive(child)
        process.kill()
        process.wait(timeout=3)
        deadline = monotonic() + 3
        while alive(child) and monotonic() < deadline:
            sleep(.02)
        assert not alive(child), "Child survived its owning controller's death"
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=3)
        process.stderr.close()
        if child is not None and alive(child):
            os.kill(child, signal.SIGKILL)


PROPOSAL = '''
import json
def generate(request, limits):
    candidates = []
    value = request['editable']['control.json']
    if value['value']['increment'] != 1:
        candidates = [dict(rationale='Test one-step increments after development feedback',
            edits=[dict(path='control.json', before_sha256=value['sha256'], value=dict(increment=1))])]
    return dict(text=json.dumps(dict(parent_sha256=request['parent_sha256'], candidates=candidates)))
'''


def test_isolated_system2_runs_real_campaign_admission_and_lineage(tmp_path, isolation_runtime):
    strategy = IsolatedProposal(PROPOSAL, isolation=PythonIsolation(isolation_runtime), output=tmp_path / "strategy")
    parent, _, suite, campaign = setup(tmp_path, strategy)
    result = campaign.run(parent)
    assert [row["outcome"] for row in result["rounds"]] == ["inherited", "retained"]
    child = result["current"]["harness"]
    assert read_json(Path(child["root"]) / "control.json") == dict(increment=1)
    assert len(suite.resets) == 6
    calls = list((tmp_path / "strategy").glob("*/result.json"))
    assert len(calls) == 2
    assert campaign.run(parent) == result
    assert list((tmp_path / "strategy").glob("*/result.json")) == calls
    assert_stopped(tmp_path / "strategy")


def test_system2_process_timeout_is_not_automatically_retried(tmp_path, isolation_runtime):
    source = "def generate(request, limits):\n    while True: pass\n"
    strategy = IsolatedProposal(source, isolation=PythonIsolation(isolation_runtime), output=tmp_path / "strategy")
    parent, _, suite, campaign = setup(tmp_path, strategy, seconds=1)
    for _ in range(2):
        with pytest.raises(ReconciliationRequired):
            campaign.run(parent)
    assert len(list((tmp_path / "strategy").glob("*/request.json"))) == 1
    assert suite.resets == [3]
    assert campaign.state.resolve()["harness"] == parent
    assert_stopped(tmp_path / "strategy")


def test_isolated_system2_has_no_primitive_bridge(tmp_path, isolation_runtime):
    source = '''
import json
def generate(request, limits):
    with open('/output/result.json', 'w') as stream:
        json.dump(dict(kind='call', id=0, method='control.execute', args=[], kwargs={}), stream)
    while True: pass
'''
    strategy = IsolatedProposal(source, isolation=PythonIsolation(isolation_runtime), output=tmp_path / "strategy")
    parent, _, suite, campaign = setup(tmp_path, strategy)
    with pytest.raises(ReconciliationRequired):
        campaign.run(parent)
    process = read_json(next((tmp_path / "strategy").glob("*/processes/isolated/process.json")))
    assert process["calls"] == [] and process["stopped"]
    assert "capability" in process["error"]
    assert suite.resets == [3] and campaign.state.resolve()["harness"] == parent
