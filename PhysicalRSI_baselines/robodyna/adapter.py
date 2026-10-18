"""Thin Core ports backed by an isolated native RoboDyna process."""
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import select
import signal
import subprocess
import time
import threading
import uuid

from PhysicalRSI_core.contracts import Contract
from PhysicalRSI_core.embodiment import Embodiment, System1
from PhysicalRSI_core.experiments import TrialAdapters
from PhysicalRSI_core.infra.storage import digest, file_digest, read_json
from PhysicalRSI_core.self_harness.artifacts import verify_harness
from physicalrsi.skills.dynamic_interception import dynamic_interception

TASKS = ('catch_ramp_ball', 'catch_valley_ball')
PINNED_UPSTREAM = '6422e622d8025c3eaa4559bb4250c968fe4b2873'
OBSERVATION = Contract('robodyna-native-observation/v1')
ACTION = Contract('robodyna-native-action/v1')


class NativeSuite:
    def __init__(self, upstream_root, python_executable, *, runtime_environment=None,
                 worker_timeout_s=600, output_root):
        self.root = Path(upstream_root).resolve()
        self.python = str(Path(python_executable).absolute())
        self.runtime_environment = dict(runtime_environment or {})
        self.timeout = float(worker_timeout_s)
        self.output = Path(output_root).resolve()
        revision = subprocess.check_output(['git','-C',str(self.root),'rev-parse','HEAD'], text=True).strip()
        if revision != PINNED_UPSTREAM:
            raise ValueError('Native integration requires the audited upstream revision')
        self.upstream_revision = revision
        probe = "import importlib.metadata as m,json;print(json.dumps({n:m.version(n) for n in ['sapien','torch','nvidia-curobo','mplib','numpy','warp-lang','imageio-ffmpeg']}))"
        self.packages = json.loads(subprocess.check_output([self.python, '-c', probe], text=True))

    def identity(self):
        from physicalrsi.skills import dynamic_interception as module
        # Hash runtime source/configuration, including local path-expanded robot config.
        native_files = [p for directory in ('envs','task_config','assets/embodiments/ur5-wsg') for p in (self.root/directory).rglob('*')
                        if p.is_file() and p.suffix in ('.py','.yml','.json') and '__pycache__' not in p.parts]
        local = [Path(__file__), Path(__file__).with_name('worker.py'),
                 Path(__file__).with_name('execution.py'), Path(module.__file__)]
        return dict(kind='native-robodyna', upstream_revision=self.upstream_revision,
            python=self.python, packages=self.packages, runtime_environment=self.runtime_environment,
            native_sources={str(p.relative_to(self.root)):file_digest(p) for p in sorted(native_files)},
            adapter_sources={str(p):file_digest(p) for p in local},
            control_mode='native-interactive-robot', observation_mode='simulator-state-and-static-geometry')

    def admit(self, candidate):
        verify_harness(candidate)
        params = read_json(Path(candidate['root'])/'dynamic_interception.json')
        skill = dynamic_interception(**params)
        return dict(accepted=True, evidence=dict(composition_revision=skill.composition.revision))

    def trial(self, candidate, task):
        if task not in TASKS:
            raise ValueError('Unknown native task')
        environment = NativeEnvironment(self, task)
        return TrialAdapters(environment, NativePolicy(candidate, environment), NativeVerifier())


class NativeEnvironment:
    supports_video = True

    def __init__(self, suite, task):
        self.suite, self.task = suite, task
        self.process = None
        self.cached = None
        self.folder = None
        self._buffer = b''
        self._monitor_stop = threading.Event()
        self._monitor_thread = None

    def identity(self):
        return dict(suite=self.suite.identity(), task=self.task)

    def describe(self):
        return Embodiment('robodyna-ur5-wsg', digest(self.identity()), 'simulation', OBSERVATION, ACTION)

    def _rpc(self, command, value=None, context=None, timeout=None):
        if context:
            context.check()
        self.process.stdin.write(json.dumps(dict(command=command, value=value), allow_nan=False)+'\n')
        self.process.stdin.flush()
        deadline = time.monotonic()+(self.suite.timeout if timeout is None else timeout)
        if context and context.deadline is not None:
            deadline = min(deadline, context.deadline)
        while b'\n' not in self._buffer:
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Native worker timed out; inspect '+str(self.folder))
            ready, _, _ = select.select([self.reply_fd], [], [], min(remaining, 1.))
            if ready:
                data = os.read(self.reply_fd, 65536)
                if not data:
                    raise RuntimeError('Native worker exited: '+str(self.folder))
                self._buffer += data
            if context:
                context.check()
        line, self._buffer = self._buffer.split(b'\n', 1)
        response = json.loads(line)
        if not response['ok']:
            raise RuntimeError(response['error']+'; native log: '+str(self.folder/'worker.log'))
        return response['result']

    def reset(self, case, context):
        if self.process is not None:
            raise RuntimeError('Create a new isolated worker for each trial')
        if type(case['seed']) is not int or case['mode'] not in ('frozen','async'):
            raise ValueError('Invalid native evaluation case')
        self.folder = self.suite.output/uuid.uuid4().hex
        self.folder.mkdir(parents=True)
        self.log = (self.folder/'worker.log').open('xb')
        self.reply_fd, write_fd = os.pipe()
        worker = Path(__file__).with_name('worker.py')
        env = dict(os.environ, **self.suite.runtime_environment)
        env.pop('DISPLAY', None)
        env.setdefault('ROBODYNA_DISABLE_RAY_TRACING', '1')
        try:
            self.process = subprocess.Popen([self.suite.python, '-u', str(worker), '--upstream',str(self.suite.root),
                '--output',str(self.folder/'evidence'), '--reply-fd',str(write_fd)],
                stdin=subprocess.PIPE, stdout=self.log, stderr=subprocess.STDOUT,
                text=True, env=env, pass_fds=(write_fd,), start_new_session=True)
        except BaseException:
            os.close(self.reply_fd)
            self.log.close()
            raise
        finally:
            os.close(write_fd)
        try:
            (self.folder/'dispatch.json').write_text(json.dumps(dict(pid=self.process.pid, case=case,
                task=self.task, started_at=time.time(), machine=os.uname().nodename,
                implementation=self.identity(), gpu=subprocess.check_output(['nvidia-smi',
                    '--query-gpu=index,uuid,pci.bus_id,memory.used,utilization.gpu','--format=csv,noheader'],text=True, timeout=5)), indent=2))
            self._monitor_thread = threading.Thread(target=self._monitor, daemon=True)
            self._monitor_thread.start()
            self.cached = self._rpc('reset', dict(case, task=self.task), context)
        except BaseException:
            self.close()
            raise
        return dict(ready=True, observation=deepcopy(self.cached), terminated=self.cached['terminated'])

    def _monitor(self):
        with (self.folder/'gpu-resources.jsonl').open('x') as stream:
            while not self._monitor_stop.is_set():
                try:
                    result = subprocess.check_output(['nvidia-smi',
                        '--query-gpu=index,uuid,pci.bus_id,memory.used,utilization.gpu',
                        '--format=csv,noheader'], text=True, timeout=5)
                    stream.write(json.dumps(dict(time=time.time(), pid=self.process.pid,
                                                 gpu_csv=result))+'\n')
                    stream.flush()
                except Exception as error:
                    stream.write(json.dumps(dict(error=str(error)))+'\n')
                    stream.flush()
                self._monitor_stop.wait(1.)

    def observe(self):
        return deepcopy(self.cached)

    def step(self, action, context):
        self.cached = self._rpc('step', action, context)
        return dict(observation=deepcopy(self.cached), terminated=self.cached['terminated'])

    def native_success(self):
        return self.cached['native']['success']

    def native_progress(self):
        return self.cached['native']['progress']

    def episode_evidence(self):
        return self.export_evidence()

    def export_evidence(self):
        if self.folder is None:
            return {}
        return {str(p.relative_to(self.folder)):dict(path=str(p), sha256=file_digest(p))
                for p in self.folder.rglob('*') if p.is_file()}

    def close(self):
        if self.process is None:
            return
        try:
            if self.process.poll() is None:
                self._rpc('close', timeout=30)
                self.process.wait(timeout=30)
        except (Exception, KeyboardInterrupt):
            if self.process.poll() is None:
                os.killpg(self.process.pid, signal.SIGTERM)
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait(timeout=10)
        finally:
            self._monitor_stop.set()
            if self._monitor_thread is not None:
                self._monitor_thread.join(timeout=6)
            self.process.stdin.close()
            os.close(self.reply_fd)
            self.log.close()
            self.process = None
            try:
                final_gpu = subprocess.check_output(['nvidia-smi',
                    '--query-gpu=index,uuid,pci.bus_id,memory.used,utilization.gpu',
                    '--format=csv,noheader'], text=True, timeout=5)
                (self.folder/'gpu-final.txt').write_text(final_gpu)
            except Exception as error:
                (self.folder/'gpu-final-error.json').write_text(json.dumps(dict(
                    error_type=type(error).__name__, error=str(error))))



class NativePolicy:
    def __init__(self, candidate, environment):
        self.revision = verify_harness(candidate)
        self.skill = dynamic_interception(**read_json(Path(candidate['root'])/'dynamic_interception.json'))
        self.environment = environment
        self.latest = None

    def identity(self):
        return dict(revision=self.revision, skill=self.skill.identity())

    def describe(self):
        return System1('dynamic-interception', self.revision, OBSERVATION, ACTION)

    def begin_episode(self, task, case, observation, context):
        self.latest = None
        context.check()

    def act(self, observation, context):
        t0 = time.perf_counter()
        self.latest = self.skill.execute(observation['policy_observation'], context)
        action = (dict(kind='target', target_position_m=self.latest['target_position_m'])
                  if self.latest['feasible'] else dict(kind='observe', steps=25))
        return dict(action, prediction=self.latest, inference_seconds=time.perf_counter()-t0)

    def end_episode(self, context):
        self.environment.close()
        return dict(stopped=True)


class NativeVerifier:
    def identity(self):
        return dict(kind='native-robodyna-verdict', source=file_digest(Path(__file__)))

    def verify(self, case, trace):
        native = trace[-1]['observation']['native']
        success = native['success']
        return dict(outcome='success' if success else 'failure', reason='Official task check_success',
            measurements=dict(native_progress=native['progress'], native_success=success,
                latency_seconds=native['latency']['inference_latency_mean_s'],
                inference_latency_seconds=native['latency']['inference_latency_mean_s'],
                sim_time_seconds=native['sim_time_s'], score_detail=native['score_detail'],
                episode_wall_seconds=native['episode_wall_seconds'],
                execution_wall_seconds=native['execution_wall_seconds'],
                latency_scope=native['latency_scope'], stop_reason=native['stop_reason']))
