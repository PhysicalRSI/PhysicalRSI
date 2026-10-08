"""Observation-gated stage composition for code and learned inference providers.

Providers propose actions; only the caller's admitted controller actuates. A
returned chunk must be acknowledged before another observation or handoff.
This is a single-episode coordinator, not a native controller or a sandbox.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from functools import wraps
from pathlib import Path
from threading import Lock
from time import monotonic

from .contracts import Contract, Context, Operation, ReconciliationRequired
from .infra.storage import atomic_json, digest, file_digest
from .timing import ControlTiming, finite_seconds, observation_reference

REQUEST = Contract('physicalrsi.skill-stage-request/v1')
CHECK = Contract('physicalrsi.skill-stage-check/v1')
RELEASE = Contract('physicalrsi.skill-release/v1')
STOPPED = Contract('physicalrsi.skill-stopped/v1')


def _serial(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('Concurrent or reentrant executor calls are forbidden')
        try:
            return method(self, *args, **kwargs)
        finally:
            self._lock.release()
    return call


@dataclass(frozen=True)
class SkillStage:
    name: str
    goal: str
    provider: Operation
    monitor: Operation
    release: Operation
    max_actions: int
    max_observations: int
    seconds: float

    def __post_init__(self):
        if not self.name.strip() or not self.goal.strip():
            raise ValueError('Stage needs a name and grounded goal')
        if any(type(n) is not int or n < 1 for n in (self.max_actions, self.max_observations)):
            raise ValueError('Positive stage budgets required')
        finite_seconds(self.seconds)
        if (self.provider.input != REQUEST or self.monitor.input != REQUEST
                or self.monitor.output != CHECK or self.release.input != RELEASE
                or self.release.output != STOPPED):
            raise ValueError('Stage operation contracts differ')

    def identity(self):
        def operation(value):
            return dict(name=value.name, revision=value.revision,
                        input=asdict(value.input), output=asdict(value.output))
        return dict(name=self.name, goal=self.goal,
                    provider=operation(self.provider), monitor=operation(self.monitor),
                    release=operation(self.release), max_actions=self.max_actions,
                    max_observations=self.max_observations, seconds=self.seconds)


class SkillExecutor:
    """Serial inference ownership with independently checked stage completion.

Monitor status is running, succeeded, failed or unknown, with nonempty public
evidence. Unknown clears provider state and issues no action. Failed terminates
the plan; recovery requires a separately reviewed plan. Provider calls must be
bounded externally as well as cooperating with Context deadlines.
"""
    def __init__(self, stages, *, instruction, action_contract, timing, validate_actions,
                 validator_revision, workspace, teardown_seconds=2):
        self.stages = tuple(stages)
        if not self.stages or any(not isinstance(s, SkillStage) for s in self.stages):
            raise ValueError('A nonempty stage plan is required')
        if len({s.name for s in self.stages}) != len(self.stages):
            raise ValueError('Stage names must be unique')
        if not isinstance(timing, ControlTiming) or not callable(validate_actions):
            raise ValueError('Controller timing and action validator required')
        if (not isinstance(instruction, str) or not instruction.strip()
                or not isinstance(validator_revision, str) or not validator_revision.strip()):
            raise ValueError('Full task instruction and validator revision required')
        if any(s.provider.output != action_contract for s in self.stages):
            raise ValueError('Provider action contract differs from controller; use an explicit adapter')
        self.timing, self.validate_actions = timing, validate_actions
        self.teardown_seconds = finite_seconds(teardown_seconds)
        self.root = Path(workspace)
        self.root.mkdir(parents=True, exist_ok=False)
        self._identity = dict(schema='physicalrsi.skill-executor/v1',
                             stages=[s.identity() for s in self.stages],
                             instruction=instruction, validator_revision=validator_revision,
                             implementation_sha256=file_digest(Path(__file__)),
                             action_contract=asdict(action_contract), timing=asdict(timing))
        atomic_json(self.root / 'plan.json', self._identity)
        self.revision = digest(self._identity)
        self.index = 0
        self.state = 'ready'
        self.pending = None
        self.owner = False
        self.episode = None
        self.clock_domain = None
        self.sequence = -1
        self.captured_at = -1
        self.started = None
        self.actions = self.observations = self.event_index = 0
        self._lock = Lock()

    @property
    def identity(self):
        return deepcopy(self._identity)

    def _event(self, kind, **values):
        atomic_json(self.root / 'events' / f'{self.event_index:06d}.json',
                    dict(kind=kind, stage_index=self.index, plan_revision=self.revision,
                         episode=self.episode, **values))
        self.event_index += 1

    def _release(self):
        if not self.owner:
            return
        # Inference cleanup must remain possible after the execution deadline.
        stage = self.stages[self.index]
        cleanup = Context(self.episode, deadline=monotonic() + self.teardown_seconds)
        try:
            reply = stage.release(dict(stage=stage.name), cleanup)
            if not isinstance(reply, dict) or reply.get('stopped') is not True:
                raise ReconciliationRequired('Provider did not acknowledge release')
            self._event('released', provider=stage.provider.revision)
        except BaseException as error:
            self.state = 'needs_reconciliation'
            raise ReconciliationRequired('Provider shutdown is uncertain') from error
        self.owner = False

    def _result(self, status, **values):
        result = dict(status=status, stage_index=self.index, qualification=None, **values)
        self._event(status, result=result)
        return result

    @_serial
    def advance(self, observation, context):
        if self.pending is not None:
            raise ReconciliationRequired('Acknowledge the outstanding controller chunk first')
        if self.state != 'ready':
            raise RuntimeError('Executor is not ready: ' + self.state)
        try:
            context.check()
            observation = deepcopy(observation)
            reference = observation_reference(observation)
            now = monotonic()
            captured = finite_seconds(observation['captured_at'], positive=False)
            sequence = observation['sequence']
            clock = observation['clock_domain']
            if (observation['episode'] != context.episode
                    or (self.episode is not None and context.episode != self.episode)
                    or not isinstance(clock, str) or not clock
                    or (self.clock_domain is not None and clock != self.clock_domain)
                    or type(sequence) is not int or sequence <= self.sequence
                    or captured < self.captured_at or captured > now
                    or now - captured >= self.timing.max_observation_age_seconds):
                raise ValueError('Observation is stale or belongs to a different episode/clock')
            self.episode, self.clock_domain = context.episode, clock
            self.sequence, self.captured_at = sequence, captured
            if self.started is None:
                self.started = now
            stage = self.stages[self.index]
            self.observations += 1
            if self.observations > stage.max_observations or now >= self.started + stage.seconds:
                self._release()
                self.state = 'failed'
                return self._result('failed', reason='stage budget exhausted')
            deadline = min(context.deadline or float('inf'), self.started + stage.seconds)
            bounded = replace(context, deadline=deadline)
            request = dict(instruction=self._identity['instruction'], goal=stage.goal,
                           stage=stage.name, observation=observation)
            verdict = stage.monitor(deepcopy(request), bounded)
            if (not isinstance(verdict, dict) or set(verdict) != {'status', 'evidence'}
                    or verdict['status'] not in {'running', 'succeeded', 'failed', 'unknown'}
                    or not isinstance(verdict['evidence'], dict) or not verdict['evidence']):
                raise ValueError('Monitor must return a status and public evidence')
            self._event('checked', observation=reference, verdict=verdict)
            status = verdict['status']
            if status != 'running':
                self._release()
                if status == 'unknown':
                    return self._result('needs_observation', evidence=verdict['evidence'])
                if status == 'failed':
                    self.state = 'failed'
                    return self._result('failed', evidence=verdict['evidence'])
                self.index += 1
                self.started = None
                self.actions = self.observations = 0
                if self.index == len(self.stages):
                    self.state = 'completed'
                    return self._result('completed', evidence=verdict['evidence'])
                return self._result('handoff', evidence=verdict['evidence'])
            if self.actions >= stage.max_actions:
                self._release()
                self.state = 'failed'
                return self._result('failed', reason='action budget exhausted')
            if not self.owner:
                # Clear any prior episode/chunk state on first acquisition as well.
                self.owner = True
                self._release()
                bounded.check()
            # Mark ownership before entering provider code: failure may leave state.
            self.owner = True
            chunk = stage.provider(deepcopy(request), bounded)
            if (not isinstance(chunk, dict)
                    or set(chunk) != {'schema', 'observation', 'period_seconds', 'actions'}
                    or chunk['schema'] != 'physicalrsi.action-chunk/v1'
                    or chunk['observation'] != reference
                    or chunk['period_seconds'] != self.timing.period_seconds
                    or not isinstance(chunk['actions'], list)
                    or not 1 <= len(chunk['actions']) <= min(self.timing.max_chunk_steps,
                                                           stage.max_actions - self.actions)):
                raise ValueError('Provider returned an invalid or mismatched action chunk')
            self.validate_actions(deepcopy(chunk['actions']))
            bounded.check()
            if monotonic() - captured >= self.timing.max_observation_age_seconds:
                raise ValueError('Observation expired during inference')
            self.actions += len(chunk['actions'])
            token = digest(dict(plan=self.revision, stage=self.index, observation=reference,
                                chunk=chunk, actions=self.actions))
            self.pending = token
            return self._result('action', token=token, command=deepcopy(chunk),
                                provider_revision=stage.provider.revision)
        except BaseException:
            if self.pending is not None:
                self.state = 'needs_reconciliation'
            elif self.state != 'needs_reconciliation':
                self.state = 'failed'
                self._release()
            raise

    @_serial
    def acknowledge(self, token):
        """Caller attests the entire chunk finished; never an assertion of success.

        On uncertain execution leave it pending and reconcile with the native
        controller. This module cannot independently attest caller honesty.
        """
        if self.state != 'ready' or self.pending is None or token != self.pending:
            raise ReconciliationRequired('Controller acknowledgement does not match pending action')
        self._event('acknowledged', token=token)
        self.pending = None

    @_serial
    def close(self):
        if self.pending is not None or self.state == 'needs_reconciliation':
            raise ReconciliationRequired('Reconcile outstanding execution before closing')
        self._release()
        if self.state == 'ready':
            self.state = 'closed'
