"""Controller arbitration; monitor judgments never enter policy observations."""
import math
import time

from .contracts import (Assessment, CapabilityUnavailable, ExecutionUncertain,
                        HumanDecisionPending, Level, Outcome, Progress, SensorUnavailable, TaskFinished, TargetLost)


class Coordinator:
    def __init__(self, *, task, robot, monitor, judge, episode, controllers, monitor_period_s=1,
                 allowed_levels=tuple(Level), human_judgment_timeout_s=180, before_success_stop=None,
                 rescue_learning=None):
        if set(controllers) != set(Level) or monitor_period_s <= 0:
            raise ValueError("L1/L2/L3 and a positive monitor period are required")
        self.task, self.robot, self.monitor, self.judge = task, robot, monitor, judge
        self.episode, self.controllers = episode, controllers
        self.monitor_period_s = monitor_period_s
        if allowed_levels not in (tuple(Level), (Level.L1,)):
            raise ValueError("Use the full hierarchy or L1-only independent validation")
        self.allowed_levels = allowed_levels
        if (type(human_judgment_timeout_s) not in (int, float)
                or not math.isfinite(human_judgment_timeout_s) or human_judgment_timeout_s <= 0):
            raise ValueError("Human judgment timeout must be positive and finite")
        self.human_judgment_timeout_s = human_judgment_timeout_s
        self.before_success_stop = before_success_stop
        self.rescue_learning = rescue_learning
        self.level = None
        self.last_observation = None
        self.last_assessment = None
        self.pending = None
        self.pending_observation = None
        self.last_submit = float("-inf")
        self.last_assessed_id = None
        self.action_revision = 0
        self.pending_revision = None
        self.human_pending = None
        self.human_revision = None
        self.human_observation = None
        self.human_unavailable = False
        self.human_resolved_revision = None
        self.pending_level = self.human_level = self.finished_by = None

    def _accept_assessment(self, assessment, revision, level):
        # A delayed judgment describes its own images. If another physical action
        # occurred, retain it as evidence and obtain a fresh judgment; it cannot
        # terminate a later controller or claim credit for that controller.
        if revision != self.action_revision:
            return
        self.last_assessment = assessment
        if assessment.target_lost:
            self.robot.hold()
            raise TargetLost()
        if (assessment.outcome == Outcome.SUCCESS or
                (assessment.outcome == Outcome.FAILURE and self.allowed_levels == (Level.L1,))):
            self.finished_by = level
            self.robot.hold()
            if assessment.outcome == Outcome.SUCCESS and self.before_success_stop is not None:
                # Keep ownership and grasp commands until the operator secures
                # the object. A normal driver shutdown can otherwise drop it.
                # Errors/stop requests still unwind through the normal stop path.
                self.before_success_stop()
            raise TaskFinished()

    def _resolve_human(self):
        if self.human_pending is None or not self.human_pending.done():
            return
        pending, observed, revision, level = self.human_pending, self.human_observation, self.human_revision, self.human_level
        self.human_pending = self.human_observation = self.human_revision = None
        try:
            assessment = pending.result()
        except HumanDecisionPending:
            self.human_unavailable = True
            return
        if assessment.observation_id != observed.id:
            raise ValueError("Human judgment references another observation")
        self.episode.assessment(assessment)
        self.human_resolved_revision = revision
        self._accept_assessment(assessment, revision, level)

    def _request_human(self, observed, assessment, revision, level):
        self.human_pending = self.judge.submit(self.task, observed, assessment)
        self.human_observation, self.human_revision = observed, revision
        self.human_level = level

    def _wait_for_final_human(self):
        """Only a finished L1-only trial waits; exploration stays asynchronous.

        A pending answer to pre-action images is retained but cannot settle the
        final state. Once consumed, request a judgment for the final observation.
        Unknown/unavailable human answers remain uncertain, never failed trials.
        """
        if self.judge is None:
            return
        deadline = time.monotonic() + self.human_judgment_timeout_s
        while True:
            self._resolve_human()
            if (self.human_unavailable or self.human_resolved_revision == self.action_revision
                    or self.last_assessment is not None and self.last_assessment.outcome != Outcome.UNCERTAIN):
                return
            if time.monotonic() >= deadline:
                return
            if self.human_pending is None:
                self._request_human(self.last_observation, self.last_assessment,
                                    self.action_revision, self.level)
                continue
            # This is a held-state evaluation interval, not permission for a new
            # action or another controller. Hardware/stop errors still propagate.
            self.robot.hold()
            time.sleep(min(.05, max(0, deadline - time.monotonic())))

    def _resolve_monitor(self, *, wait):
        self._resolve_human()
        if self.pending is None or (not wait and not self.pending.done()):
            return
        observed = self.pending_observation
        revision = self.pending_revision
        level = self.pending_level
        try:
            assessment = self.pending.result()
        except Exception as exc:
            assessment = Assessment(observed.id, Outcome.UNCERTAIN, Progress.UNKNOWN,
                                    "视觉监控未能给出有效判定：" + type(exc).__name__)
        finally:
            self.pending = self.pending_observation = self.pending_revision = None
        self.episode.assessment(assessment)
        if (assessment.outcome == Outcome.UNCERTAIN and self.judge is not None
                and self.human_pending is None and not self.human_unavailable):
            self._request_human(observed, assessment, revision, level)
        self.last_assessed_id = observed.id
        self._accept_assessment(assessment, revision, level)
        self._resolve_human()

    def checkpoint(self, *, force=False, observation=None):
        """Called from host control ticks; returns ONLY public measured evidence."""
        # Force refreshes sensor evidence, not a synchronous GPT/human barrier.
        # The independent monitor keeps running while controllers explore.
        from .guidance import consume
        consume(self.episode)
        self._resolve_monitor(wait=False)
        now = time.monotonic()
        # SDK control loops call this frequently. Capture/inference are throttled,
        # but a completed monitor decision is handled on the next control tick.
        if not force and observation is None and now - self.last_submit < self.monitor_period_s:
            return self.last_observation
        try:
            obs = observation or self.robot.observe()
            self.episode.observation(obs)
        except SensorUnavailable:
            raise
        except Exception as exc:
            raise ExecutionUncertain("Current sensor evidence is unavailable; do not switch controllers") from exc
        self.last_observation = obs
        if self.pending is None and obs.id != self.last_assessed_id:
            self.pending = self.monitor.submit(obs)
            self.pending_observation = obs
            self.pending_revision = self.action_revision
            self.pending_level = self.level
            self.last_submit = now
        self._resolve_monitor(wait=False)
        return obs

    def public_observation(self):
        obs = self.checkpoint(force=True)
        return obs.public()

    def action(self, method, arguments, invoke):
        """Persist intent before dispatch; unknown effects never fall through to L2/L3."""
        self._resolve_monitor(wait=False)
        intent = self.episode.event("action_started", level=self.level.value,
                                    method=method, arguments=arguments,
                                    observation_id=self.last_observation.id if self.last_observation else None)
        self.action_revision += 1
        self.last_assessment = None
        try:
            receipt = invoke()
        except (TaskFinished, TargetLost, HumanDecisionPending):
            self.episode.event("action_interrupted_by_monitor", level=self.level.value, intent=intent)
            raise
        except CapabilityUnavailable:
            # Port contract guarantees pre-execution rejection for this exception.
            self.episode.event("action_unavailable", level=self.level.value, intent=intent)
            raise
        except SensorUnavailable as exc:
            if exc.completed_action is None:
                self.episode.event("action_effect_uncertain", level=self.level.value, intent=intent,
                                   error_type=type(exc).__name__)
                raise ExecutionUncertain("Sensor loss without confirmed action receipt") from exc
            self.episode.event("action_completed", level=self.level.value, intent=intent,
                               receipt=exc.completed_action)
            raise
        except BaseException as exc:
            self.episode.event("action_effect_uncertain", level=self.level.value, intent=intent,
                               error_type=type(exc).__name__)
            raise ExecutionUncertain("Action failed after dispatch; inspect retained evidence") from exc
        self.episode.event("action_completed", level=self.level.value, intent=intent, receipt=receipt)
        self.checkpoint(force=True)
        return receipt

    def run(self):
        status = "completed"
        outcome = Outcome.UNCERTAIN
        solved_by = None
        try:
            self.checkpoint(force=True)
            index = 0
            unsuccessful_l2_rescues = 0
            while index < len(self.allowed_levels):
                level = self.allowed_levels[index]
                self.level = level
                if level == Level.L1:
                    self.episode.event("attempt_started", level="L1")
                self.episode.event("controller_enter", level=level.value)
                returned = None
                try:
                    returned = self.controllers[level].run(self)
                except CapabilityUnavailable as exc:
                    self.episode.event("controller_unavailable", level=level.value, reason=str(exc))
                # A controller returning or claiming success is never an outcome.
                self._resolve_monitor(wait=False)
                self.robot.hold()
                self.episode.event("controller_returned", level=level.value)
                if self.rescue_learning is not None and level in {Level.L2, Level.L3}:
                    changed = self.rescue_learning(self, level)
                    if level == Level.L2 and not (returned and returned.get("learning_checkpoint")):
                        unsuccessful_l2_rescues += 1
                    if level == Level.L3:
                        unsuccessful_l2_rescues = 0
                    if changed and not (level == Level.L2 and unsuccessful_l2_rescues >= 2):
                        # New development candidate, same scene. This is not a
                        # fresh L1-only validation or evidence of independent success.
                        index = 0
                        continue
                    if level == Level.L3 or returned and returned.get("learning_checkpoint"):
                        index = self.allowed_levels.index(Level.L2)
                        continue
                index += 1
            if self.allowed_levels == (Level.L1,):
                # An earlier in-flight monitor result may describe the scene
                # before the last action. Settle it, then judge fresh held-state
                # RGB; absence of a current success is not itself task failure.
                self._resolve_monitor(wait=True)
                self.checkpoint(force=True)
                self._resolve_monitor(wait=True)
                self._wait_for_final_human()
                if self.last_assessment is None or self.last_assessment.outcome == Outcome.UNCERTAIN:
                    status = ("awaiting_human_judgment" if self.human_pending is not None or self.human_unavailable
                              else "evaluation_uncertain")
                else:
                    # The finished policy left an independently observable,
                    # unfinished task. Success/failure assessments exit earlier
                    # via TaskFinished; ongoing here is a definite failed trial.
                    outcome = Outcome.FAILURE
            else:
                # Controller exhaustion is not completion, nor proof of task
                # failure. The session runner must retain an unfinished task.
                status = "exploration_unfinished"
        except TargetLost:
            status = "target_dropped_and_unlocatable"
        except TaskFinished:
            outcome = self.last_assessment.outcome
            solved_by = self.finished_by if outcome == Outcome.SUCCESS else None
        except HumanDecisionPending:
            status = "awaiting_human_judgment"
        except SensorUnavailable as exc:
            status = "awaiting_sensor_recovery"
            self._record_interruption(exc, status)
        except ExecutionUncertain as exc:
            status = "needs_reconciliation"
            self._record_interruption(exc, status)
        except (KeyboardInterrupt, EOFError):
            status = "interrupted"
        except Exception as exc:
            status = "system_error"
            # Exception details may include evaluator text: keep them private.
            from .evidence import write_new
            write_new(self.episode.private / "system-error.json", dict(type=type(exc).__name__, message=str(exc)))
        finally:
            try:
                self.robot.stop()
            except BaseException:
                status, outcome, solved_by = "stop_unconfirmed", Outcome.UNCERTAIN, None
            self.monitor.close()
        return self.episode.finish(outcome=outcome, solved_by=solved_by, status=status)

    def _record_interruption(self, exc, status):
        # Full exception chains are diagnostics only, never execution feedback:
        # a nested exception could originate in independent monitoring.
        from .evidence import write_new
        chain, seen = [], set()
        while exc is not None and id(exc) not in seen:
            seen.add(id(exc))
            chain.append(dict(type=type(exc).__name__, message=str(exc)))
            exc = exc.__cause__ or exc.__context__
        write_new(self.episode.private / "interruption.json", dict(status=status, causes=chain))
