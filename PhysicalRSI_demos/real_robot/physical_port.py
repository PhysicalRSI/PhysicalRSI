"""Execution host client. SDKs and ownership stay in the separate worker."""
import json
import os
from pathlib import Path
import shutil
import socket
import threading
import time
import uuid

from .contracts import (ActionRejected, CapabilityUnavailable, ExecutionUncertain,
                        Observation, ObservationTimingError, SensorUnavailable)


class PhysicalRobot:
    capabilities = dict(reobserve=True, gripper=True, direct_eef=True, scene=False, approach=False,
                        limitation="Metric scene/planning are unavailable; host look/align/relate are not physically qualified; no collision/contact planner")
    def __init__(self, workspace, *, cameras, evidence):
        self.workspace, self.cameras = Path(workspace).resolve(), cameras
        self.evidence = Path(evidence).resolve()
        self.token = json.loads((self.workspace / "session.json").read_text())["token"]
        self.socket_path = json.loads((self.workspace / "status.json").read_text())["socket"]
        self.released = False
        self.closed = threading.Event()
        self.heartbeat_error = None
        self.last_observation = None
        self.beat = threading.Thread(target=self._heartbeat, name="robot-owner-heartbeat", daemon=True)
        self.beat.start()

    def rpc(self, method, arguments=None):
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as stream:
                stream.settimeout(2)
                stream.connect(self.socket_path)
                stream.sendall(json.dumps(dict(token=self.token, method=method,
                    arguments=arguments or {}), allow_nan=False).encode()+b"\n")
                with stream.makefile("rb") as file:
                    line = file.readline(4*1024*1024)
            response = json.loads(line)
        except Exception as exc:
            raise ExecutionUncertain("Hardware IPC failed; do not retry an action") from exc
        if not response["ok"]:
            if response.get("rejected"):
                raise ActionRejected(response["error"])
            raise ExecutionUncertain(response["error"])
        return response["result"]

    def _heartbeat(self):
        while not self.closed.is_set():
            try:
                self.rpc("heartbeat")
            except Exception as exc:
                self.heartbeat_error = exc
                return
            self.closed.wait(.25)

    def wait_ready(self):
        while True:
            status = self.rpc("status")
            if status["phase"] == "ready":
                return status
            if status["phase"] in {"needs_reconciliation", "closed", "closing"}:
                raise ExecutionUncertain("Worker is unavailable: " + status["phase"])
            time.sleep(.1)

    def observe(self):
        if self.released or self.heartbeat_error is not None:
            raise ExecutionUncertain("The hardware owner is unavailable")
        for attempt in range(3):
            try:
                if hasattr(self.cameras, "capture_with_state"):
                    frames, state = self.cameras.capture_with_state(lambda: self.rpc("snapshot"))
                else:
                    frames = self.cameras.capture()
                    state = self.rpc("snapshot")
            except ExecutionUncertain:
                raise
            except Exception as exc:
                raise SensorUnavailable("Fresh three-view RGB is unavailable") from exc
            obs = Observation(uuid.uuid4().hex, state["captured_ns"], frames,
                              tuple(state["qpos26"]), state["eef"])
            try:
                obs.validate(live=True)
            except ObservationTimingError as exc:
                if attempt < 2:
                    time.sleep(.02)
                    continue  # New RGB and real state; never rebind old frames.
                raise SensorUnavailable("Camera/state synchronization did not recover") from exc
            except ValueError as exc:
                raise SensorUnavailable("Camera/state observation is invalid") from exc
            self.last_observation = obs
            return obs

    def scene(self, names):
        raise CapabilityUnavailable("Camera/base and grasp-TCP calibration are not installed; "
                                    "PhysicalRSI metric grounded scene is unavailable")

    def approach(self, arm, position_m, quaternion_wxyz, *, checkpoint):
        raise CapabilityUnavailable("Scene-grounded full-goal planning is not calibrated; use L2")

    def _job(self, kind, targets, steps, checkpoint, observation=None):
        if self.last_observation is None:
            raise ActionRejected("Current camera observation is required")
        observed = observation or self.last_observation
        if observed.id != self.last_observation.id or observed.captured_ns != self.last_observation.captured_ns:
            raise ActionRejected("Action must use the exact current observation, not refreshed timestamps")
        job_id = uuid.uuid4().hex
        request = dict(id=job_id, kind=kind, targets=targets, steps=steps,
                       observed_ns=observed.captured_ns)
        if kind == "direct":
            # An unchanged observed EEF means hold, not a new IK objective.
            # Otherwise repeated measured-pose targets can ratchet a passive
            # arm as it sags and the redundant IK posture objective changes.
            request["controlled"] = [side for side, pose in targets.items()
                if any(pose[key] != observed.eef[side][key]
                       for key in ("position", "quaternion_wxyz"))]
        self.rpc("submit", request)
        deadline = time.monotonic()+70
        while time.monotonic() < deadline:
            receipt = self.rpc("job", dict(id=job_id))
            if receipt["status"] not in {"accepted", "running", "committing"}:
                receipt = self._retain_receipt(receipt)
                if receipt["status"] in {"stopped", "interrupted"}:
                    raise ExecutionUncertain("Hardware job was interrupted")
                return receipt
            try:
                checkpoint()
            except SensorUnavailable as exc:
                # RGB acquisition can time out while the separate SDK worker
                # finishes the already admitted bounded action. Poll that exact
                # job read-only; do not resubmit or extend its movement horizon.
                final = self.rpc("job", dict(id=job_id))
                settle_deadline = min(deadline, time.monotonic()+1.)
                while (final["status"] in {"accepted", "running", "committing"}
                       and time.monotonic() < settle_deadline):
                    time.sleep(.02)
                    final = self.rpc("job", dict(id=job_id))
                if final["status"] not in {"completed", "residual"}:
                    raise ExecutionUncertain("RGB lost before action completion was confirmed") from exc
                retained = self._retain_receipt(final)
                raise SensorUnavailable(str(exc), completed_action=retained) from exc
            time.sleep(.02)
        raise ExecutionUncertain("Hardware job completion is unknown; do not retry")

    def _retain_receipt(self, receipt):
        source = self.workspace / "jobs" / receipt["id"]
        target = self.evidence / "hardware" / receipt["id"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target)
        return dict(receipt, evidence=str(target), independent_success=False)

    def act_eef(self, targets, steps, *, checkpoint, observation=None):
        return self._job("direct", targets, steps, checkpoint, observation=observation)

    def gripper(self, arm, action, mode=None, *, checkpoint, observation=None):
        from .gripper import command
        request = command(arm, action, mode)
        return self._job("gripper", {arm:{k:request[k] for k in ("action", "mode")}},
                         None, checkpoint, observation=observation)

    def hold(self):
        if not self.released:
            self.rpc("hold")

    def release_for_human(self):
        self.stop()

    def stop(self):
        if self.released:
            return
        try:
            self.rpc("release")
        except ExecutionUncertain:
            # A local operator may already have stopped and closed this exact
            # worker. Its durable stop receipt, not a new connection, decides.
            status = json.loads((self.workspace / "status.json").read_text())
            if not status.get("closed"):
                raise
        self.closed.set()
        self.beat.join(timeout=2)
        until = time.monotonic()+15
        while time.monotonic() < until:
            status = json.loads((self.workspace / "status.json").read_text())
            if status["phase"] == "needs_reconciliation":
                raise ExecutionUncertain("Hardware stop requires reconciliation")
            if status.get("closed"):
                try:
                    os.kill(status["pid"], 0)
                except ProcessLookupError:
                    self.released = True
                    return
                # A reaped process is ideal; zombies have no FDs or owners.
                stat = Path(f"/proc/{status['pid']}/stat")
                if stat.exists() and stat.read_text().split(") ", 1)[1].startswith("Z "):
                    self.released = True
                    return
            time.sleep(.05)
        raise ExecutionUncertain("Worker exit/lock release is not confirmed")
