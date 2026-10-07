"""An independent, continuously advancing native MuJoCo actuator process.

The fault demonstration kills only the controller process. The surviving device
process must inhibit effort and measure settling without any client heartbeat.
This is simulation fault evidence; no physical watchdog or robot is qualified.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import replace
from functools import partial
import json
import math
import os
from pathlib import Path
import secrets
import signal
import sys
from tempfile import TemporaryDirectory
from time import monotonic, sleep

from PhysicalRSI_core.contracts import Context
from PhysicalRSI_core.infra.actuation import ActuationGuard, ActuationLimits, local_clock_domain
from PhysicalRSI_core.infra.actuation_rpc import ActuationClient, ActuationDriver, ActuationHost
from PhysicalRSI_core.infra.clock_sync import ClockSyncPolicy
from PhysicalRSI_core.infra.controller import ControllerGateway
from PhysicalRSI_core.infra.controller_rpc import ControllerClient, ControllerHost, controller_transport
from PhysicalRSI_core.infra.operator import OperatorClient, recovery_request
from PhysicalRSI_core.infra.rpc.tls import ClientTLS, ServerTLS, certificate_fingerprint
from PhysicalRSI_core.infra.services import ServiceSpec, Services
from PhysicalRSI_core.infra.storage import atomic_json, digest, file_digest, locked, read_json
from PhysicalRSI_core.timing import ControlTiming, action_chunk
from .mujoco_control import MuJoCoDriver, specification

TIMING = ControlTiming(.1, 3, 2, .5)
LIMITS = ActuationLimits(lease_seconds=2, max_command_seconds=1, max_poll_gap_seconds=.2)
SCOPE = "Continuously advancing MuJoCo watchdog fixture; simulation only; no physical qualification"
MODULE = "PhysicalRSI_demos.mujoco_watchdog"


def context(seconds=10):
    return Context("watchdog-trial", deadline=monotonic() + seconds, harness_revision="reviewed-watchdog-fixture")


class ContinuousMuJoCo:
    def __init__(self, model, *, clock_offset_seconds=0):
        if (type(clock_offset_seconds) not in (int, float) or not math.isfinite(clock_offset_seconds)
                or abs(clock_offset_seconds) > 10000):
            raise ValueError("Declare a finite simulation clock offset within 10000 seconds")
        self.clock_offset_seconds = float(clock_offset_seconds)
        self.driver = MuJoCoDriver(model)
        # A declared passive damper makes stopping observable within this small
        # fault demonstration. It is not an inferred brake for a physical robot.
        self.driver.model.dof_damping[:] = 5
        self.remainder = 0.
        self.inhibited = True

    def identity(self):
        return dict(driver=self.driver.identity(), damping=5, settled_speed=.02,
                    simulation_clock_offset_seconds=self.clock_offset_seconds,
                    implementation=file_digest(Path(__file__)))

    def validate(self, actions):
        self.driver.validate(actions)

    def apply(self, action):
        self.validate([action])
        self.driver.data.ctrl[:] = action
        self.inhibited = False

    def stop(self):
        self.driver.data.ctrl[:] = 0
        self.inhibited = True

    def quiescence(self):
        speed = abs(float(self.driver.data.qvel[0]))
        return dict(quiescent=self.inhibited and speed <= .02, output_inhibited=self.inhibited,
                    measured_speed=speed, effort=self.driver.data.ctrl.tolist(),
                    simulation_seconds=float(self.driver.data.time))

    def reset(self, case):
        self.stop()
        self.remainder = 0
        return self.driver.reset(case, context())

    def observe(self):
        sample = self.driver.observe(context())
        return replace(sample, captured_at=sample.captured_at + self.clock_offset_seconds)

    def tick(self, elapsed):
        self.remainder += elapsed
        dt = self.driver.model.opt.timestep
        # Bound catch-up work after a suspended process. The guard has already
        # inhibited expired output; omitted wall time is not simulated evidence.
        steps = min(int(self.remainder / dt), 100)
        self.remainder = self.remainder % dt
        for _ in range(steps):
            self.driver.mujoco.mj_step(self.driver.model, self.driver.data)


def make_backend(root, model, *, clock_offset_seconds=0):
    backend = ContinuousMuJoCo(model, clock_offset_seconds=clock_offset_seconds)
    domain = local_clock_domain()
    if clock_offset_seconds:
        domain = "simulation-offset:" + digest(dict(base_clock=domain, offset=backend.clock_offset_seconds))
    return ActuationGuard(root, backend=backend, resources=("independent-mujoco:" + model,), limits=LIMITS,
                          clock=lambda: monotonic() + backend.clock_offset_seconds, clock_domain=domain)


def make_controller(root, model, client):
    # Validation is duplicated at the device boundary. This local validator has
    # no actuator handle and cannot affect the running native device process.
    bound = 10 if model == "slider" else 3

    def validate(actions):
        import math
        for action in actions:
            if (not isinstance(action, list) or len(action) != 1 or type(action[0]) not in (int, float)
                    or not math.isfinite(action[0]) or not -bound <= action[0] <= bound):
                raise ValueError("Expected one finite effort inside the fixture limit")

    driver = ActuationDriver(client, validator=validate,
        validation_identity=dict(effort_bound=bound, source=file_digest(Path(__file__))))
    base = specification(model)
    spec = replace(base, timing=TIMING, resources=("independent-mujoco:" + model,),
                   revision=digest(dict(model=base.revision, passive_damping=5, continuous=True)))
    return ControllerGateway(root, driver=driver, embodiment=spec)


def _tls_fixture(directory):
    from .tls_fixture import certificates
    pki = certificates(directory / "certificates", names=("backend-server", "controller-server", "backend-client", "controller-client"))
    configs = {}
    for kind, methods in dict(
        backend=["actuation.describe", "actuation.inspect", "actuation.status", "actuation.call", "actuation.clock", "actuation.intent"],
        controller=["control.describe", "control.state", "control.status", "control.ownership", "control.execute"],
    ).items():
        server = dict(ca=pki["ca"], **pki[kind + "-server"], peers={certificate_fingerprint(pki[kind + "-client"]["certificate"]):
            dict(principal="simulation-" + kind + "-client", methods=["healthz", "service.describe", "shutdown", *methods])})
        client = dict(ca=pki["ca"], **pki[kind + "-client"], server_sha256=certificate_fingerprint(pki[kind + "-server"]["certificate"]))
        for role, value in dict(server=server, client=client).items():
            path = directory / (kind + "-" + role + ".json")
            atomic_json(path, value)
            configs[kind + "_" + role] = path
    return configs


def run(workspace, *, model="slider", recover_after_fault=False, clock_offset_seconds=0, clock_sync=False, tls_fixture=False):
    root = Path(workspace).resolve()
    with locked(root / ".fault.lock"), ExitStack() as stack:
        # Keep private sockets and ephemeral test keys outside the evidence
        # workspace. No credentials survive this simulation fixture's lifetime.
        operator = Path(stack.enter_context(TemporaryDirectory(prefix="prsi-watchdog-op-"))) if recover_after_fault else None
        tls = _tls_fixture(Path(stack.enter_context(TemporaryDirectory(prefix="prsi-watchdog-tls-")))) if tls_fixture else {}
        return _run(root, model, operator_directory=operator, clock_offset_seconds=clock_offset_seconds, clock_sync=clock_sync, tls=tls)


def _run(root, model, *, operator_directory=None, clock_offset_seconds=0, clock_sync=False, tls=None):
    if any(path.name != ".fault.lock" for path in root.iterdir()):
        raise ValueError("Fault demonstrations require a fresh workspace; never repeat an uncertain trial")
    atomic_json(root / "fault-study.json", dict(schema="physicalrsi.actuation-fault-study/v1", model=model, scope=SCOPE,
                                              recovery_requested=operator_directory is not None,
                                              clock_offset_seconds=clock_offset_seconds, clock_sync=clock_sync, mutual_tls=bool(tls)))
    backend_transport = partial(controller_transport, tls=ClientTLS(**read_json(tls["backend_client"]))) if tls else controller_transport
    control_transport = partial(controller_transport, tls=ClientTLS(**read_json(tls["controller_client"]))) if tls else controller_transport
    backend_tls = ("--tls-server-config", str(tls["backend_server"])) if tls else ()
    controller_tls = ("--tls-server-config", str(tls["controller_server"]), "--tls-client-config", str(tls["backend_client"])) if tls else ()
    probe = make_backend(root / "backend", model, clock_offset_seconds=clock_offset_seconds)
    expected = probe.identity()
    probe.close()
    atomic_json(root / "backend-identity.json", expected)
    backend_operator = ("--operator-socket", str(operator_directory / "backend.sock")) if operator_directory else ()
    controller_operator = ("--operator-socket", str(operator_directory / "controller.sock")) if operator_directory else ()
    controller_clock = ("--clock-sync",) if clock_sync else ()
    backend_spec = ServiceSpec("backend", expected, command=(sys.executable, "-m", MODULE,
        "--serve", "backend", "--workspace", str(root), "--model", model,
        "--clock-offset-seconds", str(clock_offset_seconds)) + backend_operator + backend_tls)
    with Services(root / "backend-service", client_factory=backend_transport) as backend_services:
        rpc = backend_services.start([backend_spec])["backend"]
        backend = ActuationClient(rpc, expected=expected, clock_policy=ClockSyncPolicy() if clock_sync else None)
        endpoint = read_json(backend_services.handles["backend"]["announcement"])["endpoint"]
        probe = make_controller(root / "controller", model, backend)
        controller_expected = probe.identity()
        probe.close()
        spec = ServiceSpec("controller", controller_expected, command=(sys.executable, "-m", MODULE,
            "--serve", "controller", "--workspace", str(root), "--model", model, "--backend-endpoint", endpoint) + controller_operator + controller_clock + controller_tls)
        with Services(root / "controller-service", client_factory=control_transport) as services:
            control_rpc = services.start([spec])["controller"]
            controller = ControllerClient(control_rpc, expected=controller_expected)
            credential = secrets.token_hex(32)
            grant = controller.acquire("grant", "watchdog-owner", credential, context())
            claim = dict(grant["claim"], credential=credential)
            initial = controller.reset("reset", "episode", dict(initial=0, target=.3,
                position_tolerance=.02, velocity_tolerance=.05), context(), claim=claim)["observation"]
            command = action_chunk(initial, [[1.], [1.], [1.]], period_seconds=TIMING.period_seconds)
            before = backend.inspect(context())
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(controller.step, "killed-command", command, context(3), claim=claim)
                until = monotonic() + 3
                while True:
                    moving = backend.inspect(context())
                    if (moving["state"]["phase"] == "executing" and moving["quiescence"]["measured_speed"] > .02):
                        break
                    if monotonic() >= until:
                        raise RuntimeError("Did not observe native motion before controller fault")
                    sleep(.01)
                process = services.handles["controller"]["process"]
                os.kill(process.pid, signal.SIGKILL)
                try:
                    future.result(timeout=5)
                except Exception as error:
                    controller_error = type(error).__name__
                else:
                    raise RuntimeError("Controller unexpectedly completed before the injected fault")
            until = monotonic() + 8
            while True:
                stopped = backend.inspect(context())
                if (stopped["quiescence"]["quiescent"] is True
                        and stopped["ownership"]["phase"] == "recovery_required"):
                    break
                if monotonic() >= until:
                    raise RuntimeError("Surviving actuator did not inhibit output and settle")
                sleep(.02)
            contender = backend.call("acquire", context(), request_id="competing-gateway", owner="new-controller",
                credential=secrets.token_hex(32), harness_revision="reviewed-watchdog-fixture")
            if contender["state"] != "rejected":
                raise RuntimeError("Expired device ownership was handed off without recovery")
            result = dict(scope=SCOPE, qualification=None, model=model, controller_error=controller_error,
                transport_security=dict(controller=control_rpc.security_identity(), actuator=rpc.security_identity()),
                clock_transport=backend.timing_identity(), initial_observation=initial,
                controller_process_killed=process.poll() is not None,
                backend_process_survived=backend_services.handles["backend"]["process"].poll() is None,
                before=before, observed_motion=moving, after=stopped, competing_controller=contender,
                continued_simulation=stopped["quiescence"]["simulation_seconds"] > moving["quiescence"]["simulation_seconds"],
                trial_outcome="uncertain", automatic_takeover=False)
            atomic_json(root / "result.json", result)
            if operator_directory:
                # This is explicitly requested simulation recovery. Persist the
                # uncertain trial report first; never turn it into a task pass.
                result["operator_recovery"] = _recover_fixture(root, operator_directory, backend, spec, controller_expected, result, control_transport)
                atomic_json(root / "recovery-result.json", result["operator_recovery"])
            return result


def _recover_fixture(root, directory, backend, controller_spec, controller_expected, fault, control_transport):
    def recover(kind):
        client = OperatorClient(directory / (kind + ".sock"))
        snapshot = client.call("inspect")
        atomic_json(root / (kind + "-inspection.json"), snapshot)
        request = recovery_request(snapshot, request_id="recover-" + kind,
            reason="Explicit native simulation recovery after injected controller process death",
            evidence=dict(simulation_only=True, fault_report_sha256=digest(fault),
                          report="result.json", inspected_snapshot=snapshot["id"]))
        atomic_json(root / (kind + "-recovery-request.json"), request)
        receipt = client.call("recover", request)
        atomic_json(root / (kind + "-recovery-receipt.json"), receipt)
        return receipt

    backend_receipt = recover("backend")
    # Restart only the confirmed-dead controller; the native device process
    # remains alive and continues advancing throughout both recovery steps.
    with Services(root / "controller-recovery-service", client_factory=control_transport) as services:
        rpc = services.start([controller_spec])["controller"]
        controller = ControllerClient(rpc, expected=controller_expected)
        controller_receipt = recover("controller")
        credential = secrets.token_hex(32)
        grant = controller.acquire("recovered-grant", "new-inspected-session", credential, context())
        if grant["state"] != "completed":
            raise RuntimeError("Recovered controller did not admit a fresh owner")
        claim = dict(grant["claim"], credential=credential)
        initial = controller.reset("recovered-reset", "new-episode", dict(initial=0., target=.3,
            position_tolerance=.02, velocity_tolerance=.05), context(), claim=claim)["observation"]
        command = action_chunk(initial, [[.5]], period_seconds=TIMING.period_seconds)
        action = controller.step("recovered-command", command, context(), claim=claim)
        stop = controller.quiesce("recovered-stop", "new-episode", context(), claim=claim)
        final_backend, final_controller = backend.inspect(context()), controller.ownership()
        if (action["state"] != "completed" or stop.get("quiescent") is not True
                or final_backend["ownership"]["phase"] != "free" or final_controller["phase"] != "free"):
            raise RuntimeError("New simulation session did not complete and release both authorities")
        return dict(scope=SCOPE, qualification=None, original_trial_outcome="uncertain",
                    backend=backend_receipt, controller=controller_receipt,
                    new_session=dict(grant=grant, action=action, stop=stop),
                    final_backend=final_backend, final_controller=final_controller)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--model", choices=("slider", "hinge"), default="slider")
    parser.add_argument("--serve", choices=("backend", "controller"))
    parser.add_argument("--backend-endpoint")
    parser.add_argument("--operator-socket", type=Path, help="Optional private local operator socket for --serve")
    parser.add_argument("--recover-after-fault", action="store_true", help="Explicitly exercise operator recovery in this simulation fixture")
    parser.add_argument("--clock-sync", action="store_true", help="Opt into bounded device-clock mapping")
    parser.add_argument("--clock-offset-seconds", type=float, default=0, help="Simulation-only additive backend clock offset")
    parser.add_argument("--tls-fixture", action="store_true", help="Use temporary simulation-only certificates for both RPC hops")
    parser.add_argument("--tls-server-config", type=Path, help="Explicit ServerTLS JSON configuration for --serve")
    parser.add_argument("--tls-client-config", type=Path, help="Explicit ClientTLS JSON configuration for controller-to-backend RPC")
    args = parser.parse_args()
    if args.serve and args.recover_after_fault:
        parser.error("--recover-after-fault belongs to the fault demonstration, not a service")
    if args.operator_socket and not args.serve:
        parser.error("--operator-socket is a service option; the recovery demonstration creates private sockets")
    if args.serve == "controller" and args.clock_offset_seconds:
        parser.error("--clock-offset-seconds belongs to the simulated backend")
    if args.serve and args.tls_fixture:
        parser.error("--tls-fixture belongs to the fault demonstration")
    if (args.tls_server_config and not args.serve) or (args.tls_client_config and args.serve != "controller"):
        parser.error("TLS configuration files belong to their respective service roles")
    server_tls = ServerTLS(**read_json(args.tls_server_config)) if args.tls_server_config else None
    client_tls = ClientTLS(**read_json(args.tls_client_config)) if args.tls_client_config else None
    root = Path(args.workspace).resolve()
    if args.serve == "backend":
        ActuationHost(make_backend(root / "backend", args.model, clock_offset_seconds=args.clock_offset_seconds),
                      operator_socket=args.operator_socket, tls=server_tls).serve(
            transport="http", host="127.0.0.1", port=0, parent_watch=True)
    elif args.serve == "controller":
        backend = ActuationClient(controller_transport(args.backend_endpoint, tls=client_tls), expected=read_json(root / "backend-identity.json"),
                                  clock_policy=ClockSyncPolicy() if args.clock_sync else None)
        ControllerHost(make_controller(root / "controller", args.model, backend), operator_socket=args.operator_socket, tls=server_tls).serve(
            transport="http", host="127.0.0.1", port=0, parent_watch=True)
    else:
        print(json.dumps(run(root, model=args.model, recover_after_fault=args.recover_after_fault,
                             clock_offset_seconds=args.clock_offset_seconds, clock_sync=args.clock_sync, tls_fixture=args.tls_fixture), indent=2))


if __name__ == "__main__":
    main()
