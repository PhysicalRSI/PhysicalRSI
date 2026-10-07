"""Expose the bounded controller gateway through existing RPC transports.

Only gateway methods are exposed; no raw driver step/reset bypass exists here.
The receiver stamps queue arrival in its own clock before main-thread dispatch.
"""

from copy import deepcopy
import ipaddress
from time import monotonic
from urllib.parse import urlsplit

from ..contracts import Context, Contract
from ..embodiment import Embodiment
from ..timing import ControlTiming, finite_seconds
from .rpc import ServiceHost
from .rpc.deadline_http import DeadlineHttpRpcClient
from .rpc.authenticated_http import ControlTransport
from .rpc.main_thread_serve import FixedThreadDispatch


def controller_transport(endpoint, *, enable_sessions=False, tls=None):
    """Direct literal-IP HTTP(S), optional mutual TLS and a total call deadline."""
    if enable_sessions:
        raise ValueError("Controller episodes own state; RPC model sessions are not used")
    url = urlsplit(endpoint)
    if tls is None and not ipaddress.ip_address(url.hostname).is_loopback:
        raise ValueError("Non-loopback controller connections require mutual TLS")
    return DeadlineHttpRpcClient(endpoint, tls=tls)


class ControllerHost(FixedThreadDispatch, ControlTransport, ServiceHost):
    def __init__(self, gateway, *, operator_socket=None, tls=None):
        super().__init__(metadata=gateway.identity())
        self.gateway = gateway
        self.configure_transport(tls, gateway.root / "transport")
        self.operator_socket, self._operator = operator_socket, None
        self._rpc.update({"control.describe": gateway.identity, "control.execute": self.execute})
        self._readonly_methods.add("control.describe")

    def _builtin_dispatch(self, method, args, kwargs):
        # Status remains observable while a driver is blocked on the main thread.
        if method == "control.status":
            return self.gateway.status(kwargs["request_id"])
        if method == "control.state":
            return self.gateway.state()
        if method == "control.ownership":
            return self.gateway.ownership()
        if method == "control.execute" and kwargs.get("operation") in {"acquire", "renew", "revoke"}:
            # Ownership metadata has its own lock/transaction. In particular,
            # revocation must remain reachable while the driver is running.
            return self.execute(*args, **kwargs)
        return super()._builtin_dispatch(method, args, kwargs)

    def _dispatch_authenticated(self, dispatch, method, args, kwargs, *, session_id, received_at):
        return dispatch(method, args, kwargs, session_id=session_id, _transport_received_at=received_at)

    def _dispatch_main_thread(self, method, args, kwargs, *, session_id=None, _transport_received_at=None):
        if method == "control.execute":
            kwargs = dict(kwargs, _received_at=monotonic() if _transport_received_at is None else _transport_received_at)
        return super()._dispatch_main_thread(method, args, kwargs, session_id=session_id)

    def execute(self, request_id, operation, inputs, budget_seconds, harness_revision="", *, _received_at):
        finite_seconds(budget_seconds)
        if operation not in {"reset", "step", "quiesce", "acquire", "renew", "revoke"}:
            raise ValueError("Unknown controller operation")
        episode = (request_id if operation in {"acquire", "renew", "revoke"} else
                   inputs["command"]["observation"]["episode"] if operation == "step" else inputs["episode"])
        context = Context(episode, deadline=_received_at + budget_seconds,
                          harness_revision=harness_revision)
        # A request that expired while queued must not reach reset or execution.
        context.check()
        return getattr(self.gateway, operation)(request_id, context=context, **inputs)

    def serve(self, **kwargs):
        # The mixin creates its queue before this optional listener starts.
        from .operator import OperatorServer, RecoveryTarget
        try:
            if self.operator_socket:
                self._operator = OperatorServer(self.operator_socket, RecoveryTarget(
                    self.gateway, kind="controller", main_thread=self.call_on_main_thread))
            return super().serve(**kwargs)
        finally:
            if self._operator:
                self._operator.close()

    def close(self):
        if self._operator:
            self._operator.close()
        self.gateway.close()


class ControllerClient:
    def __init__(self, rpc, *, expected):
        actual = rpc.call("control.describe", timeout_s=5)
        if actual != expected:
            raise ValueError("Controller service identity mismatch")
        self.rpc, self._identity = rpc, deepcopy(actual)
        spec = dict(actual["embodiment"])
        spec["observation"] = Contract(**spec["observation"])
        spec["action"] = Contract(**spec["action"])
        spec["timing"] = ControlTiming(**spec["timing"])
        self.embodiment = Embodiment(**spec)

    def identity(self):
        return deepcopy(self._identity)

    def state(self):
        return self.rpc.call("control.state", timeout_s=5)

    def status(self, request_id):
        return self.rpc.call("control.status", kwargs=dict(request_id=request_id), timeout_s=5)

    def ownership(self):
        return self.rpc.call("control.ownership", timeout_s=5)

    def _execute(self, request_id, operation, inputs, context):
        context.check()
        if context.deadline is None:
            raise ValueError("Remote controller calls require a bounded context deadline")
        remaining = context.deadline - monotonic()
        finite_seconds(remaining)
        try:
            return self.rpc.call("control.execute", kwargs=dict(
                request_id=request_id, operation=operation, inputs=inputs,
                budget_seconds=remaining, harness_revision=context.harness_revision), timeout_s=remaining)
        except Exception as error:
            error.request_id = request_id
            raise

    def acquire(self, request_id, owner, credential, context):
        return self._execute(request_id, "acquire", dict(owner=owner, credential=credential), context)

    def renew(self, request_id, claim, context):
        return self._execute(request_id, "renew", dict(claim=claim), context)

    def revoke(self, request_id, claim, reason, context):
        return self._execute(request_id, "revoke", dict(claim=claim, reason=reason), context)

    def reset(self, request_id, episode, case, context, *, claim=None):
        return self._execute(request_id, "reset", dict(episode=episode, case=case, claim=claim), context)

    def step(self, request_id, command, context, *, claim=None):
        return self._execute(request_id, "step", dict(command=command, claim=claim), context)

    def quiesce(self, request_id, episode, context, *, claim=None):
        return self._execute(request_id, "quiesce", dict(episode=episode, claim=claim), context)
