"""Direct HTTP RPC with total wall-clock and response-size limits.

Literal-IP endpoints avoid unbounded DNS resolution. A watchdog interrupts the
socket even when a peer keeps trickling headers or body bytes. A timeout does
not cancel remote actions: use invoke/status and retain uncertain outcomes.
"""

import http.client
import ipaddress
import json
import math
import socket
import time
from urllib.parse import urlsplit

from .http_rpc import _NumpyEncoder, _from_json
from .rpc_client import RpcClient, RpcError, check_response
from .authenticated_http import SocketDeadline
from .tls import ClientTLS, fingerprint


class DeadlineHttpRpcClient(RpcClient):
    def __init__(self, endpoint, *, response_bytes=1024 * 1024, tls=None):
        super().__init__()
        url = urlsplit(endpoint)
        if (
            url.scheme not in {"http", "https"}
            or url.username
            or url.password
            or url.query
            or url.fragment
            or url.path not in {"", "/"}
        ):
            raise ValueError(
                "Expected a direct HTTP(S) endpoint without credentials or path"
            )
        ipaddress.ip_address(url.hostname)
        if type(response_bytes) is not int or response_bytes <= 0:
            raise ValueError("Positive response size limit required")
        if (url.scheme == "https") != (tls is not None) or (tls is not None and not isinstance(tls, ClientTLS)):
            raise ValueError("HTTPS requires explicit mutual TLS; plaintext downgrade is forbidden")
        self.host, self.port = url.hostname, url.port or (443 if tls else 80)
        self.tls = tls
        self.response_bytes = response_bytes

    def security_identity(self):
        return self.tls.identity() if self.tls else None

    def call(self, method, args=(), kwargs=None, *, timeout_s=None):
        if timeout_s is None or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("A positive finite RPC deadline is required")
        deadline = time.monotonic() + timeout_s
        body = json.dumps(
            dict(method=method, args=list(args), kwargs=kwargs or {}),
            cls=_NumpyEncoder,
            allow_nan=False,
        ).encode()
        connection = http.client.HTTPConnection(self.host, self.port, timeout=timeout_s)
        watchdog = None
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("RPC deadline reached before connection")
            family = socket.AF_INET6 if ipaddress.ip_address(self.host).version == 6 else socket.AF_INET
            connection.sock = socket.socket(family, socket.SOCK_STREAM)
            connection.sock.settimeout(remaining)
            watchdog = SocketDeadline(connection.sock, remaining)
            connection.sock.connect((self.host, self.port))
            if self.tls:
                connection.sock = watchdog.wrap(self.tls.context, server_hostname=self.host)
                connection.sock.settimeout(max(.001, deadline - time.monotonic()))
                connection.sock.do_handshake()
                if fingerprint(connection.sock.getpeercert(binary_form=True)) != self.tls.server_sha256:
                    raise ValueError("Server certificate differs from the pinned identity")
            if watchdog.expired or time.monotonic() >= deadline:
                raise TimeoutError("RPC deadline reached during connection or TLS handshake")
            connection.request(
                "POST", "/call", body=body, headers={"Content-Type": "application/json"}
            )
            with connection.getresponse() as response:
                raw = response.read(self.response_bytes + 1)
                # Deadline shutdown may yield a partial body rather than a
                # socket error. Classify it before decoding truncated JSON.
                if watchdog.expired or time.monotonic() >= deadline:
                    raise TimeoutError("RPC total deadline reached")
                if len(raw) > self.response_bytes:
                    raise ValueError("RPC response exceeds size limit")
                if response.status != 200:
                    raise RpcError(method, "HTTP status " + str(response.status))
            result = check_response(_from_json(json.loads(raw)), method)
            if watchdog.expired or time.monotonic() >= deadline:
                raise TimeoutError("RPC total deadline reached")
            return result
        except (OSError, http.client.HTTPException) as error:
            if (watchdog is not None and watchdog.expired) or time.monotonic() >= deadline:
                raise TimeoutError("RPC total deadline reached") from error
            raise RpcError(method, str(error)) from error
        finally:
            if watchdog is not None:
                watchdog.cancel()
            connection.close()
