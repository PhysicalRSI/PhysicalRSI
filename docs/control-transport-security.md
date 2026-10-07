# Authenticated control transports

`ControllerHost` and `ActuationHost` can require mutually authenticated TLS. Each service validates the client certificate against an explicit CA file and exact leaf-certificate allowlist. Clients validate the server's CA chain, IP subject alternative name and pinned leaf fingerprint before sending an RPC body. The minimum protocol is TLS 1.2. The implementation uses Python's standard [`ssl` certificate verification and explicit handshake APIs](https://docs.python.org/3/library/ssl.html); it adds no runtime certificate-generation dependency.

The layers have separate responsibilities:

| Boundary | What it establishes |
| --- | --- |
| TLS certificate and exact method grant | Which configured peer may call an RPC method |
| Control claim | Which generation and frozen System 1 revision may control the device |
| Timing and actuator watchdog | Whether a command remains admissible and when output must be inhibited |
| Local operator channel | Explicit inspection and recovery for a specific saved generation |
| System 2 | Proposals, bounded evaluation, selection and lineage between trials |

A diagnostic owner label is not an authenticated principal. Each TLS principal comes from the server's certificate configuration, never from request fields. Allowing `control.execute` or `actuation.call` grants access to that method's supported operations; current-generation claims and all existing action checks still apply. Neither method exposes operator recovery. Do not give control certificates or raw device access to System 2 proposal workers. TLS does not create process or filesystem isolation for untrusted code sharing the service's UID.

## Configure each control hop

Certificate provisioning is operator owned. Server certificates need an IP subject alternative name matching the literal endpoint address and server authentication usage; client certificates need client authentication usage. Protect keys outside candidate workspaces, logs and evidence. Use separate certificates for the System 1 client, controller-to-actuator client and operator-managed services. CA and leaf files are PEM. Fingerprints are lowercase SHA-256 over DER leaf certificates.

For an already constructed controller gateway, configure an explicit peer:

```python
from PhysicalRSI_core.infra.controller_rpc import ControllerHost, controller_transport
from PhysicalRSI_core.infra.rpc.tls import ClientTLS, PeerGrant, ServerTLS, certificate_fingerprint

server = ServerTLS(
    ca="/run/physicalrsi-keys/ca.pem",
    certificate="/run/physicalrsi-keys/controller.pem",
    private_key="/run/physicalrsi-keys/controller.key",
    peers={
        certificate_fingerprint("/run/physicalrsi-keys/system1.pem"): PeerGrant(
            "system1-runtime",
            ("healthz", "service.describe", "control.describe", "control.state",
             "control.status", "control.ownership", "control.execute"),
        ),
    },
    read_seconds=5,
    request_bytes=1024 * 1024,
    max_connections=32,
)
host = ControllerHost(gateway, tls=server)
host.serve(transport="http", host="127.0.0.1", port=9443)
```

The host announces an `https://` endpoint. `transport="http"` selects the JSON HTTP codec; the supplied TLS policy protects the listener. Configure the client in its own process:

```python
from PhysicalRSI_core.infra.controller_rpc import controller_transport
from PhysicalRSI_core.infra.rpc.tls import ClientTLS, certificate_fingerprint

client = ClientTLS(
    ca="/run/physicalrsi-keys/ca.pem",
    certificate="/run/physicalrsi-keys/system1.pem",
    private_key="/run/physicalrsi-keys/system1.key",
    server_sha256=certificate_fingerprint("/run/physicalrsi-keys/controller.pem"),
)
rpc = controller_transport("https://127.0.0.1:9443", tls=client)
print(rpc.call("control.ownership", timeout_s=2))
```

These snippets assume provisioned files and a gateway; they do not create a robot adapter. Configure `ActuationHost(guard, tls=server)` independently. Its client usually needs `healthz`, `service.describe`, `actuation.describe`, `actuation.inspect`, `actuation.status`, `actuation.clock`, `actuation.intent` and `actuation.call`. A monitoring certificate should receive only the inspection methods it needs. `shutdown` is an additional explicit grant when needed by trusted service administration; it is not a recovery or verified-stop interface. Wildcard grants are rejected. Generic request wrappers are not needed by either control client.

Plain control listeners and clients are limited to loopback; loopback alone does not authenticate another local process. HTTPS requires `ClientTLS`, supplying TLS for a plain HTTP endpoint fails, and no fallback or automatic request retry occurs. TLS hosts reject the pickle/socket transport. Generic model RPC hosts are unchanged. The reference demos still bind localhost; this feature does not publish a listener or modify network access rules.

## Bounds, audit and evidence

The server bounds admitted connections, declared request size, and total handshake/header/body reading time. TLS handshakes run in admitted connection threads so a stalled peer does not block the accept loop. Malformed or duplicate JSON members and non-finite numbers are rejected. Unauthorized methods never enter framework or driver dispatch. An unavailable audit store also prevents dispatch. These bounds do not guarantee service availability against connection floods or a blocked SDK.

The client's total deadline covers connection, TLS, request and response reads. Controller operation budgets start at server connection handling, before TLS and body reading; audit and queue time consume that budget. The System 1-to-controller protocol still uses a receiver-relative budget, which does not measure the preceding network transit. It is not a cross-host absolute-deadline guarantee. Controller-to-actuator calls retain their checked device-domain deadlines and optional [bounded clock conversion](clock-domains.md).

Under each controller or actuator root, `transport/policies/<sha256>.json` stores a public configuration snapshot and `transport/access/<id>.json` records the authenticated principal, client leaf fingerprint, policy fingerprint, method, request ID when present, request-body digest, authorization decision, TLS version and cipher. Audit precedes dispatch; an authorized record alone does not prove execution. Resolve actual completion through the existing controller/actuator receipt and status APIs. The audit omits credential fields, private keys and raw payloads; it is local evidence, not a signed remote attestation. Connections rejected during handshake do not produce method-access records.

`GatewayEnvironment.identity()` includes the public client security configuration for a remote controller. `ActuationDriver.identity()` includes it for the actuator hop. These record CA-file, client-leaf and server-leaf fingerprints and verification/runtime settings. The server policy snapshot remains in transport audit. Rotation uses newly configured contexts and public identities; there is no automatic CA distribution, online revocation, CRL loading, key rotation or permission-management service.

TLS setup adds latency and audit writes use synchronous storage. Deadline expiry still means an unobserved operation may have executed. Read durable status with the same request ID, preserve an uncertain outcome, and use explicit operator recovery where required. Authentication does not turn a lost response into permission to repeat motion.

## Reproduce the evidence

CPU tests generate one-day certificates in temporary private directories using the OpenSSL CLI:

```bash
python -m pytest -q tests/test_control_tls.py tests/test_controller_rpc.py
```

These tests exercise certificate requirements, CA/SAN/fingerprint rejection, method denial including a generic wrapper, audit failure, request/connection bounds, incomplete readers, stalled handshakes, trickled replies, forged receiver timestamps, main-thread driver affinity and durable completion after a lost reply. Missing OpenSSL explicitly skips certificate-dependent checks.

With the existing optional MuJoCo environment:

```bash
python -m PhysicalRSI_demos.mujoco_watchdog \
  --workspace "$(mktemp -d /tmp/physicalrsi-tls-watchdog.XXXXXX)" \
  --model slider --tls-fixture --clock-offset-seconds 120 \
  --clock-sync --recover-after-fault
python -m pytest -q tests/test_mujoco_watchdog.py
```

`--tls-fixture` protects both RPC hops with distinct client/server certificates. It keeps temporary keys outside the evidence workspace and removes them during normal teardown. The fixture kills its own controller, observes the surviving native actuator inhibit effort and settle, then exercises explicitly requested local recovery and a new control generation. `result.json` preserves the original `uncertain` trial and `qualification: null`; `recovery-result.json` records the separate recovery. Public certificate fingerprints and authorization records remain inspectable after test credentials are removed.

This is localhost simulation evidence with an injected clock offset. It does not qualify a physical robot, real multi-host clock drift, network availability, sensor accuracy, emergency stops or a hardware watchdog. Those require measured evidence from the deployed device and independently enforced authority at its actual actuator.
