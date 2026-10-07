# Local operator inspection and recovery

`./physicalrsi operator` provides an explicit recovery workflow for a running `ActuationHost` or `ControllerHost`. It saves an inspection, prepares a reviewable request, submits that exact request and retrieves durable status after a lost response. Each request binds the inspected implementation, authority ID, resources and ownership generation. Recovery releases control only after the existing backend or driver independently confirms quiescence.

This is a Linux local operator interface. It uses a private pathname Unix socket and the kernel's `SO_PEERCRED` identity at both ends; see the primary [Linux Unix socket documentation](https://man7.org/linux/man-pages/man7/unix.7.html). The server accepts only its own effective UID. The socket directory must be owned by that UID with no group/other access; the socket is mode 0600. Paths must be absolute without symlink traversal. Client and server must share a Linux boot and time namespace so absolute monotonic deadlines can include connection, framing and queue time. JSON frames are bounded to 64 KiB for requests and 1 MiB for responses. This endpoint does not deserialize Python objects.

The OS identity is the authorization boundary. An untrusted policy or model worker running as the same UID already has equivalent filesystem/process privileges. Deploy it under a different identity with appropriately restricted device and filesystem access. This module does not create that deployment isolation or authenticate a human independently of the local account.

## Enable an endpoint

The interface is disabled by default. The device-host application opts in when creating its host:

```python
from PhysicalRSI_core.infra.actuation_rpc import ActuationHost
from PhysicalRSI_core.infra.controller_rpc import ControllerHost

# Each path's parent must be a private directory owned by the service UID.
device_host = ActuationHost(guard, operator_socket="/tmp/physicalrsi-operator/backend.sock")
controller_host = ControllerHost(gateway, operator_socket="/tmp/physicalrsi-operator/controller.sock")
```

The owning application calls each host's existing `serve()` method in its respective process. The `PhysicalRSI_demos.mujoco_watchdog` reference service supports `--operator-socket` with either `--serve backend` or `--serve controller`. The application still supplies a backend, workspace and ordinary controller connection. The complete simulation command below manages these services and private socket paths. No real robot driver is supplied by these examples.

The optional listener is separate from ordinary control RPC. `recover` remains absent from `control.execute` and `actuation.call`. The public operator command is also outside the conversation's slash commands and model tools. A controller recovery first quarantines ownership using its metadata lock, then queues the actual driver stop on the same thread that owns reset, sensing and execution. Cooperative drivers can observe revocation immediately. An expired queued stop is discarded rather than executed late.

## Inspect, prepare and execute

These commands assume the host application has already enabled the specified private socket. Use fresh artifact paths; the CLI refuses existing output files before contacting the device.

```bash
operator_socket=/tmp/physicalrsi-operator/backend.sock
artifacts="$(mktemp -d /tmp/physicalrsi-operator-review.XXXXXX)"

./physicalrsi operator inspect \
  --socket "$operator_socket" --output "$artifacts/inspection.json"

# Inspect the saved ownership and device state. For this simulation example:
cat > "$artifacts/evidence.json" <<'JSON'
{
  "scope": "native MuJoCo fixture only",
  "inspection": "Reviewed inhibited output and measured settled velocity",
  "operator_decision": "Release the abandoned simulation owner"
}
JSON

./physicalrsi operator prepare \
  --inspection "$artifacts/inspection.json" \
  --request-id inspected-recovery-001 \
  --reason 'Recover the inspected abandoned simulation owner' \
  --evidence "$artifacts/evidence.json" \
  --output "$artifacts/request.json"

cat "$artifacts/request.json"

./physicalrsi operator recover \
  --socket "$operator_socket" --request "$artifacts/request.json" \
  --output "$artifacts/receipt.json" --timeout 10
```

`inspect` saves a server-side snapshot and returns its exact contents. An actuator snapshot includes measured quiescence; a controller snapshot includes its durable state and ownership, without taking its occupied driver thread. A controller state snapshot alone is not stop evidence. `prepare` only creates a local JSON request. Its evidence should describe the actual inspection; the sample text above is not evidence for a real robot. `recover` fences the exact generation, stops through the existing implementation and measures quiescence again before release. An old inspection cannot revoke a newer owner, and a saved completed receipt is returned without another stop.

The server derives `operator: unix-uid:<uid>` from kernel credentials. The first request's PID, UID and GID are retained in the receipt and core intent, together with the request and inspection digests. A later invocation from a new CLI process uses those original bound inputs when resolving a completed operation; a caller-supplied operator label cannot replace them.

## Resolve an interrupted response

An RPC timeout is not a cancellation or proof that nothing happened. Query the original request ID:

```bash
./physicalrsi operator status \
  --socket "$operator_socket" --request-id inspected-recovery-001 \
  --output "$artifacts/status.json"
```

| Observed state | Meaning and next step |
| --- | --- |
| `missing` | No saved operator request under this ID; confirm the target and submitted ID before explicitly delivering the original file |
| `completed` | The exact recovery result is saved; inspect current device ownership separately before starting a new session |
| `started`, core `missing` | The outer intent exists but no core device effect was dispatched; the same immutable request can be explicitly delivered again |
| `started`, core `completed` | The core result survived but the outer checkpoint did not; delivery of the same immutable request resolves the cached result without another stop |
| `started`, core `started` or `uncertain` | A device effect may have occurred; automatic re-execution is blocked and device inspection is required |

Status reads the durable core journal without waiting for a blocked device mutex. The client does not retry automatically. One saved inspection authorizes one request ID and body, so renaming an unknown request does not reuse the same inspection as authorization. If a failed attempt inhibited output while motion was still settling, inspect the device again and explicitly prepare a new request when justified. The original attempt remains uncertain. A late listener error cannot overwrite an already completed receipt.

The service saves `operator/inspections/`, `operator/inspection-use/` and `operator/requests/` under its existing controller or actuator workspace. Core intents and request receipts remain in their existing locations; actuator release also retains the exact measurement in `quiescence/`. Do not edit or delete these records to regain control.

## Recover the correct layers

| Layer | Recovery responsibility |
| --- | --- |
| Actuation backend | Fence its own generation, inhibit output, measure settling and release its claim |
| Controller gateway | Revoke its own generation, confirm driver quiescence and release its claim; restart only a confirmed-dead gateway against its existing journal |
| Experiment `DeviceRegistry` | Reconcile the interrupted experiment's separate resource claim with actual stop evidence using the existing [device recovery API](embodied-runtime.md#device-ownership-and-recovery) |
| Experiment and System 2 | Preserve the original uncertain trial and exclude it from definite evaluation evidence |

For the supplied two-process adapter after controller death, recover the surviving backend before recovering the restarted controller. The CLI addresses one endpoint at a time; it does not coordinate an atomic multi-device release or reconcile the experiment registry automatically. System 1 requests bounded runtime control for its frozen revision. System 2 proposes and evaluates later revisions. Operator recovery restores inspected availability and does not select a candidate, continue an unknown action or reclassify a trial.

## Native recovery evidence

With the optional MuJoCo dependency installed:

```bash
python -m PhysicalRSI_demos.mujoco_watchdog \
  --workspace "$(mktemp -d /tmp/physicalrsi-recovery-study.XXXXXX)" \
  --model slider --recover-after-fault

python -m pytest -q tests/test_operator.py tests/test_mujoco_watchdog.py
```

The explicit recovery flag first runs the controller-death fault study. The backend remains alive, inhibits effort and measures settling. The demonstration then saves inspection/request/receipt artifacts, recovers the backend, restarts the confirmed-dead gateway against its existing state and recovers that gateway. A new generation executes one bounded pulse and releases both authorities after measured quiescence. This automated, explicitly requested simulation fixture exercises the protocol; it does not stand in for a human physical inspection.

`result.json` retains the original uncertain fault report. `recovery-result.json` separately records both recovery receipts and the new session. The old controller request remains `started`, the original trial remains `uncertain`, and `qualification` remains `null`. The software tests also cover kernel credential rejection, altered inspections, stale generations, dropped responses, interrupted checkpoints, bounded framing and main-thread stop dispatch.

A successful software recovery is not physical qualification. This reference cannot interrupt an indefinitely blocked SDK operation or stop hardware after the backend process or host dies. Robot-specific stop modes, independent hardware watchdogs, sensor validity, SDK exclusivity and deployment isolation still require separate evidence.
