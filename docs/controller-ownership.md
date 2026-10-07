# Controller ownership and recovery

`ControllerGateway` is the authority for control through its endpoint. Each reset, action and quiescence request carries a claim for its current ownership generation. Two clients with separate local filesystems cannot simultaneously control the same gateway. Ownership is bound to one System 1 revision and to all resources declared by that gateway.

This extends the central dispatch and worker recovery boundary reviewed in [OpenRSI's sandbox](https://github.com/FrontisAI/OpenRSI/blob/71ae803a035d5e3b78c19fa49ed9f67d0550cbaa/OpenMLE-Gym/openmle-sandbox/README.md). The distinction between command ownership and stopping a robot also appears in the primary [Spot lease documentation](https://dev.bostondynamics.com/docs/concepts/lease_service.html) and [E-Stop documentation](https://dev.bostondynamics.com/docs/concepts/estop_service.html). Those are design references, not installed robot drivers or copied implementations. physicalRSI requires verified quiescence before handing off even an expired claim.

## Claims and command admission

The gateway keeps a persistent authority ID and an increasing generation. A client supplies a fresh 32-byte random credential when acquiring control and retains it in memory. The grant returns the authority ID, generation and lease duration. A claim contains those identifiers and the credential; the diagnostic owner label is not an authentication identity.

| Field or record | Purpose |
| --- | --- |
| Persistent authority ID | Distinguish separate gateways, including ones whose counters happen to match |
| Generation | Distinguish consecutive owners of the same gateway |
| Credential | Prove possession of this particular claim |
| System 1 revision | Prevent a claim from silently changing the executing candidate |
| Gateway boot ID | Invalidate active commands and renewal after a server restart |
| Gateway monotonic expiry | Reject expired claims without comparing client/server clocks |
| Independent quiescence result | Permit release only after the driver confirms the stop |

`authority.json` stores ownership, credential digests and metadata operation receipts in one atomic transaction. Plaintext credentials do not appear in that ledger, command intents, public ownership status or experiment evidence. The persistent authority ID and lease policy are included in the environment's executable identity; replacing an authority changes the evaluation conditions.

Controller reset and action calls require a live matching claim before reaching the driver. An action's declared horizon and execution slack must fit within the remaining claim lifetime. The check runs again at dispatch, and the driver receives a `Context` whose `check()` also validates ownership. Missing, forged, expired, revoked, foreign-authority and old-generation claims are rejected. A client that knows the current observation still needs the current claim.

Completed requests remain readable and replayable as saved results. Replaying a completed grant or renewal does not allocate another generation or extend its original lifetime. A saved successful result is evidence about that request; consult `ownership()` for current availability. A new request ID with an old claim cannot issue a new action or stop a later owner's episode.

## Using the boundary

`GatewayEnvironment` manages this lifecycle for `ExperimentRuntime`: it acquires a claim before reset, labels the owner with the experiment ID, renews between decisions when half the advertised lease duration has elapsed, and releases only after quiescence. Renewal scheduling uses a duration measured from the client's request start; server monotonic timestamps are never subtracted from client timestamps. There is no background thread keeping an abandoned trial alive. Long planning intervals need a suitably declared lease duration or an explicit owner keepalive strategy, as well as the existing observation freshness requirements.

Lower-level clients use the same methods on a local `ControllerGateway` or remote `ControllerClient`:

```python
import secrets
import uuid

def acquire_controller(client, context):
    credential = secrets.token_hex(32)
    grant = client.acquire(uuid.uuid4().hex, context.episode, credential, context)
    if grant["state"] != "completed":
        raise RuntimeError(grant["reason"])
    # Retain this only in the controlling process, not in a public artifact.
    return dict(grant["claim"], credential=credential)

def reset_owned(client, claim, episode, case, context):
    return client.reset(uuid.uuid4().hex, episode, case, context, claim=claim)

def stop_owned(client, claim, episode, context):
    return client.quiesce(uuid.uuid4().hex, episode, context, claim=claim)
```

Remote contexts require a deadline. `client.renew(request_id, claim, context)` renews a still-live claim. `client.revoke(request_id, claim, reason, context)` rejects subsequent action admission and sets recovery-required ownership; it explicitly returns `quiescent: false`. An already admitted operation remains in flight until its next cooperative check or backend stop. Status, ownership inspection, acquisition, renewal and revocation remain reachable while the main driver thread is occupied. Driver effects remain serialized on that thread. Requests queued before a revocation recheck the claim when they reach dispatch.

The transport must be trusted or use the explicit [mutual TLS configuration](control-transport-security.md). A claim is a bearer capability for control coordination; the owner label does not authenticate a person. `ControllerHost` and `ActuationHost` accept certificate allowlists and exact method grants, with non-loopback control listeners requiring TLS. The included demonstrations still bind localhost. Private-key isolation, network availability and certificate lifecycle remain deployment responsibilities.

## Expiry, revocation and process death

Expiry prevents new commands; it does not free the resource. Revocation is likewise an admission change, not a stop acknowledgment. The same holder can still request bounded quiescence with an expired or revoked claim. It can also quiesce and release an unstarted claim if cancellation occurred between acquisition and reset. On a successful stop, the gateway makes the device state idle and commits an ownership release. Only then can another owner receive a higher generation.

After gateway restart, old claims cannot command or renew, even if their old timestamps look recent. They remain available for matching-owner quiescence. A killed process leaves its unfinished command receipt intact. Recovery does not convert that receipt into a completed trial or task failure.

When the controlling process has lost its credential, the host application can use the local operator recovery method:

```python
def recover_owned_gateway(gateway, snapshot, context, *, operator, reason, evidence):
    return gateway.recover(
        request_id=uuid.uuid4().hex,
        generation=snapshot["generation"],
        operator=operator,
        reason=reason,
        evidence=evidence,
        context=context,
    )
```

Take `snapshot = gateway.ownership()` when inspecting the interrupted controller. Recovery requires the exact generation, operator, reason and nonempty inspection evidence. It quarantines that generation first, then serializes the driver's quiescence call and checks its result. A failed stop leaves ownership unavailable. Replaying an old recovery result cannot release or stop a replacement owner. `recover` is intentionally absent from the ordinary RPC command list. The optional [local operator interface](operator-recovery.md) uses kernel peer credentials, saved inspections and explicit CLI request artifacts; its driver stop runs on the gateway's original thread. Direct embedding applications remain responsible for authorization and thread ownership when calling the method themselves.

The process lock covers both driver operations and metadata commits during shutdown. Closing a gateway cannot release the lock while an acquisition or renewal still writes its authority state.

The client-side `DeviceRegistry` remains a separate record of experiment resource ownership. If a trial was interrupted, reconcile that registry with the actual stop evidence as described in the [embodied runtime guide](embodied-runtime.md#device-ownership-and-recovery). Releasing controller ownership does not rewrite an interrupted experiment or its local device claim.

## What this proves, and what still requires a robot

This gateway is one authoritative endpoint serving multiple clients. Separate gateways can additionally use the shared [actuation backend](actuation-watchdog.md), whose own generation gates actual writes and survives a controller process failure. All actuator access must pass through that single backend. These modules do not replicate ownership across hosts, fence a process bypassing the backend through a raw robot SDK, or transfer control automatically. A physical device must enforce its own ownership and stop/watchdog contract. Deleting the journal directory or starting another gateway against the same hardware is not a recovery procedure.

The validity callback is cooperative. Drivers must call `context.check()` between bounded operations; the MuJoCo driver does so during simulation stepping. Python cannot interrupt an indefinitely blocked SDK/native call or guarantee that a robot stops when the gateway process dies. Hardware watchdogs, sensor-clock mapping, physical stop behavior and backend fencing need independent per-robot evidence. A gateway expiry or software test is not physical qualification.

System 2 does not control this lifecycle at every action. It chooses the next frozen System 1 through the [experiment and campaign interfaces](system2-campaigns.md). System 1's runtime acquires and uses a claim for that revision. Controller generation references are retained in reset, action and quiescence evidence, so the selection result can be traced to the actual controlling episode.

## Verification

```bash
python -m pytest -q tests/test_controller.py tests/test_controller_rpc.py
```

These tests exercise real localhost RPC processes, competing clients without shared client-side locks, queued commands after revocation, claim expiry, lost grant responses, a killed controller, recovery failures, stale-owner stops, generation isolation and the shutdown/write race. They also check that plaintext credentials are absent from evidence.

With the optional MuJoCo dependency installed:

```bash
python -m pytest -q tests/test_mujoco_control.py tests/test_mujoco_evolution.py
python -m PhysicalRSI_demos.mujoco_evolution \
  --workspace "$(mktemp -d /tmp/physicalrsi-owned-evolution.XXXXXX)" \
  --rounds 2 --trials 12
```

The native checks verify control claims and confirmed releases for slider and hinge fixtures, paired selection and actual lineage inheritance, and no additional claims or actions when replaying a completed study. Simulation scope and `qualification: null` remain explicit.
