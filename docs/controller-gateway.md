# Controller gateway and timed System 1 execution

The gateway connects a frozen System 1 to a bounded controller operation. System 1 reads a timestamped observation, proposes a chunk, and echoes the exact observation reference. The controller checks the reference and its own clock immediately before dispatch. System 2 consumes the resulting experiment evidence through the existing comparison interface.

This extends the shared execution/worker boundary studied in [OpenRSI](https://github.com/FrontisAI/OpenRSI/blob/71ae803a035d5e3b78c19fa49ed9f67d0550cbaa/OpenMLE-Gym/openmle-sandbox/README.md) with physical-control semantics. The distinction between steady hardware time and simulation time is also reflected in the [ROS 2 clock design](https://design.ros2.org/articles/clock_and_time.html). No ROS driver or ROS dependency is included here.

## Timing and observation identity

```python
from PhysicalRSI_core.timing import ControlTiming, action_chunk

timing = ControlTiming(
    period_seconds=0.02,
    max_chunk_steps=5,
    max_observation_age_seconds=0.5,
    execution_slack_seconds=0.1,
)

# Inside a compatible policy's act(observation, context):
def make_command(observation, actions):
    return action_chunk(observation, actions, period_seconds=timing.period_seconds)
```

Both `Embodiment.timing` and `System1.timing` declare the same contract. The gateway enforces the configured period, the maximum chunk length and the source age. `max_chunk_steps * period_seconds` bounds the commanded horizon. The driver receives a completion deadline equal to dispatch time plus the actual chunk duration and execution slack. A returned call beyond that deadline is uncertain, even if it reports completion.

The controller's observation packet contains:

| Field | Meaning |
| --- | --- |
| `episode` | Explicit controller episode, changed on a new reset |
| `clock_domain` | Gateway boot identity; timestamps from a prior process cannot authorize new actions |
| `sequence` | Driver-owned source sequence, increasing after each executed effect |
| `captured_at` | Earliest possible sensor capture time mapped into the gateway monotonic clock |
| `capture_uncertainty_seconds` | Optional width of the capture interval; zero for an exact local sample |
| `timing_evidence` | Optional original capture and bounded clock-conversion evidence |
| `payload` | Copied sensor/task state in the declared observation contract |

`action_chunk()` includes a digest of this entire packet. The gateway compares it to the stored observation rather than trusting a client-supplied time. The driver supplies `SensorSample`; a delayed or cached frame must retain its original capture time. A post-reset or post-action observation must have been captured after that effect returned, and post-action sequences must advance. Future timestamps, stale samples and backwards clocks block further execution.

Client wall clocks and server monotonic clocks are never subtracted from each other. Simulation time is separate payload data. The MuJoCo adapter advances simulation ticks deterministically while measuring RPC and observation age in the controller host's monotonic clock.

The optional [device clock mapping](clock-domains.md) connects a gateway's actuator client to another monotonic domain. It retains a bounded capture interval and uses its earliest time for freshness. The whole interval must follow the preceding effect and end no later than the current controller time. Driver results, including lower-level actuation receipts and mapped deadlines, remain in the returned `driver` evidence and experiment trajectory.

## Driver responsibilities

`ControllerDriver` has six methods: `identity`, `reset`, `observe`, `validate`, `execute` and `quiesce`. The gateway wraps it with `GatewayEnvironment` for `ExperimentRuntime`.

| Method | Contract |
| --- | --- |
| `identity()` | Stable implementation, model/calibration and configuration identity |
| `reset(case, context)` | Return explicit readiness and optional termination |
| `observe(context)` | Return a `SensorSample` using the source sequence and mapped capture clock |
| `validate(actions)` | Check the complete chunk, dimensions and limits without effects |
| `execute(actions, period_seconds=..., deadline=..., context=...)` | Execute bounded control ticks; return actual `executed_steps` and explicit `terminated` |
| `quiesce(context)` | Independently confirm that the device/controller can be released |

The gateway does not interpolate trajectories, clip invalid actions, rescale units, infer a safe stop, or schedule a hard real-time loop in Python. Drivers must bound every hardware call, honor cancellation, and implement the appropriate watchdog/stop behavior. A model that takes longer to plan must use a suitable declared observation-age budget or explicitly obtain a new observation through its task-specific sensing workflow; stale plans are not silently retimed.

Every gateway declares named resources. Cooperating clients on a host use the same `DeviceRegistry` for those resources. The gateway also enforces its own control claims, including persistent authority/generation identity, expiry, revocation and verified release. Independent RPC clients cannot bypass ownership by using different local registries. One local gateway process owns its journal directory at a time. This does not coordinate separate gateways or prevent a driver outside this boundary from accessing the hardware. See [controller ownership](controller-ownership.md) for the protocol and its recovery limits.

## Durable requests, queues and recovery

Every reset, action and quiescence call has an explicit request ID and input binding. Completed requests return the saved result without another effect. Reusing an ID with different inputs is rejected. `intents/<request-id>.json` preserves the public command and implementation binding; `requests/<request-id>.json` records the execution lifecycle and result. Controller state and dispatch intent are durable before the driver call. An exception after dispatch preserves an uncertain request and blocks subsequent actions and resets until explicit quiescence/recovery. Gateway identity includes the RPC, journal and timing implementation hashes so a changed execution stack cannot silently reuse an old trial's identity.

A malformed, expired or observation-mismatched action returns a completed admission receipt with `state: rejected` and `dispatched: false`. The experiment wrapper preserves that receipt and aborts the trial. It conservatively retains device ownership because the experiment already reset/owned a controller episode; a rejection does not stand in for quiescence or a task score.

`ControllerHost` exposes only gateway operations, with no raw `env.step` route. Device calls run on its main thread. Request status, controller state and ownership remain observable while execution is blocked; control-claim acquisition, renewal and revocation use a separate metadata lock. The host stamps the receive time before queueing, so time spent behind another operation counts against the request's budget. A client-supplied receive timestamp is overwritten. Expired queued requests and invalid ownership claims cannot later reset or move the device.

`controller_transport()` uses literal-IP HTTP with total response deadlines and a bounded response size. `Services(..., client_factory=controller_transport)` selects this transport without changing other service groups. The default response bound is 1 MiB; deployments with larger observations must explicitly configure their transport or artifact representation. A timeout does not cancel remote work. Inspect `ControllerClient.status(request_id)` before deciding what happened; the error carries the request ID, and interrupted experiment receipts retain it.

After a gateway restart, saved completed receipts remain readable, but observations and live control claims from the old boot cannot authorize a new action. An unfinished controller episode requires matching-owner `quiesce()` or explicit local operator recovery. Quiescence is not inferred from closing the Python gateway object or losing the RPC connection. The matching device lease must also be reconciled using the [device recovery procedure](embodied-runtime.md#device-ownership-and-recovery).

## Run the MuJoCo comparison

Install the optional simulator dependency in your chosen Python environment:

```bash
pip install -e '.[mujoco]'
python -m PhysicalRSI_demos.mujoco_control \
  --workspace "$(mktemp -d /tmp/physicalrsi-mujoco.XXXXXX)"
```

This uses MuJoCo's native Python bindings and headless `mj_step` dynamics. The [official Python documentation](https://mujoco.readthedocs.io/en/stable/python.html) describes these APIs and why recorded state arrays must be copied. No viewer, GPU, external robot assets, model provider or checkpoint is required.

The run starts two separate local controller processes and evaluates two fixed policies on two cases for each embodiment:

| Fixture | Observation units | Action units | Dynamics |
| --- | --- | --- | --- |
| Slider | Position in meters; velocity in meters/second | Force in newtons | One damped translational joint |
| Hinge | Angle in radians; angular velocity in radians/second | Torque in newton meters | One damped rotational joint |

The zero-effort policy is a deliberately weak control comparison. The feedback policy uses declared proportional/derivative gains. Independent verification reads native simulator position and velocity against the same case targets and tolerances. Both candidates use the same runtime budgets, actuator limits, solver model and verifier.

The workspace contains frozen comparison/cohort files, eight experiment directories, per-controller request journals and `result.json`. The result retains `qualification: null`, simulation scope and `lineage_committed: false`: this example exercises paired evaluation and survivor selection, not an autonomous proposal loop, training pipeline or deployment. It uses small fixed smoke cases and does not establish held-out transfer. Reusing an unchanged completed workspace verifies and resumes the saved receipts; changed code or candidate identities require a new workspace.

The separate [System 2 campaign example](system2-campaigns.md) reuses these same controller processes for development, a reviewed skill proposal, fresh paired validation and actual Self-Harness inheritance. It adds durable trial reservations and a subsequent iteration while retaining simulation-only scope.

The [independent actuation demonstration](actuation-watchdog.md) uses a different execution mode: native dynamics continue in a separate device process while the controller is killed. It verifies bounded-output expiry and measured settling without treating the controller's disappearance as evidence of a stop. Its fixture damping and clock behavior are explicitly declared and differ from this step-driven comparison.

Verification commands:

```bash
python -m pytest -q tests/test_controller.py tests/test_controller_rpc.py
python -m pytest -q tests/test_mujoco_control.py
```

The first command checks expiry, clock/sequence errors, horizon/period/limit rejection, execution uncertainty, restart behavior and actual localhost RPC faults. The second requires MuJoCo and verifies native dynamics, both actuator contracts, paired selection, copied observation evidence and receipt reuse. Without MuJoCo that test module is explicitly skipped; a skipped simulator test is not simulator evidence.
