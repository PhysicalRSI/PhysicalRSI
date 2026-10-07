# Device clock domains and bounded timing evidence

The controller and the actuator may have different monotonic clock origins. `infra.clock_sync` supplies an explicit conversion policy for that boundary. It preserves uncertainty when mapping sensor capture times and uses a conservative device deadline for commands. The existing default still requires a shared Linux boot and time namespace; a different domain requires an explicit `ClockSyncPolicy`.

The design uses four exchange timestamps and growing error bounds, concepts described in [NTPv4's protocol specification](https://www.rfc-editor.org/rfc/rfc5905.html#section-8). This implementation does not set either host's clock or implement NTP. Simulation time also remains distinct from control time: the [ROS 2 clock design](https://design.ros2.org/articles/clock_and_time.html) distinguishes steady time from time sources that can pause or jump. A simulator's `/clock` value or a camera's wall-clock header must not be passed as a monotonic deadline without an explicit, qualified mapping.

## Exchange and assumptions

`ActuationHost` exposes a read-only `actuation.clock` exchange. The controller records local send and receive times; the device returns its receive and send times, its declared clock domain, authority ID and the request's nonce. The client verifies those identifiers before using the sample. Queueing and serialization remain inside the measured round trip.

Let `t1, t4` be the controller timestamps and `t2, t3` the device timestamps. With constant offset and nonnegative communication delays, the feasible device-minus-controller offset is:

```text
t3 - t4 <= offset <= t2 - t1
```

The implementation expands that interval for declared timestamp error, floating-point rounding and relative clock drift during the exchange and subsequent use. It does not assume equal outbound and return latency or use a midpoint as an exact offset. A device deadline uses the lower offset bound at the requested controller deadline, so conversion can shorten the execution window and cannot extend it under the declared assumptions.

| Policy field | Meaning |
| --- | --- |
| `max_round_trip_seconds` | Maximum admitted duration of one clock exchange |
| `max_offset_width_seconds` | Maximum full width of the offset interval, including accumulated uncertainty |
| `max_sample_age_seconds` | Maximum age of the exchange when used |
| `max_relative_drift_ppm` | Declared bound on change in device-minus-controller offset per controller second |
| `timestamp_error_seconds` | Declared timestamp error allowance, in addition to arithmetic rounding |

These are explicit integration assumptions, not oscillator properties proved by sending a probe. A robot deployment must establish clock continuity, rate error and timestamp accuracy for its actual hosts and sensors, including suspend/resume behavior. Clock domain identities must change when their origin or continuity changes. The standard local identity includes the Linux boot ID and time namespace. Application-supplied clocks must supply an appropriate domain.

An exchange with excessive latency, impossible ordering or excessive uncertainty is rejected before device effects. An expired mapping requires a new read-only probe. A reply for another authority/domain/nonce or a refreshed mapping contradicting the previous drift envelope latches a session fault. New actions then require reconciliation; the client does not silently discard the contradiction and accept the new clock. Device request status and inspection remain readable after a remote mapping fault, using the caller's still-valid local timeout. A failed local clock cannot supply reliable deadline arithmetic; use a fresh trusted client or the device's local operator endpoint for inspection.

## Opt in at the controller adapter

The device and its pinned identity are supplied by the existing service integration. Configure the timing assumptions explicitly on its client:

```python
from PhysicalRSI_core.infra.actuation_rpc import ActuationClient
from PhysicalRSI_core.infra.clock_sync import ClockSyncPolicy

client = ActuationClient(
    rpc,
    expected=pinned_backend_identity,
    clock_policy=ClockSyncPolicy(
        max_round_trip_seconds=0.2,
        max_offset_width_seconds=0.05,
        max_sample_age_seconds=5,
        max_relative_drift_ppm=100,
        timestamp_error_seconds=0.000001,
    ),
)
```

These example values are the software fixture defaults, not recommended robot calibration values. Calls obtain a mapping when needed within the existing `Context` deadline. `client.synchronize(context, force=True)` explicitly probes again without actuating. Mapping expiration may cause another read-only probe; actuator operations are not automatically retried.

`ActuationDriver.identity()` includes the conversion policy, source implementation and clock domains. Those conditions therefore join the frozen environment identity used by experiments and paired System 2 evaluation. System 1 continues to consume controller-domain observations and produce observation-bound action chunks. System 2 changes candidates between trials; it cannot silently loosen clock uncertainty or observation-age limits to improve a candidate's score.

Clock exchange can use the explicit [mutual TLS transport](control-transport-security.md); the clock nonce itself only checks exchange correlation. Without TLS the caller relies on trusted loopback transport. The supplied examples bind localhost, and deployment still owns network and credential isolation. No host clock adjustment, distributed lock service or hardware watchdog is enabled by clock conversion.

## Source capture and freshness

`SensorSample` now carries an optional bounded capture interval:

```text
[captured_at, captured_at + capture_uncertainty_seconds]
```

An exact sample in the local domain has zero uncertainty. An actuator `observe()` must return a newly captured sample whose entire interval lies inside that call on the device. An adapter must not make a stale buffer appear fresh by replacing its source timestamp with a receipt timestamp; the software check assumes those source timestamps are truthful. The backend also checks ownership again after sensing and rejects observation completion after the request deadline.

For a mapped sample, the client converts the full source interval and intersects it with the local send/receive interval for that exact observation call. This intersection is justified by the backend's fresh-capture check. It does not apply to a cached or independently streamed sample without another causal contract. The controller rejects a capture interval extending before the required post-action observation or into the future. At action dispatch, freshness uses the earliest possible capture time, including time spent persisting intent.

`timing_evidence` retains the original source timestamp/sequence, capture uncertainty, four exchange timestamps, policy, clock domains, mapping digest and local call interval. The source payload remains unchanged in the observation. The observation reference hashes the whole packet, so actions are bound to these timing conditions as well as the sensor value.

This mapping covers the controller-to-actuator boundary. It does not calibrate a separate System 1 caller's clock to `ControllerHost`; that RPC retains its existing receiver-side queue budget. A camera, encoder or vendor controller with its own clock still needs a device-side adapter that produces a valid device-domain `SensorSample`, including source timestamp uncertainty.

## Commands and durable replay

The transport converts both the request deadline and the bounded command deadline into the device domain. The device checks the named clock domain before dispatch, then applies its existing deadline, lease, action-horizon and independent watchdog checks. Network transit does not restart the command budget at receipt. Action periods remain declared nominal seconds in the device scheduler; this Python transport does not implement a hard real-time servo loop.

The first submission saves the converted deadline and its mapping evidence in the device intent. The logical request digest binds the original inputs and controller clock domain. A subsequent submission with the same ID reads that intent and reuses the original converted deadline, even after recalibration or client restart. Changing its actions, source deadline, revision or claim is rejected. A completed core request returns its saved acknowledgment; an unknown effect remains blocked. Query command status to distinguish submission acknowledgment from actual completion.

After command completion or interruption, the device command receipt retains its deadline and timing evidence. `ActuationDriver` carries that receipt into the gateway's `driver.actuation` result; experiment trajectories retain the result alongside independent observations and verification. `control.deadline` belongs to the controller clock, while `driver.actuation.deadline` belongs to the device clock. Compare them using the saved mapping, never as raw timestamps from a shared origin.

Clock conversion does not renew control ownership, authorize automatic takeover, change a stop mode or reclassify an uncertain trial. The [local operator recovery procedure](operator-recovery.md) remains available at the device, independently of a remote client's mapping.

## Reproduce the evidence

Software checks cover asymmetric delays, both drift directions, stale calibration, impossible/replayed clock replies, source timestamp errors, requests delayed in transit, immutable mapped-command replay and conservative observation freshness:

```bash
python -m pytest -q tests/test_clock_sync.py tests/test_controller.py
```

With the optional MuJoCo dependency installed, run a native fault and recovery study using a deliberately different device-clock origin:

```bash
python -m PhysicalRSI_demos.mujoco_watchdog \
  --workspace "$(mktemp -d /tmp/physicalrsi-clock-study.XXXXXX)" \
  --model slider --clock-offset-seconds 120 --clock-sync \
  --recover-after-fault

python -m pytest -q tests/test_mujoco_watchdog.py
```

The fixture offsets its device clock and source capture times by 120 seconds. Native dynamics continue in a separate process. The controller's initial observation records mapped capture bounds, the injected controller death retains an uncertain trial, and explicit recovery creates a new generation whose command carries the converted deadline receipt. Both authorities are released after measured settling. A second native experiment test uses a negative 120-second offset through `ExperimentRuntime`, `DeviceRegistry` and the independent task verifier.

These processes run on one host over localhost. The tests prove behavior with different clock origins and injected timing faults; they do not prove actual multi-host oscillator bounds, network availability, synchronized cameras, real robot behavior or physical qualification. Reports retain simulation scope and `qualification: null`.
