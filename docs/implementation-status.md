# Implementation status

This preview now includes a complete local software Self-Harness interaction in `PhysicalRSI_demos/`. The interaction is intentionally deterministic so it can verify the control flow without external model weights or a simulator.

| Area | Preview status | Evidence |
| --- | --- | --- |
| Memory snapshot and typed read | Implemented | `PhysicalRSI/Embodied_Harness/memory/` |
| System 1 execution contract | Implemented in the local service demo | `PhysicalRSI_demos/runtime.py` |
| Trajectory with proposal and transition records | Implemented | `PhysicalRSI_core/infra/trajectory.py` and the runtime demo |
| System 2 proposal boundary | Implemented as injected ports and a reusable JSON edit contract | `self_harness/proposals.py`, reviewed enumeration, frozen requests/replies, schema admission and separate materialization |
| Bounded external model proposal transport | Implemented; local HTTP fixture tested | `infra/proposal_model.py`, owned worker, total deadline, reported usage, no tool execution or automatic retry of unknown calls; real model repair quality unverified |
| Parent and child admission | Implemented | `PhysicalRSI_core/self_harness/` |
| Fresh paired evaluation and survivor selection | Implemented for the local counter contract | `PhysicalRSI_demos/runtime_evolution.py` |
| Persistent lineage and restart-safe receipts | Implemented | `PhysicalRSI_core/lineage/` and the demo workspace |
| Declared embodiment / System 1 compatibility | Implemented for experiment adapters; CPU-tested | `embodiment.py`, exact units/frame/embodiment matching, per-episode policy initialization |
| Durable device ownership across local processes | Implemented; tested with a killed subprocess | `infra/devices.py`, explicit recovery evidence and active-owner exclusion |
| Experiment receipts in System 2 comparisons | Implemented; CPU-tested | `self_harness/experiments.py`, paired execution/scoring protocol checks |
| Controller timing and observation-bound action chunks | Implemented; fault-tested | `timing.py`, `infra/controller.py`, stale/late/replayed commands and interrupted driver calls |
| Bounded controller RPC and status during execution | Implemented; tested over localhost subprocesses | `infra/controller_rpc.py`, expired queued reset and lost response recovery |
| Gateway-enforced control ownership | Implemented; RPC contention and crash tests | `infra/control_authority.py`, persistent generations, expiry, live revocation, verified release and local operator recovery |
| Independent actuator authority and software watchdog | Implemented; deterministic faults and native process-death tests | `infra/actuation.py`, `infra/actuation_rpc.py`, shared device generations, bounded commands, measured settling, no automatic takeover; hardware watchdog unverified |
| Local operator inspection and recovery | Implemented; Linux UID authentication, fault and native recovery tests | `infra/operator.py`, public `./physicalrsi operator` CLI, saved inspection/generation binding, durable status and driver-thread recovery; deployment isolation remains external |
| Controller-to-device clock conversion | Implemented as an explicit bounded policy; software faults and native offset tests | `infra/clock_sync.py`, conservative deadlines/capture intervals, immutable timing evidence, drift/age limits; actual multi-host and physical clock bounds unverified |
| Authenticated controller and actuator RPC | Implemented as optional mutual TLS; certificate, permission and transport fault tests | `infra/rpc/tls.py`, `infra/rpc/authenticated_http.py`, exact method grants, audit before dispatch, bounded handshakes/requests, public security identities; credential deployment and physical networks unqualified |
| Isolated candidate programs and policy teardown | Implemented as optional stdlib-only Linux execution; OS boundary, lifecycle and native campaign tests | `infra/isolated_policy.py`, `infra/isolated_proposal.py`, chroot/seccomp, bounded data exchange, process cleanup evidence, retained unresolved claims; GPU runtimes and physical deployment outside this profile |
| Executable skill artifacts bound to candidates | Implemented; actual isolated CPU source-edit campaign | `self_harness/code_policy.py`, exclusive skill ownership, candidate and file hashes, no host-side source execution, measured inheritance; model generation quality and hardware qualification unverified |
| Continuously advancing native device process | Implemented as optional MuJoCo fixtures | `PhysicalRSI_demos/mujoco_watchdog.py`, killed controller, surviving actuator, stopped effort, measured velocity, and normal experiment release |
| MuJoCo integration through the shared experiment boundary | Implemented as optional small joint fixtures | `PhysicalRSI_demos/mujoco_control.py`, force/torque contracts, paired scripted controller comparison |
| Experiment-backed Self-Harness evaluator | Implemented; recovery and case-separation tests | `self_harness/evaluation.py`, frozen plans, definite validation outcomes, no replay of unknown trials |
| Durable trial admission budget | Implemented for cooperating local processes | `infra/trial_quota.py`, atomic whole-pool reservations, restart-stable accounting |
| Bounded multi-round System 2 campaigns | Implemented; checkpoint fault tests | `self_harness/campaign.py`, one committed parent, retained-round audit and lineage CAS recovery |
| MuJoCo proposal/evaluation/inheritance loop | Implemented with reviewed, enumerated and isolated strategies; external model option | `PhysicalRSI_demos/mujoco_evolution.py`, reviewed feedback skill, sampled validation, actual lineage and a subsequent round |
| External model-generated code or skill repair | Adapter boundary only | Requires an explicitly configured provider |
| pi05 / pi05-sparse-memory adapter metadata | Implemented | `PhysicalRSI_baselines/robodojo/skills/` |
| pi05 / pi05-sparse-memory training | Provider required | RoboDojo is evaluation-only; the separate Dexjoco demo has SmolVLA and pi05 training entry points, with external dependencies and weights |
| External provider and checkpoint manifest | Implemented | `PhysicalRSI_baselines/robodojo/provider.py` |
| Reviewed primitive-only code-policy demo | Implemented | `PhysicalRSI_baselines/robodojo/code_policy_demo.py` |
| Core skill and exploration-memory catalog demo | Implemented | `PhysicalRSI_demos/skill_memory.py`, `/skill-memory` |
| Native RoboDojo or robot qualification | Not included | Requires external simulator, policies, assets, and environment evidence |

Run the local evidence path with:

```bash
python -m PhysicalRSI_demos.runtime --workspace /tmp/physicalrsi-runtime
python -m PhysicalRSI_demos.runtime_evolution --workspace /tmp/physicalrsi-evolution
```

The `/evolve` CLI command invokes the same Self-Harness demo. A successful local run proves software contracts, evidence binding, and lineage behavior; it does not prove physical-task success or benchmark performance.

The public release also includes the versioned experiment runtime, a workspace-bound media server, four bundled recordings, a copied piano skill library, and terminal/conversation commands for Dexjoco layouts, collection, training and cycle control. Simulator and training execution require configured external installations.

The [System 2 campaign guide](system2-campaigns.md) documents the shared evaluator, trial budget, recovery behavior and native MuJoCo inheritance example. The [structured proposal guide](system2-proposals.md) adds a common JSON edit contract for finite search and external model calls. Native tests enable an existing reviewed feedback skill; local model-API fixtures prove protocol integration only. These checks do not establish real model repair quality, open-ended invention, model training or physical qualification.

The [embodied runtime guide](embodied-runtime.md) records the OpenRSI infrastructure review, current execution boundary, adapter responsibilities and remaining work toward multiple robot embodiments. The [controller gateway](controller-gateway.md) adds source timing, durable commands and a headless MuJoCo comparison. [Controller ownership](controller-ownership.md) enforces claims for clients of one gateway; the [actuation backend](actuation-watchdog.md) adds device-side fencing shared by controller processes and a continuously polled software watchdog. The [operator interface](operator-recovery.md) provides explicit local inspection, recovery and durable status using the service's OS identity. [Clock-domain conversion](clock-domains.md) adds explicit uncertainty bounds and timing evidence between controller and actuator. [Control transport security](control-transport-security.md) adds certificate authentication and exact RPC method grants on both hops. [Isolated candidates](isolated-candidates.md) keep supported stdlib-only policy and proposal programs outside the trusted host. Local experiment resource records still use a configured device registry. Actual multi-host clock qualification, device-specific sensor mapping, raw-SDK exclusion, hardware watchdog validation, deployment isolation beyond this candidate profile and real-device qualification remain integration work.
