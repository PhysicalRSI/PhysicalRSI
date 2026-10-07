# Embodied experiment runtime

The core execution boundary makes an experiment reusable across explicitly declared environments and System 1 implementations. A System 2 evaluator can run a frozen candidate through that boundary and pass the resulting evidence to the existing Self-Harness. The current checks are software checks; no robot driver or physical qualification is supplied by these interfaces.

## What we learned from OpenRSI

This review used [OpenRSI revision `71ae803`](https://github.com/FrontisAI/OpenRSI/tree/71ae803a035d5e3b78c19fa49ed9f67d0550cbaa) on 2026-10-05. The implementation below was written for physicalRSI; upstream runtime code was not copied.

| Upstream mechanism | PhysicalRSI design consequence |
| --- | --- |
| Benchmark adapters share a search runtime and its persistence machinery. [OpenMLE-Evo overview](https://github.com/FrontisAI/OpenRSI/blob/71ae803a035d5e3b78c19fa49ed9f67d0550cbaa/OpenMLE-Evo/README.md) | Robot/task adapters should share experiment execution, evidence and selection infrastructure. |
| Asynchronous attempts retain their own identities and commit through one writer; strict resume preserves committed work. [Execution and resume guide](https://github.com/FrontisAI/OpenRSI/blob/71ae803a035d5e3b78c19fa49ed9f67d0550cbaa/OpenMLE-Evo/docs/usage.md) | Bind every physical trial to its executing revision and originating comparison. An unknown physical effect needs reconciliation before another attempt. |
| The sandbox separates job lifecycle, worker allocation and independent scoring. [Sandbox architecture](https://github.com/FrontisAI/OpenRSI/blob/71ae803a035d5e3b78c19fa49ed9f67d0550cbaa/OpenMLE-Gym/openmle-sandbox/README.md) | Keep task outcome separate from process completion, device availability and release readiness. |
| Unhealthy workers are quarantined and probed before reuse. [Dispatcher recovery implementation](https://github.com/FrontisAI/OpenRSI/blob/71ae803a035d5e3b78c19fa49ed9f67d0550cbaa/OpenMLE-Gym/openmle-sandbox/node_controller/task_dispatcher/task_dispatcher.py) | Preserve an uncertain device claim across process death. Restoring availability requires evidence that the actual device is quiescent. |

These are design adaptations. In particular, a retryable software job and an interrupted robot action have different recovery requirements. The device registry has no timeout-based ownership takeover.

## System 1 and System 2 are independent choices

| Boundary | Responsibility | Possible implementation |
| --- | --- | --- |
| Environment | Declare contracts and device resources; reset, observe, execute, confirm quiescence | Software fixture, simulator adapter, robot controller gateway |
| System 1 | Initialize episode state; consume observations and frozen skill/tool/memory dependencies; return actions | Classical feedback controller, reviewed code-policy, provider-backed VLA, hierarchical skill router |
| Verifier | Measure the declared task outcome independently of policy assertions | Task-specific state check or independent sensor/annotation pipeline |
| System 2 | Develop evidence, propose candidate revisions, admit, compare, select, commit lineage | Deterministic memory editor, search over skills/prompts/code, provider-backed proposal strategy |

A planning model used during an episode belongs to System 1 even if it reasons slowly. A System 2 method can be deterministic. The two roles may use different models, the same model, or no neural model. System 2 changes the candidate between trials; it does not mutate a running System 1 revision.

System 1 dependencies explicitly distinguish model, skill, tool and memory references. A model reference is provider-owned metadata. A skill route does not imply a local checkpoint. Memory references require content hashes, and new observations intended for inheritance produce a new snapshot. Transient episode history is initialized by `begin_episode()` and is not itself a new inherited memory revision.

The current Self-Harness fixes the foundation component within an iteration. Training a new policy, adapting foundation weights, and improving System 2 itself need their own declared artifacts, compute budgets and independent evaluation protocols. Those capabilities are not established by a successful CPU comparison.

## Adapter contract

`Environment` and `Policy` retain their original minimal interfaces for legacy software adapters. New adapters implement the declarations together:

```python
from PhysicalRSI_core.contracts import Contract
from PhysicalRSI_core.embodiment import Embodiment, System1

observation = Contract("joint-observation-v1", unit="rad", frame="joint", embodiment="arm-model-a")
action = Contract("joint-target-v1", unit="rad", frame="joint", embodiment="arm-model-a")

# Returned by environment.describe(); actual hardware access lives in the adapter.
environment_spec = Embodiment(
    "arm-model-a", "controller-7-calibration-3", "physical",
    observation, action, resources=("cell-1:arm-serial-42", "cell-1:shared-workspace"),
)

# Returned by policy.describe(); use verify_harness(candidate) as the revision
# when a System 2 trial evaluates that frozen candidate.
policy_spec = System1("joint-controller", "controller-implementation-3", observation, action)
```

The runtime rejects unequal contracts before reset. Unit/frame/embodiment conversion must be an explicit, versioned adapter or tool. Payload dimensions, sensor calibration, clock synchronization and physical limits remain the adapter's validation responsibility; equality of declarations does not inspect a sensor or transform an array. Optional `ControlTiming` declarations must also match. The [controller gateway](controller-gateway.md) enforces source-observation freshness, action periods and bounded chunks at the driver boundary; declaring timing alone does not implement those checks in a custom adapter.

The call sequence is:

1. Freeze the public task case, budget, adapter identities, declarations and optional System 2 attribution.
2. Acquire every named resource atomically before reset. A busy device leaves no started trial.
3. Reset and record readiness. An unready reset produces `invalid` without policy execution.
4. Call `policy.begin_episode(task, case, observation, context)` with copies of the frozen inputs. Policies must clear cached chunks/history or initialize them from the declared snapshot here.
5. Call `policy.act(observation, context)` and record the intended action before `environment.step()`. Persist returned observations before checking cancellation. Adapter identity drift blocks the next dispatch.
6. Verify the outcome independently. Environment termination and step-budget exhaustion remain explicit receipt fields; neither asserts task success.
7. For leased environments, call `environment.quiesce(context)` and require `{"quiescent": true, ...}` before finalizing evidence and releasing ownership.

`describe()` and `identity()` must be side-effect-free and stable during a trial. Identity declarations must cover executable configuration and dependencies, not changing episode state. All case data passed to System 1 is public input; keep held-out answers in the verifier. The runtime itself is not an isolation boundary against an untrusted policy. The optional [isolated candidate adapters](isolated-candidates.md) execute stdlib-only System 1 or System 2 source in restricted Linux processes. Policies with owned workers can expose `end_episode(context)`; normal completion then requires a hashed `policy.json` report confirming teardown, separately from device quiescence.

Adapters must bound their own network/device calls. This Python loop and its synchronous evidence writes do not supply hard real-time servo control. The controller gateway validates chunk limits and measures deadlines; the driver owns real-time control and stop/watchdog handling. On an exception the core retains the claim for recovery; it cannot infer that the device stopped.

## Device ownership and recovery

Configure one `DeviceRegistry` directory on local storage for all cooperating clients of a controller host, including clients that write experiment receipts to different workspaces:

```python
from PhysicalRSI_core.experiments import ExperimentRuntime
from PhysicalRSI_core.infra.devices import DeviceRegistry

registry = DeviceRegistry("/tmp/physicalrsi-controller-devices")
runtime = ExperimentRuntime("/tmp/physicalrsi-trials", device_registry=registry)
```

Use canonical names for actual resources, including shared workspaces or fixtures. Every adapter that can access the same resource must use the same registry and key. There is no automatic robot discovery or alias resolution.

SQLite transactions own the full resource set; a per-lease POSIX lock detects a still-running local owner. Independent devices remain usable. Normal completion frees ownership after the verified receipt and quiescence evidence are durable. Exceptions preserve `needs_reconciliation`. Abrupt process death can leave `held` with `owner_active: false`; this is uncertain work, not an idle device. Inspection and recovery are explicit:

```python
occupied = registry.occupied()  # resource name -> lease token
# Select the token belonging to the interrupted experiment, inspect its owner,
# and independently stop/inspect all devices in that lease before reconciliation.
# registry.inspect(token)
# registry.reconcile(
#     token,
#     operator="controller-operator",
#     reason="Controller stopped; all claimed devices independently verified idle",
#     evidence={"inspection_record": "local-inspection-id", "controller_idle": True},
# )
```

`reconcile()` stores the evidence and its digest and refuses a live owner. Repeating the same decision is idempotent; changing it is rejected. Replaying an old recovery cannot release a newer lease. Reconciliation restores resource availability; it does not replay the experiment or turn its interrupted receipt into a completed trial. Run a new ID only after recovery. A process death between receipt completion and lease release also requires device reconciliation even though the completed trial's evidence remains readable.

This registry coordinates local, cooperating processes. The [controller ownership protocol](controller-ownership.md) adds server-enforced claims for independent clients of one gateway, including clients with different local registries. Filesystem leases alone do not supply that protection. The [independent actuation backend](actuation-watchdog.md) now supplies a shared generation check and measured release for controller processes using that backend. Excluding raw-SDK bypasses and enforcing liveness in actual hardware still require per-device integration.

## Feeding trials into Self-Harness

A task-specific evaluator creates `System2Trial` with a versioned System 2 identity, candidate ID and the SHA256 returned by `verify_harness(candidate)`. Validation additionally binds `digest(comparison)` and `digest(cohort)`. The policy's `System1.revision` must match the frozen candidate; wrapper adapters should retain their underlying provider identities in `identity()` and dependencies.

The evaluator invokes `runtime.run(..., system2=trial)` for each public case. In its existing `evaluate(candidate, comparison, cohort, output)` port it can then use:

```python
from PhysicalRSI_core.self_harness.experiments import experiment_result

# run_ids identify this candidate's trials under the comparison evidence root.
# Each case's digest must equal the corresponding cohort layout identity.
def collect_result(runtime, run_ids, candidate, comparison, cohort, output):
    return experiment_result(
        runtime, run_ids,
        candidate_id=candidate["id"], comparison=comparison, cohort=cohort,
        evaluator_revision="task-evaluation-protocol-v1", evidence_root=output,
    )
```

The default score is binary task success. `score_measurement="metric_name"` instead reads a finite scalar from the independent verifier's `measurements`; declare that scoring choice in the evaluator protocol and native score range. Invalid, uncertain, incomplete or repeated trials invalidate the comparison. Changing environment, embodiment, budget, verifier, System 2 identity or scoring choice between paired candidates also invalidates selection. Policies and their dependency revisions are allowed to differ.

This helper returns the existing evaluator-result contract. Self-Harness remains responsible for fresh validation, admission, selection and compare-and-swap inheritance. Trial attribution does not prove that a split is independent or that a proposed candidate's dependency declaration is complete.

## Reproducible software checks

```bash
./physicalrsi --plain --workspace "$(mktemp -d /tmp/physicalrsi-core.XXXXXX)" --command '/experiment counter-v2'
python -m pytest -q tests/test_experiments.py tests/test_devices.py tests/test_embodied_experiments.py tests/test_experiment_selection.py
```

The CLI runs a declared CPU System 1 against a software counter and returns a v2 receipt. Tests cover an actually killed subprocess, exclusion across independent processes and workspaces, whole-resource acquisition, live-owner recovery refusal, stateful episode initialization, candidate/memory attribution, independent outcomes and paired selection. They do not connect to a robot, train a model or establish simulator fidelity.

## Remaining work toward multiple robots

| Work | Required evidence |
| --- | --- |
| Integrate concrete simulator and robot gateways with these ports | Adapter conformance, real reset/quiescence records, bounded calls and independently measured outcomes for each embodiment |
| Expand the implemented controller timing contract to physical drivers | Gateway timing and optional bounded controller/device clock conversion have software and MuJoCo evidence; actual sensor-clock assumptions and device watchdogs still need per-robot validation |
| Coordinate devices across controller hosts | Shared actuator generations, controller-death recovery and mapped clock offsets have local software/MuJoCo evidence; actual multi-host clock qualification, SDK exclusivity, distributed authority and hardware fencing still require integration |
| Schedule long-running System 2 searches | Bounded sequential campaigns, trial reservations and lineage checkpoint recovery are implemented; queue/compute/device time accounting and bounded candidate concurrency remain |
| Broaden System 2 operators and model providers | Explicit changed components, dependency-closed candidates, fixed task/compute comparisons and held-out transfer evidence |
| Qualify physical execution | Per-robot calibration, task coverage, safety/recovery behavior and independently attributable physical trial evidence |

The [MuJoCo control demo](controller-gateway.md#run-the-mujoco-comparison) exercises two distinct actuator contracts through separate controller processes and paired System 2 evaluation. It is a small simulator integration with scripted controllers, not a qualification of a robot or an autonomous proposal/training system.

The [System 2 campaign](system2-campaigns.md) connects those experiments to a deterministic skill proposal, independent sampled validation, actual inheritance and another iteration. The generic evaluator and campaign do not depend on MuJoCo; suites supply their own System 1, embodiment, sampling and scoring adapters.
