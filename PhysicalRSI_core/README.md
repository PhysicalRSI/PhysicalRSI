# PhysicalRSI Core

`PhysicalRSI_core` is the task-independent engine underneath the PhysicalRSI
CLI. It owns the execution boundary, durable evidence, Self-Harness campaign
state, skill mutations, population records, and lineage. It does not contain a
robot, simulator, benchmark adapter, or model provider.

## Runtime boundary

System 1 executes a composed skill through an environment adapter. System 2
receives development evidence, proposes a `SkillMutation`, creates executable
candidate bindings, evaluates the candidate against the incumbent, selects a
survivor, and records the next campaign state.

```text
observation → primitive skills → composition → adapter execution
      ↑                                      ↓
      └──── evidence ← verifier ← action ───┘

System 2: propose → admit → compare → select → inherit
```

The engine accepts simple Python ports for proposal, evaluation and selection.
Task adapters provide embodiment-specific observations and actions; the core
never assumes a task name, simulator actor, robot model, or evaluation seed.

## Durable state

Campaign state, reports, memory revisions and lineage are written beneath the
workspace supplied by the caller. A resumed campaign reuses completed stages
and checks that its declared execution context is unchanged. `PopulationRecord`
and `SkillMutation` make primitive and composed skills first-class evolution
units without changing survivor selection or recovery semantics.

Executable identity is carried by the skill registry and binding records. A
provider resolver can reconstruct a callable in a new process; the public API
uses skill IDs and revisions rather than requiring callers to manage storage
identifiers.

## Integration shape

A task adapter should provide:

- observations and actions that the target embodiment can actually expose;
- a verifier that reports outcome evidence separately from policy inputs;
- a proposer that returns a bounded candidate mutation;
- an evaluator that records trajectories and paired comparisons.

## Trial runtime

The runtime records policy verdicts as `success` or `failure`. A policy that
uses its declared budget is recorded as `deadline`, which is a policy failure
for scoring. The budget charges time spent inside policy and environment port
calls, including reset, step and quiescence calls; environment time counts
because a policy can make a simulator or physical interaction slow through its
chosen contacts and actions. Runtime bookkeeping (journal writes, evidence and
blob publication, and `fsync`) is measured separately and does not consume
that budget. The timing receipt's `phase_seconds` reports `policy`,
`environment`, and `bookkeeping` seconds (the latter includes verifier and
runtime work outside those two port categories).

A whole-trial safety ceiling protects the host from an unbounded bookkeeping
stall. It defaults to ten times the declared budget and can be changed with
`ExperimentRuntime(..., safety_ceiling_multiple=...)`; crossing it is an
`infrastructure-fault`, even if the policy/environment allowance was also
exhausted. Other examples include an evidence/filesystem error, a lost robot
link, or a sensor timeout. Infrastructure faults close their journals and
fault evidence before the evaluator retries the same case and pairing slot
(two retries by default). Exhausted retries make the paired case unavailable
for both arms and reduce the cohort's paired sample size; they never become a
policy failure or abort the campaign. Adapters may raise the public
`InfrastructureFault` contract to declare a lost device or timeout explicitly.

`Cancelled` remains cooperative user cancellation and still stops the campaign.

The lowercase `physicalrsi` package exposes the stable skill, composition,
registry, environment, reporting and CLI interfaces. The Core package supplies
the durable Self-Harness implementation behind those interfaces.

## Checks

Run the core checks from a temporary workspace:

```bash
python -m compileall -q PhysicalRSI PhysicalRSI_core PhysicalRSI_Autoresearch src
python -m pytest -q
```

A passing software or simulator adapter test validates that adapter's declared
scope. It does not by itself qualify a physical system.
