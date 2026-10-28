# Working in the physicalRSI preview repository

## Start here

The `./physicalrsi` launcher is the public entry point. The most useful checks are:

```bash
./physicalrsi --help
./physicalrsi --plain --workspace /tmp/physicalrsi-check --command '/help'
python -m pytest -q
```

Use a temporary workspace for new experiments. Do not commit generated workspaces, caches, model checkpoints, simulator output, or credentials.

## Project shape

- `PhysicalRSI/` contains the compatibility CLI, application, task loading, and conversation behavior.
- `PhysicalRSI_core/` contains contracts, storage, isolation, trajectory evidence, Self-Harness, and lineage.
- `PhysicalRSI_Autoresearch/` contains durable research records and the review heartbeat.
- `src/physicalrsi/` contains the stable, task-independent public API and CLI façade.
- `docs/` contains the small set of core architecture and persistence notes.
- `tests/` contains release smoke checks.

## Keep the boundaries clear

The repository has one user-facing workflow: configure a task adapter, compose skills, and let System 2 evolve the declared surface. Preserve scope and qualification fields in returned JSON when changing task adapters.

System 1 is the runtime path that reads observations, task state, skills, tools, and memory. System 2 proposes changes, evaluates candidates, selects a survivor, and records lineage. Keep those responsibilities visible in code and documentation.

A skill route is not a model checkpoint. A route identifies a composable capability and its executable provider; weights, training, and physical execution remain outside the core unless an adapter explicitly supplies them.

Memory snapshots are immutable and content-addressed. New observations or lessons should create a new revision rather than silently changing an old one. Lineage records the parent, candidate, evidence, and selected revision so a result can be inspected later.

## Naming and documentation

Write repository documentation in English. Use `physicalRSI`, `PhysicalRSI`, `System 1`, `System 2`, `Self-Harness`, `skill`, and `memory` consistently. Do not reintroduce retired project names or shorthand labels that obscure what a component does. Prefer a descriptive name such as `code-policy` when naming a capability.

Explain commands with copyable examples and say what the command actually proves. Keep claims about simulators, videos, checkpoints, training, and physical robots precise.

## Active research heartbeat

The user requests a research review every 30 minutes while the benchmark goal
is active. Inspect the durable heartbeat at
`/cpfs02/user/shaoyanming/egl_runtime/autoresearch-heartbeat-001/state.json` on
continuation and before lengthy new experiments. When pending, verify current
processes and new results, reflect on the causal hypothesis, use autoresearch
to compare alternatives when progress stalls, and record an evidence-backed
review through `PhysicalRSI_Autoresearch.heartbeat.ResearchHeartbeat.complete`.
A queued timer event is not a completed research review. Experiment instances
may use the simulation, agent, or other partitions when eligible spare capacity
is verified. Check the current quota identity and permissions before provisioning,
and release borrowed capacity after use. Historical launchers do not replace
current capacity checks. Do not mutate active experiment sources.

On continuation, inspect the explicit research index at
`/cpfs02/user/shaoyanming/egl_runtime/autoresearch-pipeline-001/index.json` with
`python -m PhysicalRSI_Autoresearch.records --index <path>`.
For an indexed research track, resume its persisted next action and use
`ResearchRecord.complete_review` to bind heartbeat follow-through to its current
revision. A dispatch claim requires process/workspace reconciliation before
retrying; it is not evidence that a remote job is running. See
`docs/autoresearch-records.md` for the record lifecycle and remaining boundaries.

## Before finishing a change

Run the smallest relevant check, then the full smoke suite when shared CLI or contract code changed:

```bash
python -m compileall -q PhysicalRSI PhysicalRSI_core PhysicalRSI_Autoresearch src
python -m pytest -q
```

Remove generated `__pycache__`, `.pytest_cache`, and build metadata before committing if the test run recreated them. Do not create pull requests or send external messages from this repository unless the user explicitly asks for that action.
