# PhysicalRSI

PhysicalRSI is a research system for embodied agents that can inspect their
own skills, propose changes, test them and preserve the evidence. The main
branch is embodiment-agnostic: a simulator, a real robot or a software
environment enters through the same adapter contracts.

The mainline has four connected ideas:

- **Applied category theory for skills.** Observations, capabilities and
  actions are typed objects; skills are versioned morphisms; compositions are
  inspectable graphs with explicit contracts and effects.
- **Harness management.** A harness fixes the mutable search surface,
  evaluation split, resource budget and evidence boundary.
- **Darwinian Self-Harness.** Candidates are proposed, evaluated, selected or
  rejected, and linked by immutable lineage. Failed ideas remain available as
  counterexamples.
- **System 2 RSI loop.** System 1 executes the agent. System 2 reflects on
  development evidence, changes the declared skill or policy surface and
  resumes from durable records.

The CLI is the user-facing research workbench. Users can define task packages,
compose skills, import or design a harness and run their own evolution loop.
It is designed for the same broad workflow as a self-improving agent CLI while
keeping robot-specific execution behind explicit adapters.

## Install

```bash
git clone --recurse-submodules https://github.com/PhysicalRSI/PhysicalRSI.git
cd PhysicalRSI
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[runtime]'
```

The historical interactive CLI remains available:

```bash
./physicalrsi --plain --workspace /tmp/physicalrsi-check --command '/help'
./physicalrsi --plain --workspace /tmp/physicalrsi-check --command '/skill-memory'
```

## Define and evolve an agent

Create a data-only harness definition, then bind it to a user-owned task
adapter. The adapter may target a real robot or a simulator and must expose
the task package protocol.

```bash
physicalrsi harness design \
  --workspace /tmp/my-agent \
  --definition ./my-harness.json

physicalrsi evolve \
  --workspace /tmp/my-agent \
  --task ./my-task.json \
  --harness /tmp/my-agent/rsi-plan.json \
  --scheme hybrid
```

For an interactive session:

```bash
./physicalrsi --plain --workspace /tmp/my-agent --command '/harness show'
./physicalrsi --plain --workspace /tmp/my-agent --command '/rsi'
```

Repeat a command after interruption. The Self-Harness and Autoresearch records
reconcile the exact pending stage instead of silently repeating completed work.

## Skill model

```text
sensor / memory observation
          ↓
typed primitive skill
          ↓
composed capability graph
          ↓
task adapter for a robot or simulator
          ↓
immutable trajectory, report and lineage evidence
```

The public skill interfaces live in [`src/physicalrsi/skills`](src/physicalrsi/skills).
The research and persistence engines remain in [`PhysicalRSI_core`](PhysicalRSI_core).
The architecture and user-defined loop are described in
[`docs/mainline-architecture.md`](docs/mainline-architecture.md).

Benchmark integrations are maintained separately so the main branch remains a
general research substrate rather than a collection of task-specific policies.
