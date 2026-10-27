# PhysicalRSI mainline

The main branch is the embodiment-agnostic research system. Benchmark adapters,
simulator launchers and hardware-specific policies live on separate branches;
the mainline supplies the contracts those adapters implement.

## Four layers

1. **Applied category theory for skills.** A skill is a typed morphism between
   observation, capability and action objects. `SkillDescriptor` records the
   domain, codomain, effects and revision. `compose_sequence` is the first
   serial composition operator, and `SkillComposition` makes the resulting
   graph inspectable and content-addressed. Future parallel composition is the
   monoidal product of independent skills; adapters are explicit morphisms
   between sensor or embodiment contracts.
2. **Harness management.** A harness freezes the mutable search surface,
   fixed components, evaluation protocol, evidence roots and resource budget.
   `PhysicalRSI_core.self_harness` owns admission, selection, lineage,
   restart reconciliation and qualification boundaries.
3. **Darwinian Self-Harness.** System 2 proposes candidates, evaluates them
   under a declared split, keeps or rejects a survivor, and records immutable
   parent/candidate/evidence lineage. Research memory is append-only and
   counterexamples suspend recommendations instead of silently overwriting
   them.
4. **RSI agent loop.** System 1 executes observations, tools and skills. System
   2 reflects on development evidence, mutates the declared search surface and
   resumes from durable records. `ResearchWorkflow` binds a composition to the
   existing campaign and Autoresearch records instead of introducing another
   selection engine.

## User-defined evolution

The CLI is a generic entry point. A user supplies a `physicalrsi.task/v1`
package and optionally a data-only `physicalrsi.harness/v1` descriptor. The
task package chooses its simulator or real-robot environment; the mainline
does not assume either one.

```bash
physicalrsi harness design --workspace /tmp/my-agent \
  --definition ./my-harness.json

physicalrsi evolve --workspace /tmp/my-agent \
  --task ./my-task.json \
  --harness /tmp/my-agent/rsi-plan.json \
  --scheme hybrid
```

The interactive CLI remains available for users who prefer a persistent agent
session:

```bash
./physicalrsi --plain --workspace /tmp/my-agent --command '/harness show'
./physicalrsi --plain --workspace /tmp/my-agent --command '/rsi'
```

This is Hermes-like in the useful sense: skills, memory, task packages and
evolution are user-owned and persist across sessions. PhysicalRSI adds typed
embodiment contracts, experiment splits, evidence gates and Darwinian lineage;
it does not require a particular model provider or messaging platform.

## Portability boundary

An environment adapter supplies observations, actions, embodiment capabilities,
reset semantics and evidence export. A skill may require RGB, RGB-D,
proprioception, force or contact observations, but that requirement belongs in
its descriptor and capability contract. No skill in the mainline may assume a
benchmark task name, simulator actor handle, evaluation seed or native score.

The main branch therefore stays useful for a new simulator, a real robot, a
different hand, or a purely software environment. Benchmark-specific results
and integration code remain reviewable on their own branches.
