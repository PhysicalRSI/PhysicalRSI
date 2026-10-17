# PhysicalRSI directory refactor: Phase 1

The lowercase API lives in `src/physicalrsi/`: the root `physicalrsi` is already
an executable launcher. Existing packages and published releases stay in place.
The launcher and installed command now enter the facade, which forwards legacy
arguments to the original CLI. No persistent Core or Autoresearch schema or
engine source changes are required.

## Audit and ownership

| Responsibility | Existing implementation and public callers | Persistence and compatibility |
| --- | --- | --- |
| Agent | `PhysicalRSI.application.Application`, conversation and CLI; called by launcher and task factories | Workspace task binding, model configuration and memory retain their paths and formats. |
| Research | `PhysicalRSI_Autoresearch.records.ResearchRecord`, `heartbeat`, `study`; external agents and research scripts | Immutable record history, CAS updates, heartbeat/index. Records do not select policies. |
| Evolution | `PhysicalRSI_core.self_harness.research_campaign.ResearchCampaign`, `SelfHarness`, `selection`; demos and baseline workflows | Campaign implementation hashes, round receipts, memory, lineage. Moving or editing these sources can prevent existing campaign resume; they stay unchanged. |
| Skills | `PhysicalRSI.Embodied_Harness.skills.composition`, Core `Operation`, `SkillLibrary`; Application and RoboDojo execution skills | Exact contracts, implementation revisions, memory binding, effects. Published adapters retain old imports. |
| Environments | `PhysicalRSI_core.experiments.Environment`, `Policy`, `Verifier`, `TrialAdapters`; task suites | Reset identities, timing, device ownership and artifact verification stay in Core. |
| Runtime | Core `infra` storage, executor, services, scheduler and evidence; all workflows | Atomic persistence, locks, content hashes and unknown-effect reconciliation remain authoritative. |
| Benchmarks | RoboDojo/provider modules in `PhysicalRSI_baselines`, demonstrations in `PhysicalRSI_demos` | External checkpoints/simulators and published release copies are untouched. |

## Actual facade reuse

```text
src/physicalrsi/
  cli.py                    -> PhysicalRSI.cli.main for legacy arguments
  skills/protocol.py        -> Core Contract, Context, Operation
  skills/registry.py        -> Core digest, locked, freeze_json
  skills/compose.py         -> existing sequence composer
  environments/protocol.py  -> existing Core experiment ports
  environments/robodyna.py  -> explicit software contract fixture
  research/workflow.py      -> unchanged ResearchCampaign and ResearchRecord
  reporting/bundle.py       -> Core hashes and storage conventions
examples/robodyna/
  skills.py                 EstimateMotion -> PredictIntercept -> MoveToTarget
  study.py                  suite, bounded proposer, verifier and study wiring
```

The facade freezes a composition declaration as research preparation evidence.
Its revision is also part of the campaign's existing scope/context. Candidate
skill parameters are included in the ordinary dependency closure, and the
policy constructs the same sequence from the pinned candidate. The research
record binds the initial harness and evaluator for the campaign; the result
links the actual selected revision. A composition reference alone is not proof
that a skill improved.

## Evolution unit and validation

Only `PredictIntercept.json` belongs to the mutable `skills` component in this
example. EstimateMotion, MoveToTarget, graph order and implementation source
identities remain fixed. The proposer uses a declared constant-velocity
hypothesis to change the gain from 0 to 1; it is a scripted contract experiment,
not an autonomous scientific discovery claim. Shared skills consume observations
and parameters, never task IDs or evaluation seeds.

The real Core campaign runs one development trial, paired parent/candidate
validation on two different cases, and one report-only test. Core selects the
survivor. Autoresearch records the research disposition after that result;
completion of a process alone cannot support improvement. Six trial reservations
are expected. Only the predictor parameter changes, so the emitted diff is a
parameter-file diff, not a claim that Python source was synthesized.

The report bundle includes actual trajectories, candidate parameter diff,
config, results, research record and an execution summary log. Unsupported
videos are omitted. Reopening a completed study audits existing campaign and
record evidence without dispatching new trials. Unknown interrupted effects
still require Core reconciliation; the facade does not blindly retry them.

## Deferred changes and unverified integrations

- Native RoboDyna installation, observation/action mapping, simulator scoring,
  timing and video export are unverified. `--fixture` is mandatory.
- No real hardware or RoboDyna benchmark performance is claimed.
- Test cases are held-out conditions within the fixture, **not held-out tasks**.
  Cross-task generalization must later compare the same revised skill in
  multiple frozen compositions, with independent held-out tasks used only for
  reporting. Current fixture success is not evidence of such transfer.
- Future skill libraries may use perception/manipulation/verification/recovery
  folders. Phase 1 does not invent executable grasp or recovery implementations.
- The registry loads data only. After process restart callers must explicitly
  supply a trusted implementation with the recorded identity; no plugin loading
  or code import from untrusted manifests occurs.
- General branching, loops, automatic graph search, ontology enumeration,
  cross-embodiment conversion and distributed research are deferred.
- Providers remain responsible for correctly binding implementation revisions.
  File hashes verify identity, not scientific validity or tamper resistance
  against an actor able to replace the whole workspace.

Keep the full study workspace when archiving: the report references original
research evidence and is not a replacement for lineage, memory and round receipts.
