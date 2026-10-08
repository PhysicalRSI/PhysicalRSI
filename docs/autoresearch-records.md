# Durable research records

The first integration step adds `PhysicalRSI_Autoresearch.records.ResearchRecord`.
It connects an external agent's hypothesis to the existing selected harness,
research memory, declared budget, experiment identity, result and disposition.
It does not run a model, launch a job, or replace Self-Harness selection.

## Inspect and resume

For the current research workspace, run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m PhysicalRSI_Autoresearch.records \
  --index /cpfs02/user/shaoyanming/egl_runtime/autoresearch-pipeline-001/index.json
```

For another workspace, use `--workspace /absolute/path/to/research-record`.
The command verifies recorded evidence, memory and selected-parent lineage, and
prints the persisted stage and next action. It does not inspect remote processes
or launch work. A stale parent is reported explicitly; corrupted evidence raises
an error rather than returning a healthy status.

The index contains explicit record paths, rather than guessing the newest run:

```json
{
  "schema": "physicalrsi.research-index/v1",
  "records": {"example": "/absolute/path/to/research-record"}
}
```

`current.json` points to a content-addressed snapshot under `history/`. Updates
require the expected revision and use a writer lock. A restarted agent should
read the index and current record before taking action. A concurrent stale
writer must reload; it cannot dispatch the same record again.

## Lifecycle

| State | Next operation | Required binding |
| --- | --- | --- |
| `preparing` | `bind_experiment` | Verified candidate, evaluator SHA256, existing `CaseProtocol`, fresh output workspace, used lesson IDs, preparation evidence |
| `ready` | `claim_dispatch` | Expected record revision, current selected parent, unchanged memory/candidate/evidence, fresh output workspace, job ID |
| `awaiting_result` | Inspect the actual job; then `record_result` | Reviewed result report matching dispatch revision, experiment digest and job ID |
| `awaiting_decision` | `close` | Research disposition, evidence, memory revision and explanation of its treatment |
| `completed` | Register a new hypothesis when justified | This record remains inspectable and immutable |

Create a record through `ResearchRecord.create` with a question, hypothesis,
falsifier, competing explanations, `max_trials` and `trial_seconds`, a
`HarnessState`, a `ResearchMemory` revision, applicability context and hashed
development evidence. The selected parent is resolved from lineage, never
inferred from an arbitrary candidate directory. Retrieved lesson IDs are
recorded at creation; used IDs must be a subset when binding the experiment.
These IDs are caller attestations of use, not a measurement of causal benefit.
Constructors such as `SelfHarness` may write preparation files. If inspecting
their configuration before binding, use a separate preparation directory;
reserve the declared experiment workspace for execution after the dispatch claim.

`claim_dispatch` persists intent **before** an external launcher acts. Its return
does not establish a running process. If the agent crashes, inspect the job ID,
registry and experiment workspace before recording a result. The module does
not automatically retry, and it refuses to close a dispatched record until the
job has been reconciled through a result report. An undispatched proposal may
be rejected or superseded with evidence.

The external reviewed adapter writes a result envelope with these fields:

```python
from PhysicalRSI_core.infra.storage import atomic_json, digest, file_digest

# `dispatched` is the snapshot returned by claim_dispatch.
# `receipt_path` is a completed native result or an infrastructure audit.
atomic_json(report_path, {
    "schema": "physicalrsi.research-result/v1",
    "research_revision": dispatched["revision"],
    "experiment_sha256": digest(dispatched["body"]["experiment"]),
    "job_id": dispatched["body"]["job_id"],
    "outcome": "completed",  # or failed / inconclusive
    "evidence": {str(receipt_path.resolve()): file_digest(receipt_path)},
})
record.record_result(dispatched["revision"], report=report_path)
```

This envelope verifies association and artifact identity. The adapter remains
responsible for checking actual receipts, process termination and native
outcomes. Merely exiting a process or passing a hash check cannot demonstrate
policy success. Keep infrastructure errors distinct from hypothesis failures.

Closing a record requires `supported`, `rejected`, `inconclusive` or `superseded`,
an explanation, evidence and explicit memory handling. Retaining the existing
memory is valid when explained. A new memory revision must descend from the
pinned revision. No operation commits or promotes a harness; existing selectors
and lineage remain responsible for that decision.
Status checks also revalidate the final memory revision, its intermediate
ancestry and referenced evidence after closure.

## Heartbeat follow-through

Call `record.complete_review(heartbeat, request_id, review, expected=revision)`
after inspecting jobs and evidence. The existing five-field heartbeat review
is preserved, with its `next_experiment` string containing structured JSON:
research workspace, exact revision, computed next action and the explanation.
The immutable research snapshot is added to the review's hashed evidence.

The timer itself is unchanged. A review still needs real evidence and reflection,
and an absent agent is not silently replaced by this module. Reading a historical
review identifies what was planned then; reading the current index identifies
what has happened since.

## Current integration and limits

The first live record tracks LIBERO component-cover research. Its historical
evidence includes failed native cover trials and three successful retained
snapshot planning replays. It is registered as `preparing`; no new rollout or
fresh-layout gain is implied. Its context describes the conditional research
scope; it does not assert that a future frame passes a component check.

The record's budget is a planning bound. Dispatch adapters must enforce it
through existing trial quotas and evaluator timeouts. `bind_experiment` is not
an admission gate: `MemoryBoundSuite`, mechanism checks, negative controls and
native paired evaluation must still be wired into the eventual launcher.
The module cannot prevent another program bypassing it, enforce ownership
across separate research records, or certify public-only perception.

Run contract checks with:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  tests/test_autoresearch_records.py tests/test_research_heartbeat.py
```

These verify restart, immutable history, stale-parent rejection, duplicate
dispatch rejection, result binding, evidence corruption, memory ancestry and
heartbeat linkage. They do not measure native robotics performance.

The 2026-10-08 software integration check also exercised an actual counter
Self-Harness campaign: five trials within a six-trial budget selected the bound
candidate, then the record collected the result and closed with explicit memory
retention. A postprocessing assertion incorrectly expected the full budget to
be consumed; recovery inspected persisted evidence and completed the record
without rerunning trials. This validates software orchestration only.
