# PhysicalRSI Autoresearch

An evidence-driven research loop for code policies, starting with robot liquid
handling: **can a serial-dilution protocol use fewer tips while preserving
concentration, volume and mixing requirements?**

System 2 supplies a hypothesis and bounded policy settings. Reviewed System 1
skills compile them into an Opentrons protocol. Each experiment runs the upstream
simulator and an independent ideal liquid audit. A successful API simulation
alone cannot win. This is software research: no robot connects, no wet-lab
measurement is produced, and `qualification` remains `null`.

## Autoresearch v0.1 checkpoint

This checkpoint adds scoped research memory and durable research records for
the external-agent workflow. Lessons retain evidence and counterexamples across
immutable revisions. Records bind a hypothesis to the selected parent, pinned
memory, budget, experiment, result and explicit memory disposition. Heartbeat
reviews can reference the exact next-action revision, and an index supports
resuming registered research without guessing from historical directories.

See [research records](../docs/autoresearch-records.md),
[memory evolution](../docs/research-memory-evolution.md), and the
[pipeline integration plan](../docs/autoresearch-pipeline.md). The regression
suite includes a complete CPU Self-Harness campaign through the record lifecycle.
Native launcher adoption, automatic research proposals, mechanism-based dispatch
gates and resource scheduling remain subsequent integration work. This version
does not establish a robotics success-rate improvement from memory.

## What the loop does

1. Freeze the research question, development/validation cases, evaluator sources,
   simulator version, trial budget and proposed hypotheses.
2. Establish the mixed-transfer baseline. Evaluate each subsequent proposal on
   the same development cases; record failures as well as improvements.
3. Keep a valid candidate only when it reduces total tips, then command count.
   Ties retain the incumbent. Simulator errors cannot win.
4. Validate the final survivor on separately declared well counts and volumes.
   Validation never chooses the survivor; a failed validation remains visible.
5. Save source snapshots, generated protocols, actions, logs, receipts, decisions
   and a content-addressed memory revision. Resume only after checking that the
   study configuration, source versions and recorded evidence still match.

The built-in sequence compares explicit mixing, a no-mixing negative control,
batched water delivery, and reuse of the transfer tip for mixing in the same
well. It models no quantitative carryover, viscosity, diffusion or pipetting
bias. Fewer tips in this ideal model is not proof of physical suitability.

## Run and resume

Use Python 3.11+ for physicalRSI and a separate Python 3.10 environment for the
pinned upstream simulator:

```bash
python3.10 -m venv /tmp/physicalrsi-opentrons
/tmp/physicalrsi-opentrons/bin/pip install opentrons==8.8.2
python -m PhysicalRSI_Autoresearch.run \
  --workspace /tmp/physicalrsi-study \
  --sim-python /tmp/physicalrsi-opentrons/bin/python \
  --max-rounds 2

# Continue the remaining rounds, then validate the selected candidate.
python -m PhysicalRSI_Autoresearch.run \
  --workspace /tmp/physicalrsi-study \
  --sim-python /tmp/physicalrsi-opentrons/bin/python \
  --resume
```

Without `--max-rounds`, the runner completes the frozen proposal batch. Each
simulator invocation has a 120-second timeout (`--trial-seconds` changes it).
A completed study that fails validation exits with status 1. Interrupted trials
are retained; resume retries them in a fresh attempt directory. Completed
rounds are not rerun. Workspaces must be outside version control.

| Artifact | Purpose |
| --- | --- |
| `study.json`, `sources/` | Frozen protocol, proposals, installed packages and source snapshots |
| `state.json` | Atomic checkpoint after each completed development round |
| `results.tsv` | Human-readable keep/discard/error ledger |
| `development/`, `validation/` | Attempt-specific actions, protocols, simulator output and hashed receipts |
| `report.md` | Decisions and scope limitations |
| `result.json`, `memory/` | Final selection, validation outcome and immutable evidence references |

A local writer lock prevents simultaneous resume processes. Digests detect
accidental changes; they are not signatures or protection against a malicious
workspace owner. The runner checks source identity but is not a sandbox for
arbitrary Python. Only reviewed local policy functions execute.

## Propose another experiment

Pass a JSON array with `--proposals proposals.json`. Each entry has exactly
`name`, `hypothesis` and `settings`, for example:

```json
[
  {"name": "baseline", "hypothesis": "Explicit mixing provides a valid reference.",
   "settings": {"mix_repetitions": 3, "batch_water": false}},
  {"name": "batch-and-reuse", "hypothesis": "Batch water and reuse the transfer tip for mixing at the same well.",
   "settings": {"mix_repetitions": 3, "batch_water": true, "reuse_mix_tip": true}}
]
```

Start a new workspace for a changed hypothesis batch or evaluator. The supported
search domain is 0–10 mixing repetitions and the two boolean composition rules.
Unknown settings are rejected. This is a bounded experimental engine, not a
claim of an open-ended scientist or admitted Self-Harness campaign. An external
coding agent can follow [program.md](program.md) to review evidence, research a
new hypothesis and prepare the next batch. Live literature retrieval and model
API calls are not hidden inside the runner.

## Research heartbeat

For benchmark research beyond this liquid-handling runner, use the opt-in
[durable research record](../docs/autoresearch-records.md) to bind the hypothesis,
selected parent, memory, experiment and result. Its heartbeat adapter records an
exact next-action revision. It does not launch experiments or replace selection.

An active research agent can use a durable 30-minute review queue:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m PhysicalRSI_Autoresearch.heartbeat \
  --workspace /tmp/physicalrsi-heartbeat --interval-seconds 1800 --watch
```

The timer queues a review immediately, then keeps the original 30-minute
cadence after reviews complete. An overdue review stays pending; repeated ticks
cannot pretend it was handled. It does not start an LLM or perform literature
searches while the agent is absent. The active agent must inspect `state.json`,
verify actual jobs and new evidence, reflect on the current hypothesis, compare
alternatives with autoresearch when needed, and call `ResearchHeartbeat.complete`
with `evidence` (path-to-SHA256 map), `reflection`, `alternatives`, `decision`, and
`next_experiment`. Reviews are content-addressed and source hashes are checked.
The timer does not provision GPUs, send messages, or mutate running experiments.

## Design references

[Karpathy's autoresearch](https://github.com/karpathy/autoresearch/blob/master/program.md)
informs the baseline-first loop, fixed evaluation boundary, resource budget and
keep/discard history. [The AI Scientist](https://github.com/SakanaAI/AI-Scientist/blob/main/ai_scientist/perform_experiments.py)
informs hypothesis-driven experiments and explicit execution budgets. We adapt
those patterns to robot protocols; no upstream implementation is vendored and
neither project's scientific results transfer to this baseline.

[Code as Policies, ASPIRE, Coscientist and Opentrons](literature.json) provide the
robot-policy and laboratory references. Literature claims are kept separate
from measurements made by this runner.

## Validation and next steps

```bash
python -m pytest -q tests/test_autoresearch_liquid_handling.py tests/test_autoresearch_study.py
```

Tests exercise selection, corruption rejection, interruption/resume, simulator
errors and liquid invariants. Test doubles do not establish Opentrons execution.
The original three-policy study passed upstream Opentrons 8.8.2 simulation and
reduced development tips from 32 to 26; it predates the tip-reuse candidate.
Future work includes calibrated liquid error models, measured concentration,
and admission to the isolated Self-Harness before unrestricted code proposals.

## Verified v2 software run

On 2026-10-06, the four-proposal sequence completed ten upstream Opentrons
8.8.2 simulations: eight development trials and two final validation trials.
The no-mixing control passed API simulation but failed the independent liquid
audit. Same-well mixing-tip reuse was selected on development evidence, reducing
total tips from 32 to 16 and primitive commands from 118 to 80. Both validation
cases passed. Reopening the completed study verified its recorded evidence
without rerunning trials. These results apply to the declared ideal liquid model;
physical experiments remain zero. The main-branch smoke suite passed 122 tests.
