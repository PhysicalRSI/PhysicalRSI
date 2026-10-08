# RobotWorld baseline

Upstream: https://github.com/robotworldai/robotworld
Paper: https://arxiv.org/abs/2610.10409
Project: https://robotworldai.github.io/

The baseline is under active development on `robotworld`. The pinned source
inventory contains 84 tasks across 20 integrations. No native rollout or
benchmark improvement is claimed yet.

## Reproduce the inventory

```bash
python -m PhysicalRSI_baselines.robotworld.catalog \
  --upstream /path/to/robotworld \
  --workspace /tmp/physicalrsi-robotworld-inventory
```

This reads upstream Git objects at the declared revision, checks unique tasks
and explicit step budgets, and records a content-addressed catalog using core
storage. It does not import upstream Python, launch a simulator, or qualify a
policy. Source restoration, assets, container images and the source-built Codex
runtime remain deployment prerequisites.

## Iteration plan and evidence boundaries

1. Reconcile source restoration and host prerequisites. Start with the upstream
   RoboCasa CloseDrawer task, then expand across the full registered inventory.
   A successful first task does not establish benchmark-wide completion.
2. Run an unchanged baseline through the upstream runtime. Preserve task step
   budgets, observation/action permissions, scoring profiles and three-rollout
   reporting. Default environment-side `code_control` stays disabled.
3. Connect System 1 to the allowed observation/action tools. Codex as System 2
   proposes harness/skill changes using development evidence. Reusable agent-side
   skills must not silently enable environment-side Python or privileged state.
4. Integrate `SelfHarness`, `HarnessState`, immutable `MemoryStore` revisions,
   paired validation and `ImprovementCampaign`. Use fresh validation identities
   only where upstream reset contracts genuinely support them; record fixed-case
   adaptation separately. Do not fabricate held-out splits by relabeling runs.
5. Select survivors from independent evaluator receipts, not agent completion
   text. Preserve failures, timeouts and infrastructure errors separately. Bind
   parent/candidate sources, tools, model identity, memory, budgets and raw
   results into lineage before promoting a skill.
6. Report frozen-start performance and bounded adaptation curves separately,
   including all trials, inference cost and human interventions. Final test
   results are report-only and cannot select a candidate.

The inventory adapter and `receipts.robocasa_receipt` are implemented. The receipt
reader validates one native episode against its task, seed and horizon, preserves
completed failures, and rejects probes and infrastructure/timeout outcomes before
selection. It does not authenticate an arbitrary producer or bind a candidate;
those checks belong to the evaluator integration. Unit fixtures are not native
benchmark evidence.

The campaign binding is implemented; native proposer/evaluator ports,
selection and simulator qualification are still pending. On initial inspection,
the existing host exposed an idle NVIDIA GPU but no reachable Docker daemon;
this is a deployment finding, not a task failure. Do not provision instances in
the simulation or agent partitions to work around it.

## Self-Harness campaign binding

`workflow.campaign(...)` connects a verified catalog and explicitly selected tasks
to the existing `SelfHarness`, `ImprovementCampaign`, paired native-success
selector and `HarnessState`. The protocol freezes original task budgets and keeps
`code_control=False`. Proposer/evaluator executable identities are frozen by core.

This entry point requires real proposer and evaluator ports. Those ports must
bind native launches, reset identities, policy isolation and independent receipts
to the frozen candidate. It does not infer admission from a parsed JSON result.
The runtime ports and the first native campaign are not complete yet. A caller
can run the configured campaign with `.run(initial_harness)` only after supplying
those ports and a complete, verified initial harness closure.

## Deployment and reset identity

The first host diagnostic imported the pinned RoboCasa and robosuite sources
with MuJoCo 3.3.1 and rendered an EGL test scene. The CloseDrawer probe then
stopped during scene construction because `Window051/model.xml` was missing.
It executed no task control steps and supplies neither a policy failure nor a
success. The diagnostic uses host Python 3.12; the upstream container specifies
Python 3.11, so deployment equivalence remains unverified.

The pinned asset manifest contains 124,396 RoboCasa files (about 22.9 GiB).
Scene-first restoration can reduce the initial download to 8,723 files
(about 3.6 GiB, including shared resources) by postponing the object library.
This is an incomplete deployment until every resource needed by the actual
task is present and hash-verified. Missing resources must never trigger scene,
object, seed or task substitutions to obtain a passing result. Preserve cached
downloads and verify them against the release manifest before reuse.

The native evaluator computes the effective reset seed as
`launcher_seed + task_index * num_trials + episode_index`, where `task_index`
comes from the selected upstream task registry. An evaluator port must bind
that effective seed and registry identity to its launch and receipt. Checking
only the command-line seed is insufficient. Repeated launches with the same
effective reset are repeated trials, not held-out layouts. Paired candidate
comparisons must preserve those identities and the original task horizon.

An unchanged official agent run also requires the verified source-built Codex
runtime and its isolated observation/action bridge. A host import or rendering
probe does not establish agent isolation or qualify a benchmark result.

On this DSW, bubblewrap's new procfs mount returned `Operation not permitted`.
`isolation.empty_proc_command` provides an explicit deployment variant for a
trusted upstream command: retain its namespaces and mounts, but provide an
empty `/proc` directory. It never bind-mounts the host's `/proc`. A host diagnostic
verified hidden project paths, read-only observations, writable workspace and
startup of the installed Codex binary. The pinned source app-server subsequently
failed initialization because it requires `/proc/self/exe`.
`isolation.minimal_proc_command` adds only a fixed link to the already bound
`/runtime/codex-app-server`; this variant passed initialization and thread
creation with the pinned source build. It is not general procfs emulation:
subprocess executable discovery and complete agent/tool interaction still need
validation. Record this variant separately; do not
silently fall back, claim unchanged official deployment, or infer full isolation
qualification from these limited checks.

The pinned V8 helper archive returned HTTP 404. The app-server and CLI build
completed without that helper, using the upstream documented
`WORLD_CODEX_DISABLE_CODE_MODE=1` direct-tool mode. Keep that change in the
runtime identity; it does not enable environment-side `code_control`.
