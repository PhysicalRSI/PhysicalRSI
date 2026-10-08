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

The inventory adapter is implemented. Runtime execution, Self-Harness integration,
selection and simulator qualification are still pending. On initial inspection,
the existing host exposed an idle NVIDIA GPU but no reachable Docker daemon;
this is a deployment finding, not a task failure. Do not provision instances in
the simulation or agent partitions to work around it.
