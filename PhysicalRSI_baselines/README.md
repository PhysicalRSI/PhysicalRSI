# PhysicalRSI-baselines

## Research studies

- [Embodied Goodhart's Law](embodied_goodharts_law/README.md) (`egl`): a research
  specification for System 2 → GPT System 1 → CAP optimization and simulation
  benchmark proxy failures, using ASPIRE as the Code-as-Policies reference.
  Planned coverage: LIBERO, LIBERO-Plus and RoboTwin. Adapters and results are
  pending; no exploit result or simulator qualification is claimed.

## Laboratory automation

- [Opentrons liquid handling](opentrons_liquid_handling/README.md): an ASPIRE-like
  code-policy baseline for serial dilution, with upstream API simulation and an
  independent ideal mass-balance audit. The first
  [PhysicalRSI Autoresearch scenario](../PhysicalRSI_Autoresearch/README.md) compares
  method variants and records evidence. No wet-lab qualification is claimed.

## RoboDojo baseline

The RoboDojo baseline imports **PhysicalRSI's XPolicyLab PR #147** as a pinned git submodule. This organization-owned submission carries forward yanming03's original PR #144. It preserves the complete fork, its adapter runtime, evaluation scripts, source and license notices.

- PR: https://github.com/XPolicyLab/XPolicyLab/pull/147
- Fork: https://github.com/PhysicalRSI/XPolicyLab
- Commit: `393f730788e7277c7d86829dd5806a924ac31e74`
- Adapter: `robodojo/XPolicyLab/policy/physicalRSI/`
- Machine-readable provenance: [`robodojo/provenance.json`](robodojo/provenance.json)

The commit is pinned regardless of later PR updates. The surrounding `robodojo/` modules retain the preview release's task integration and frozen bundle tools; the submodule is the authoritative PR snapshot. Installation does not replace either source tree with the other.

## Initialize

From the PhysicalRSI repository root:

```bash
git submodule update --init PhysicalRSI_baselines/robodojo/XPolicyLab
physicalrsi --command /robodojo
```

Follow the adapter's [installation instructions](robodojo/XPolicyLab/policy/physicalRSI/README.md) to download its checksum-verified assets, prepare code programs and create a skill configuration. Retain the XPolicyLab and dependency licenses. The adapter requires its model assets, simulator, policy interpreter and an image-capable agent endpoint; these are not bundled with the lightweight CLI.

## Run from the terminal or workbench

Configure before opening the workbench:

```bash
export PHYSICALRSI_XPOLICYLAB_ROOT="$PWD/PhysicalRSI_baselines/robodojo/XPolicyLab"
export PHYSICALRSI_SKILL_CONFIG=/path/to/skills.json
export PHYSICALRSI_POLICY_PYTHON=/path/to/policy/python
export ROBODOJO_CONDA_ENV=your_robodojo_environment
# Set PHYSICALRSI_AGENT_API_KEY through your environment/secret manager.
physicalrsi
```

```text
/robodojo general_pickup
/jobs
/logs
/baselines
```

For a simulator in a virtual environment, use `PHYSICALRSI_SIM_PYTHON` instead of `ROBODOJO_CONDA_ENV`. `PHYSICALRSI_POLICY_GPU` and `PHYSICALRSI_ENV_GPU` select devices. Jobs record their actual process status, logs and result directory. Archived videos are optional and are discovered under `PHYSICALRSI_DATA_ROOT`; importing the PR does not assert a leaderboard result.

The baseline is evaluation-only. Dexjoco data generation and continual training are implemented separately under `PhysicalRSI_demos/showcase/`.
