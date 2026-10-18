<div align="center">

# PhysicalRSI

<a href="https://github.com/PhysicalRSI/PhysicalRSI/graphs/contributors"><img alt="Made with love" src="https://img.shields.io/badge/Made_with-%E2%99%A5-c46b78?style=flat-square"></a>
<a href="LICENSE"><img alt="License: Apache 2.0" src="https://img.shields.io/badge/License-Apache_2.0-4d8d80?style=flat-square"></a>
<a href="#whats-next"><img alt="Preview" src="https://img.shields.io/badge/Status-Preview-d1a66a?style=flat-square"></a>
<a href="https://mmlab.hk/research/PhysicalRSI"><img alt="Project website" src="https://img.shields.io/badge/Website-PhysicalRSI-526b83?style=flat-square"></a>
<a href="https://github.com/XPolicyLab/XPolicyLab"><img alt="XPolicyLab" src="https://img.shields.io/badge/GitHub-XPolicyLab-526b83?style=flat-square&logo=github&logoColor=white"></a>

<pre>
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠤⠐⠂⠀⠒⠢⠄⡀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡠⠊⠀⠀⠀⠀⠀⠀⠀⠀⠀⠑⠤⡀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡠⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⢦⡀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣼⣤⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢻⡀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣼⠋⠛⡫⠀⠀⠀⢀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⣧
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣰⠁⢀⡔⠀⡀⠀⠀⠈⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠹⣄
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⢤⡿⠀⠀⠁⠀⠀⢰⠀⠀⠀⣠⣤⣶⣿⣿⣦⠦⣶⣾⣷⡟
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⡎⣷⣀⡄⠀⠀⠀⠀⠀⠀⠀⠛⠋⠙⠋⠙⠁⠀⢫⠉⡟
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢨⠁⣿⣿⡣⡠⣶⠀⠀⠀⣄⠀⠀⠀⠀⠁⢀⠀⠀⠀⠀⣧
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠠⢲⣶⣹⠟⣷⣄⣴⠀⠀⠀⠉⠑⠈⠀⠀⠠⠉⠶⣶⣶⠂⢻
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠻⣧⣤⣽⣿⣿⠀⠀⠀⠀⠀⠀⠀⠔⠁⠀⢀⣀⣭⣀⠈⢳
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢻⣿⣯⠻⠀⠀⠀⠀⠀⠀⠀⠀⠀⡞⠉⠀⠀⠙⠂⠀⡆
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣰⡿⢿⡿⣧⡀⠀⢀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣼⠋⠀⠀⠙⠹⣿⣄⠘⣧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢳
⠀⠀⠀⠀⠀⠀⠀⢀⡰⠋⠀⠘⡆⠀⠀⠀⠀⠈⢿⣿⣻⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣇
⠀⠀⠀⠀⠀⢀⠔⠋⠀⠀⠀⠀⢹⡀⠀⠀⠀⠀⠀⠻⣿⣇⢷⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢽
⠀⠀⠀⢀⠔⠁⠀⠀⠀⠀⠀⠀⠀⢇⠀⠀⠀⠀⠀⠀⠙⣿⣿⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡗⢦⣀
⠀⠀⠀⠋⠈⠐⠢⣀⠀⠀⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⠘⢿⠿⣿⣔⣆⠀⠀⠀⠀⠀⠀⠀⠀⠸⡇⠀⢳⠑⡄
⠀⠀⠀⠀⠀⠀⠀⠀⠱⡄⠀⠀⠀⠈⠲⢶⡖⡶⠂⠀⠀⠀⠈⢧⠈⢿⣿⣧⣀⣀⠀⠄⠀⠀⢀⡐⠍⠒⢼⡄⠘⠄
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢆⠀⠀⠀⠀⢀⡗⠁⠀⠀⠀⠀⠀⠈⢣⠀⢣⡹⠙⢛⣿⣦⣿⣶⡄⠏⠀⠀⠀⠑
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠡⡃⠀⠀⠘⢄⠀⠀⠀⠀⠀⠀⠀⠀⢣⠀⠃⣠⠊⠈⢻⡇⢹⡁⠀⠀⠀⠀⠐
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠑⢄⠀⠀⠀⠀⠀⠀⠀⠣⡰⠁⠀⠀⠈⠇⠀⠃
</pre>

> “From so simple a beginning endless forms most beautiful and most wonderful have been, and are being, evolved.”
>
> — Charles Darwin, [*On the Origin of Species*](https://darwin-online.org.uk/Variorum/1859/1859-490-s.html)

</div>

Bring your own robot, simulator, policy, and research questions. Compose them into an RSI system you can inspect and change. A task, a skill, or even a failed experiment can become someone else's starting point.

**This is a preview.** The foundations are here; the interfaces, examples, and documentation will grow with the people using them. There is room to shape what comes next.

## Quick guidance

```bash
git clone --recurse-submodules https://github.com/PhysicalRSI/PhysicalRSI.git
cd PhysicalRSI
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[runtime]'
physicalrsi
```

![The terminal after entering physicalrsi](docs/media/physicalrsi-terminal.png)

Start with a recording, then try an experiment:

| Enter in the terminal | What happens |
| --- | --- |
| `/demo piano` | Open the classical piano performance |
| `/demo dexjoco` | Explore the Dexjoco demonstrations |
| `/robodojo` | Inspect the RoboDojo baseline and its source |
| `/experiment trial-1` | Run a CPU experiment and inspect its evidence |
| `/evolve` | Run the local Self-Harness example in a fresh workspace |
| `/help` | Discover the full command set |

You can also run `physicalrsi preview piano` directly from your shell. The command prints a local browser URL; [remote preview](docs/cli.md#preview) uses SSH port forwarding. Supplied recordings and CPU examples work without a model API key.

For conversation, configure `/model configs/model.example.json`, then ask “Open the piano demo” or “Generate two Dexjoco layouts”. The model uses the same commands as the terminal. [Full CLI guide →](docs/cli.md)

## Compose skills and run a study

The `physicalrsi` Python API connects reusable skills to the existing
Self-Harness and Autoresearch engines. Run the small CPU contract experiment:

```bash
./physicalrsi study robodyna --fixture --workspace /tmp/physicalrsi-robodyna
./physicalrsi skills --root /tmp/physicalrsi-robodyna/registry
```

The study composes `EstimateMotion → PredictIntercept → MoveToTarget` and evolves
only `PredictIntercept`. It records the comparison, trajectories, parameter diff
and a Markdown report at `/tmp/physicalrsi-robodyna/report/report.md`. Repeat the
study command with the same workspace to resume or inspect its completed result.
This example validates the software contracts using a constant-velocity fixture;
it does not run the native benchmark.

After preparing the [isolated RoboDyna runtime](PhysicalRSI_baselines/robodyna/README.md),
run the two-task native study:

```bash
./physicalrsi study robodyna --native \
  --upstream "$ROBODYNA_ROOT" \
  --python "$ROBODYNA_VENV/bin/python" \
  --runtime-env "$ROBODYNA_RUNTIME/environment.json" \
  --evaluation-mode frozen \
  --workspace /tmp/physicalrsi-robodyna-native
```

The native study shares `motion_estimation → intercept_prediction → intercept_planning`
between `catch_ramp_ball` and `catch_valley_ball`. It uses simulator-state
observations and native interactive robot execution. Ramp development can
propose a bounded prediction revision; the frozen survivor then transfers to
Valley without tuning. Keeping the parent is a valid research outcome. Repeat
the same command to resume; use a separate workspace for a different protocol.
See the [native study protocol and evidence](docs/robodyna-native-study.md) for
seed separation, native scoring and inference timing scope.
The [executed pilot results](docs/robodyna-results.md) distinguish interface reuse,
paired selection and independent task outcomes.

After that study completes, start joint evolution in a new workspace:

```bash
./physicalrsi study robodyna --native \
  --upstream "$ROBODYNA_ROOT" \
  --python "$ROBODYNA_VENV/bin/python" \
  --runtime-env "$ROBODYNA_RUNTIME/environment.json" \
  --joint-after /tmp/physicalrsi-robodyna-native \
  --worker-output /tmp/physicalrsi-robodyna-joint-workers \
  --workspace /tmp/physicalrsi-robodyna-joint
```

Joint evolution inherits the verified survivor and uses fresh seeds on both
tasks. `--worker-output` selects a staging directory; Core retains exported
worker evidence in the campaign. Resume requires the campaign's pinned source
and runtime configuration. Use an archived checkout to resume a historical
campaign after executable source changes.

[Framework and extension guide](docs/physicalrsi-framework.md) ·
[Migration and compatibility](migrations/phase1.md)

For hypothesis-driven robot protocol experiments, see
[PhysicalRSI Autoresearch](PhysicalRSI_Autoresearch/README.md). Its resumable
liquid-handling loop uses upstream Opentrons simulation, independent liquid
auditing, and recorded keep/discard decisions. Research agents can use scoped
memory, durable hypothesis/experiment records, and evidence-backed heartbeat
reviews. These are software protocol experiments; wet-lab qualification requires
physical evaluation.

## Project structure and conventions

```text
physicalrsi             public executable launcher
src/physicalrsi/        unified Python API, CLI and adapters
examples/robodyna/      composable skill study and contract fixture wiring
PhysicalRSI/             terminal, conversation, memory, tools and skills
PhysicalRSI_core/        experiment runtime, Self-Harness, evidence and lineage
PhysicalRSI_Autoresearch/ research records, hypotheses and review workflows
PhysicalRSI_baselines/   benchmark integrations and pinned upstream code
PhysicalRSI_demos/       piano, Dexjoco, recordings and runnable examples
configs/                task and model configuration examples
docs/                   setup, architecture and development notes
migrations/             compatibility and migration instructions
```

**PhysicalRSI_core** carries the improvement loop: propose → evaluate → select → inherit. The preview's Self-Harness is joined by a shared experiment runtime that records initial conditions, actions, observations and an independent outcome.

A few conventions keep contributions useful across systems: declare the task and budget, version what changes, retain the evidence, and distinguish a completed process from a successful task. Keep generated datasets, model weights and credentials outside the source tree. The lowercase public API lives under `src/`; existing package imports remain supported.

[Core guide](PhysicalRSI_core/README.md) · [Architecture](docs/architecture.md) · [Development guide](AGENTS.md)

## Baseline · RoboDojo / XPolicyLab

[XPolicyLab](https://github.com/XPolicyLab/XPolicyLab) is the upstream project for our RoboDojo integration. See the [PhysicalRSI integration and provenance](docs/xpolicylab-pr.md) for the reviewed changes and pinned source.

Our RoboDojo baseline includes the agent, task-aware memory, and the pi05, pi05-sparse-memory and code-policy skill integrations in [XPolicyLab PR #147](https://github.com/XPolicyLab/XPolicyLab/pull/147), maintained from the PhysicalRSI organization fork. It carries forward yanming03's original [PR #144](https://github.com/XPolicyLab/XPolicyLab/pull/144).

The full fork is pinned at `393f730` under `PhysicalRSI_baselines/robodojo/XPolicyLab/`; its original evaluation entry is `policy/physicalRSI/`. Models and the simulator are configured separately. Once installed, `/robodojo general_pickup` starts an evaluation and `/logs` follows its output.

[Baseline setup and provenance →](PhysicalRSI_baselines/README.md)

[Physical L1/L2/L3 method →](PhysicalRSI_demos/real_robot/README.md):
isolated code policies, direct-tool rescue, language/video/teleoperation assistance,
and independent L1-only Self-Harness comparison. Robot SDKs and deployment are external.

## Demos

### Auto-engineering lab

A self-built Isaac Sim lab for RGB-D sample transfer and an evidence-driven clearance repair. System 1 executes the code policy; System 2 proposes a bounded memory change, compares it on new paired layouts, and records the selected revision. This optional simulation demo requires an Isaac Sim 5.1 environment.

[Run the lab demo and inspect its evidence →](PhysicalRSI_demos/auto_engineering/README.md)

### Classical piano

Tchaikovsky's *October — Autumn Song*, from *The Seasons*, performed in a virtual practice room.

[Watch the performance](PhysicalRSI_demos/media/tchaikovsky_october.mp4) · [Explore the piano skill library](PhysicalRSI_demos/piano_skill_library/README.md)

The library connects finger primitives, coordination, phrase composition and practice memory. The supplied recording is a score-locked performance showcase; contact-control evaluation remains a separate experiment.

```text
/demo piano
```

### Dexjoco

**Agent generates new layouts → synthesizes new data → continually trains models.**

[Click a mouse](PhysicalRSI_demos/media/dexjoco-click-mouse.mp4) · [Water a plant](PhysicalRSI_demos/media/dexjoco-water-plant.mp4) · [Hammer a nail](PhysicalRSI_demos/media/dexjoco-hammer-nail.mp4)

The live workflow currently targets `click_mouse`. An agent requests fresh layouts, a simulation teacher generates demonstrations, and successful episodes become training data. The continual pi05 loop carries earlier data forward and compares each candidate with its incumbent on separate evaluation layouts.

```text
/demo dexjoco
/layouts 2
/collect 2
/train 100
```

Wait for each stage to complete using `/jobs` or `/logs`. These commands require the [configured Dexjoco runtime](docs/demo-workbench.md). `/train` runs one SmolVLA fine-tuning pass; `/cycle 2` runs the configured continual pi05 workflow for two rounds. `/cycle pause` requests a stop after the current round.

[Demo guide and media provenance →](PhysicalRSI_demos/README.md)

## What's next

- [ ] Make custom task and environment adapters easier to build and share.
- [ ] Connect more skill libraries to executable implementations and reusable evidence.
- [ ] Simplify setup for layout generation, data synthesis and continual training.
- [ ] Expand reproducible evaluations across tasks, simulators and robot embodiments.
- [ ] Refine the terminal experience and add end-to-end contributor examples.

An issue can start with a research question, an unexpected failure, or a system you would like to connect. Small, runnable contributions are welcome.

## Citation

If PhysicalRSI is useful in your work, please cite the [project](https://mmlab.hk/research/PhysicalRSI):

```bibtex
@misc{physicalrsi2026,
  title        = {Physical RSI 1.0: Recursive Self-Harness for Scaling Embodied Skills},
  author       = {{MMLab \& PhysicalRSI contributors}},
  year         = {2026},
  howpublished = {Project website and open-source software},
  url          = {https://mmlab.hk/research/PhysicalRSI},
  note         = {Preview release}
}
```

[Apache-2.0](LICENSE) · [Third-party and media notices](NOTICE)
