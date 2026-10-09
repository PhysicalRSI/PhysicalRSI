# PhysicalRSI demos

The package includes four recorded demonstrations. `physicalrsi preview piano` and `physicalrsi preview dexjoco` launch a localhost player with native video controls and seekable streaming. Recordings, active jobs, training logs and evaluation outcomes are labeled separately.

## Classical piano

The featured performance is the exact supplied `tchaikovsky_october_score_locked_practice_room.mp4`: Tchaikovsky, *The Seasons*, Op. 37a, No. 10, October — Autumn Song.

- [Play/download the video](media/tchaikovsky_october.mp4)
- [Original piano skill library explanation](piano_skill_library/README.md)
- [Skill catalog](piano_skill_library/catalog.json)
- [Original diagnostic metadata](media/piano-october.source.json)

The performance uses score-synchronized piano audio and score-locked ShadowHand poses. Its reported F1 describes score actuator activation; it is not evidence of a dynamically feasible, hand-contact-only policy. The copied library retains the original skill cards, contracts, compositions and memory references. References inside those cards resolve in the original Robopianist checkout; planned/experimental capabilities are not promoted by copying the library.

To render a separate performance with an installed compatible Robopianist checkout:

```bash
export ROBOPIANIST_ROOT=/path/to/robopianist-rp1m
export ROBOPIANIST_PYTHON=/path/to/robopianist/python
physicalrsi --command '/task configs/piano.task.json' --command '/check'
physicalrsi --command '/play {"piece":"october","execute":true,"seconds":2}'
```

This optional renderer path is separate from playback of the bundled score-locked recording. `/play` requires an active task and produces a new rendering with its own evidence; `/demo piano` previews the supplied recording directly.

## Dexjoco

**Agent generates new layouts → synthesizes new data → continually trains models.**

Bundled examples from the supplied Dexjoco videos:

- [Click a mouse](media/dexjoco-click-mouse.mp4)
- [Water a plant](media/dexjoco-water-plant.mp4)
- [Hammer a nail](media/dexjoco-hammer-nail.mp4)

The live workflow implements `click_mouse`. The agent requests a bounded layout sample; the Dexjoco environment generates saved initial states. A simulator-assisted teacher generates RGB, proprioception and action chunks. Only successful episodes enter training. Failed attempts remain available for diagnosis.

`/layouts 2` → wait for completion → `/collect 2` → wait → `/train 100` runs one collection/fine-tuning pass. `/cycle 2` uses the configured pi05 fleet for repeated fresh layouts, teacher repair, training with replay, and paired incumbent/candidate evaluation. The other recorded tasks are showcase examples; they are not exposed as live training adapters.

See [configuration and cycle behavior](../docs/demo-workbench.md). Model weights, the simulator and generated datasets remain external. Video playback works without them.

## Research in progress

[Dexterous manipulation with GPT-as-Policy and PhysicalRSI Core](dexterous_manipulation/README.md)
shares the in-progress experiment code and learning-loop design for camera
handling and peg insertion, with future Tianji/BrainCo hardware transfer.
Completed comparisons have no task successes; the research record documents
the failed left-hand pickup and the remaining simulation and hardware work.

## CPU examples

```text
/demo
/run
/skill-memory
/experiment trial-1
/evolve
```

Use a fresh workspace for `/evolve` to run the complete Self-Harness service example. These examples establish local software behavior only.

## Media provenance

[`media/manifest.json`](media/manifest.json) binds each copied file to its source path, size and SHA256. The original piano JSON is preserved separately. Upstream notices are in `media/LICENSE.robopianist` and `media/LICENSE.dexjoco`; see the root [NOTICE](../NOTICE). No video is presented as evidence of a newly completed training or evaluation run.


The [auto-engineering lab](auto_engineering/README.md) adds a self-built Isaac Sim sample-transfer scene and a native Self-Harness repair demonstration. Run it with `./physicalrsi auto-engineering --help`; it requires a separately installed Isaac Sim environment.
