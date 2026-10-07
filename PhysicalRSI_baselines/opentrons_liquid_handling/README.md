# Opentrons liquid-handling code-policy baseline

This baseline adapts the open-source
[Opentrons Python API](https://github.com/Opentrons/opentrons) to a bounded
ASPIRE-like research task: compose reusable pipetting skills into serial dilution
protocols, inspect execution evidence, then improve the composition.
The upstream [serial dilution tutorial](https://docs.opentrons.com/python-api/tutorial/)
provides the task family and hardware API reference. Our code is an original
adapter; upstream implementation is installed as a dependency, not vendored.
Opentrons' repository includes its own [license notices](https://github.com/Opentrons/opentrons/blob/edge/LICENSE).

The software dependency is pinned to `opentrons==8.8.2`, with OT-2 API level 2.19,
a P300 single-channel GEN2, 300 uL tips, a 96-well plate and a 12-channel reservoir.
The run records the installed dependency versions. This is a version pin, not a
claim that the entire execution environment has a content-verified lock.

## Task and policy surface

Prepare 2–12 wells with concentrations `1/2, 1/4, ..., 1/2**n` relative to stock,
each ending with the same 30–150 uL volume. Stock concentration and starting
reservoir quantities are declared task inputs. The material is an abstract
soluble tracer; no biological sample or chemical synthesis is performed.

- Primitive skills: attach/remove tip, aspirate, dispense, mix.
- Memory: mixing repetitions and batching rule.
- Skill combinations: prefill diluent, transfer and mix serially, normalize the
  final well volume by discarding the final transfer volume.
- System 1: the reviewed functions in [policy.py](policy.py) emit actions and an
  executable Opentrons protocol.
- System 2: the [Autoresearch runner](../../PhysicalRSI_Autoresearch/run.py)
  evaluates Codex-authored variants and selects from development evidence.

The upstream simulator checks API execution. The separate [audit](audit.py)
tracks liquid volume and solute mass, rejects withdrawal from an unmixed blend,
and checks final targets. Passing API simulation alone is insufficient. The
ledger assumes ideal mixing and exact pipetting; it is not fluid physics,
collision qualification or a substitute for measured concentration.

The initial negative control deliberately omits mixing. A corrected policy
mixes each dilution. A third policy batches only the initial water deliveries,
using above-well dispensing and a fresh tip per batch. Serial sample transfers
still use fresh tips. Resource comparisons apply only to valid protocols.

Run the copyable commands in the
[Autoresearch scenario](../../PhysicalRSI_Autoresearch/README.md#run).
Results retain `scope: software-protocol-research` and `qualification: null`.
No robot is contacted, no checkpoint is supplied, no ASPIRE result is reproduced,
and no formal Self-Harness or scientific qualification is implied.
