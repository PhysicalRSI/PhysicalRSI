# Evidence-bound memory evolution

## Research question and decision

Can System 2 avoid repeating disproven LIBERO interventions without turning a
successful local fix into an unconditional policy? This review follows the
external-agent protocol in `PhysicalRSI_Autoresearch/program.md`: explicit
hypotheses, negative evidence, bounded checks, and separate final validation.
The existing liquid-handling study runner is domain-specific; it was not run
as if it measured robotics memory. This is a literature-informed design review
and software validation, not a comparative robot experiment.

The recommended first implementation is an incrementally revised, scoped
playbook connected to executable checks and existing Self-Harness selection.
This is our engineering inference, not a demonstrated globally optimal method.

| Candidate | Mechanism supported by the source | Fit and remaining limitation |
| --- | --- | --- |
| [Reflexion](https://arxiv.org/abs/2303.11366) | Verbal feedback retained in episodic memory | Useful reflection baseline; prose alone does not enforce evidence scope or prevent repeated invalid experiments. |
| [ACE](https://arxiv.org/abs/2510.04618) | Structured incremental generation, reflection, and curation of a playbook | Best starting point for preserving distinct lessons and revising them without repeatedly compressing history. Needs evidence and qualification boundaries here. |
| [A-MEM](https://arxiv.org/abs/2502.12110) | Structured notes, dynamic links, and evolving contextual representations | Useful later for discovery across failure families. Similarity must not authorize transfer to a different task or sensor. Preserve changes as immutable revisions. |
| [Voyager](https://arxiv.org/abs/2305.16291) | Executable skill library, curriculum, iterative feedback and verification | Useful procedural memory model. Its Minecraft results do not establish transfer to LIBERO. |

We borrow ACE's incremental organization and Voyager's executable-skill idea.
A-MEM-style links may suggest hypotheses; exact declared conditions govern
admission. Reflection produces proposals, not authoritative facts.

## What memory should contain

Keep raw episodes and evidence immutable. A lesson records an applicability
scope, testable hypothesis, supporting evidence, mandatory checks, and known
counterexamples. Keep executable skills in frozen harness artifacts and
reference their versions; do not put unreviewed generated code in a memory
record and execute it. Keep evaluation-only state outside runtime retrieval.

For example, the LIBERO component-cover diagnosis suggests testing a dominant
public RGB-D component before constructing an attached-object cover. It does
not establish that every disconnected component is background. An ambiguous
split is a counterexample requiring rejection or another perception method.
Retained-snapshot path planning is evidence for planning feasibility only;
it cannot qualify grasp execution or fresh-layout success.

System 1 consumes a pinned skill and memory revision. System 2 reads permitted
development evidence, proposes a small revision, evaluates it, and records
selection. Neither reads hidden object poses, task seeds, layout coordinates,
or simulator success internals as policy input. An evaluator may report terminal
success separately. This API cannot certify the provenance of arbitrary caller
inputs: observation allowlists and runtime boundary audits remain necessary.

## Implemented opt-in core mechanism

`PhysicalRSI_core/self_harness/research_memory.py` provides:

- `ResearchMemory.write/read`: content-addressed snapshots with parent lineage
  and file-hash verification. Existing revisions are never edited.
- `retrieve`: active lessons matching all declared conditions. Unknown context
  does not widen scope.
- `counterexample`: retain negative evidence, add its regression check, suspend
  the recommendation, and create a new revision. Historical evidence remains.
- `refine`: propose a narrower or unchanged scope with revised hypothesis and
  evidence, retaining counterexample checks. This re-enables experimentation;
  it does not confirm the lesson. Changed or broader scope needs a new lesson.
- `ResearchMemoryGate`: require evidence-bound checks for every applicable
  lesson, binding the candidate, selected parent, memory, producer and context.
- `MemoryBoundSuite`: compose admission with an existing suite, including a
  preregistered suite. Existing paired evaluation and survivor selection remain
  responsible for performance qualification.

The `write` API remains a low-level snapshot constructor; it can create arbitrary
reviewed lessons and is not an authorization boundary for autonomous proposals.
The narrowed-scope rule is enforced by `refine`, not by all possible writes.
Suspension removes a recommendation from retrieval; it does not impose a global
ban on experiments. The gate is not a runtime collision checker. Evidence hashes
prove identity, not truth, and reports depend on a trusted reviewed producer.
Conflicting hypotheses may coexist as experiments; there is no automatic semantic
conflict resolver. Check-report hashes are preregistered for a bounded candidate
batch; new proposals need a newly frozen gate/study.

## Bounded autoresearch comparison to run next

Falsifiable hypothesis: under the same proposal and execution budget, a scoped,
counterexample-aware memory reduces repeated diagnosed failures without reducing
fresh-layout success relative to frozen or reflection-only memory.

Freeze four arms before collecting new outcomes:

1. Incumbent with no retrieved research lessons.
2. Reflection-only episodic notes, under the same context budget.
3. Scoped but frozen structured memory.
4. Scoped memory with development-only counterexample and refinement updates.

Use the same initial selected harness, perception implementation, action budget,
proposal budget, and preregistered development cases across arms. Log delivered
memory IDs, token counts, proposed revisions, check outcomes, executed primitives,
compute and failures. Rejected proposals count against the proposal budget: an
arm cannot appear efficient by hiding its rejected attempts. Keep known successes
as regression cases and ambiguous perception as negative controls. Ablate memory
updates separately from primitive changes; a new backend shared by both policies
cannot demonstrate a memory-only benefit.

Select using development evidence with incumbent-on-ties and existing regression
gates. Freeze the survivor before disjoint final validation. Do not revise memory
from final validation within that study. Report paired wins/losses, denominators,
uncertainty, repeated-failure rate, and total cost per successful case. Neither
fewer proposals nor more passing admission checks alone demonstrates progress.
The sample size and allowed improvement threshold must be preregistered before
launch; no success-rate claim is made by this document.

## Verification and scope

Run the software contract checks with:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  tests/test_research_memory.py tests/test_research_protocol.py \
  tests/test_improvement_campaign.py
```

These check immutable revisions, scope mismatch, evidence tampering, stale parents,
missing checks, negative checks, suspension and refinement. They do not simulate
LIBERO, reproduce the papers, measure retrieval quality, or demonstrate memory
improvement on robots. Existing live experiment sources were not changed and
existing campaigns do not automatically adopt this opt-in module.
