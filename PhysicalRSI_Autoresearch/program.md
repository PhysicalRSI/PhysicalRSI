# Research instructions for an external coding agent

Read README.md, literature.json, study.py and the liquid-handling policy/audit
before proposing an experiment. State the question and a falsifiable hypothesis.
Use public literature and recorded development feedback; do not treat evaluator
internals or validation outcomes as new policy input.

1. Establish a valid baseline on the declared development cases. Preserve the
   no-mixing negative control as a check that API success is insufficient.
2. Propose a small change to reviewed primitive composition or memory. For the
   current runner, submit named JSON settings using `--proposals`; do not load
   generated Python. Record why the change might reduce cost and how it may fail.
3. Run a bounded batch in a fresh temporary workspace. Use the same simulator
   and trial budget for every proposal. Never change audit tolerances to make a
   candidate pass. A new evaluator defines a new study, not an improved score.
4. Read development receipts and simulator logs. Check whether failure is in
   planning, API execution or liquid validity. Retain negative evidence. Never
   label a timeout as a successful low-cost experiment.
5. Let the runner select using development validity and resource costs. Keep the
   incumbent on ties. Inspect final validation separately; report failure rather
   than changing the survivor using validation results.
6. Write the next hypothesis from development evidence. Keep source snapshots,
   parent provenance and limitations. Resume only unchanged studies; use a new
   workspace for new code or proposal batches.
7. Report measured software outcomes and uncertainty. Never infer wet-lab
   success, physical contamination rates, or benchmark-wide performance from an
   ideal concentration ledger. Do not connect to a physical robot from this loop.

The runner performs no autonomous web searches or model calls. This protocol
specifies how an external research agent uses it; it is not evidence that such
an agent ran continuously. Respect the user's budget and stop instructions.
