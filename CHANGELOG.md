# Changelog

## Unreleased

- Separate policy deadlines from infrastructure faults in trial evidence.
- Retry infrastructure-faulted paired trials deterministically and drop an
  exhausted pair from both arms' statistics.
- Record host-versus-port timing and document retry behavior for physical
  links and sensors.
- Charge policy and environment calls to the trial budget while measuring
  bookkeeping separately, with a configurable whole-trial safety ceiling.

## 0.3.0

This release turns the PhysicalRSI core into a versioned installable dependency.

- Added structural contracts and typed skill composition.
- Added paired-t, paired-permutation, and paired-betting statistics, including cohort power and attainability checks.
- Added the sequential evaluator for block-wise early stopping and resume.
- Added a language-model mutation proposer that accepts data-only JSON edits within a declared parameter space.
- Added append-only trial journals, durability policies, and cached source digests for durable execution.
- Added content-addressed sensor blobs and evidence verification.
- Added the Gymnasium adapter and a runnable quickstart.
- Added skill design principles and a review checklist.
