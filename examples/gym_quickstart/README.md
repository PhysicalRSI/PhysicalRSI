# Gymnasium quickstart

This example runs the complete PhysicalRSI Self-Harness loop against
Gymnasium's `CartPole-v1`. It declares five `SearchField` values in
`control.json`, proposes bounded Gaussian controller mutations, evaluates each
round with paired permutation selection, and reports a held-out test set.

From the repository root, run:

```bash
python examples/gym_quickstart/run.py
```

Expected output has five round lines followed by a held-out result. The exact
incumbent IDs and success count depend on the installed Gymnasium version and
randomized controller proposals:

```text
round 1: incumbent=parent outcome=retained
round 2: incumbent=gaussian-...-2 outcome=inherited
round 3: incumbent=gaussian-...-2 outcome=retained
...
held-out success: 8/8 (1.000)
```

The task is CartPole-v1 with a 500-step horizon and the cart kept within 0.3 m.
The starting controller balances the pole with weak cart feedback and succeeds
in about 60 percent of episodes; the loop proposes four Gaussian mutations of
the four controller gains per round, evaluates each against the incumbent on 24
fresh paired validation layouts, and accepts a child only when the paired
permutation test clears alpha 0.05 split over 24 comparisons. Wall time is
about 5 to 6 minutes on one CPU core. Validation layouts must be fresh in every
round: the core rejects reused validation inputs.
