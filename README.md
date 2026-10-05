# DP-RandOpt

Differentially private post-training by **selection**. RandOpt ("Neural Thickets", Gan & Isola, ICML 2026) draws random weight perturbations without data, scores them on a small labelled set, and majority-votes the top K. DP-RandOpt makes only that selection step private, so the privacy cost depends on how many models are picked, not on model size or training steps.

**Start here: [`docs/REPORT.md`](docs/REPORT.md).** It covers what the October 4 research plan survives, what to change before spending GPU time, and the evidence for each point. Proofs are in [`docs/theory.md`](docs/theory.md).

## Status (2026-10-05)

| Part | State |
|---|---|
| Private selection mechanisms (D0, D1, D2, D3, D7, D8), accounting, voting, score table | Implemented; 70 unit tests (exactness against brute force, DP ratio checks, a bug-catching audit) |
| Accounting cross-check | Agrees with AutoDP's exponential-mechanism composition within 0.1% (`results/e7_accounting_crosscheck.md`) |
| Theory (privacy, peeling-corrected utility, N-free population bound, packing lower bound) | `docs/theory.md` |
| Simulation study (7 experiments on calibrated synthetic score tables) | `results/`, `docs/figures/` |
| GPU score-table builder for RandOpt + vLLM | `scripts/build_score_table.py`. **Written, not yet run on a GPU** |
| Phase 0/1 analysis of a real score table | `scripts/analyze_score_table.py` (CPU, minutes) |

## Layout

```
dprandopt/
  accounting.py   pure / BR-zCDP accounting, tight CKS conversion, eps0 calibration
  mechanisms.py   exponential mechanism, Gumbel top-K, threshold-subset EM (D7),
                  sharded picks (D2), Gaussian noisy scores (D0), stability release (D8)
  designs.py      every design as post-processing of a ScoreTable (+ public / random baselines)
  scoretable.py   the stored table: per-example private rewards, test answer codes
  vote.py         RandOpt's plurality vote with its exact tie rule
  bounds.py       utility / population / lower bounds, information ceiling
  audit.py        one-run, many-run and exact (Plackett-Luce) privacy audits
  stats.py        Holm, paired tests, McNemar power
  countdown.py    fresh solvable Countdown instances, solver, RandOpt-compatible rewards
  sim.py          linearised-Gaussian thicket simulator (stylised; for design, not evidence)
experiments/      e0-e5 and the figure script (each writes to results/)
scripts/          build_score_table.py (GPU), analyze_score_table.py (CPU)
tests/            pytest suite
```

## Quick start

```bash
pip install numpy scipy pandas matplotlib pytest tabulate
python -m pytest -q
python experiments/e0_plan_checks.py     # every number in the plan, re-derived and corrected
```

On a real score table:

```bash
python scripts/analyze_score_table.py table.npz --out results/run --select-on public   # choose design, K
python scripts/analyze_score_table.py table.npz --out results/run                      # pre-registered cells, once
```

```python
import numpy as np
from dprandopt import ScoreTable, designs
t = ScoreTable.load("table.npz")
cols = t.subsample_pool(1000, np.random.default_rng(0))
sel = designs.select(t, "D1", k=10, pool_cols=cols, eps=4.0, delta=1e-5, rng=np.random.default_rng(1))
print(sel.members, sel.vote_acc)   # release only the members' seeds and sigmas
```
