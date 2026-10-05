#!/usr/bin/env python3
"""Phase 0 gate and Phase 1 grid on a real score table (CPU only, minutes).

    python scripts/analyze_score_table.py tables/qwen1.5b_countdown.npz --out results/qwen1.5b_countdown

Writes <out>_phase0.json (density, plain-RandOpt K-curve, gate G1) and
<out>_phase1.csv (every design x eps x K x n cell, mean and SE over repeats,
plus per-item vote correctness for the paired tests in <out>_items.npz).

Discipline (plan R06/R09): run this on the PUBLIC proxy first (``--select-on
public``) to choose the design, sigma grid and K; run on the private pool only
for the pre-registered cells, once.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dprandopt import designs  # noqa: E402
from dprandopt.scoretable import ScoreTable  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table")
    ap.add_argument("--out", required=True)
    ap.add_argument("--select-on", choices=["pool", "public"], default="pool")
    ap.add_argument("--eps", default="1,2,4,8")
    ap.add_argument("--ks", default="1,5,10,50")
    ap.add_argument("--ns", default="200,500,1000,2000")
    ap.add_argument("--designs", default="D0,D1,D2,D7,D8")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--prescreen-m", type=int, default=None)
    ap.add_argument("--public-q", type=int, default=None)
    a = ap.parse_args()

    t = ScoreTable.load(a.table)
    if a.select_on == "public":
        # treat the public split as the 'private' pool for design selection
        t = ScoreTable(pool_reward=t.public_reward, test_answer=t.test_answer, reward_max=t.reward_max,
                       sigma=t.sigma, seed=t.seed, base_index=t.base_index)
    rng = np.random.default_rng(0)
    base = float(t.test_acc[t.base_index])

    # ---------------- Phase 0
    ks_curve = [1, 2, 5, 10, 20, 50, 100, 200]
    full = np.arange(t.pool_size)
    curve = {k: designs.select(t, "nonprivate", k=k, pool_cols=full, rng=rng).vote_acc
             for k in ks_curve if k <= t.n_candidates}
    phase0 = dict(base_test_acc=base, n_candidates=t.n_candidates, pool=t.pool_size,
                  density_test={m: t.density(m / 100) for m in (2, 5, 10)},
                  density_pool={m: t.density(m / 100, full) for m in (2, 5, 10)},
                  plain_randopt_k_curve=curve,
                  sigma_share_of_top50=None if t.sigma is None else
                  {str(s): float(np.mean(t.sigma[np.argsort(-t.pool_sum(full))[:50]] == s))
                   for s in np.unique(t.sigma)})
    n1000 = min(1000, t.pool_size)
    cols = t.subsample_pool(n1000, rng)
    k50 = designs.select(t, "nonprivate", k=min(50, t.n_candidates), pool_cols=cols, rng=rng).vote_acc
    phase0["gate_G1"] = dict(k50_at_n1000=k50, gain_points=100 * (k50 - base), passed=bool(k50 - base >= 0.05))
    with open(a.out + "_phase0.json", "w") as f:
        json.dump(phase0, f, indent=2, default=float)
    print(json.dumps(phase0, indent=2, default=float))

    # ---------------- Phase 1 grid
    rows, items = [], {}
    for n in (int(x) for x in a.ns.split(",")):
        if n > t.pool_size:
            continue
        col_sets = [t.subsample_pool(n, rng) for _ in range(a.reps)]
        for k in (int(x) for x in a.ks.split(",")):
            ref = np.mean([designs.select(t, "nonprivate", k=k, pool_cols=c, rng=rng).vote_correct
                           for c in col_sets], axis=0)
            items[f"nonprivate_n{n}_k{k}"] = ref
            for eps, d in itertools.product((float(x) for x in a.eps.split(",")), a.designs.split(",")):
                if k == 1 and d not in ("D1", "D8"):
                    continue
                corr = []
                for r, c in enumerate(col_sets):
                    s = designs.select(t, d, k=k, pool_cols=c, rng=np.random.default_rng(10_000 + r), eps=eps,
                                       delta=a.delta, prescreen_m=a.prescreen_m, public_q=a.public_q)
                    corr.append(s.vote_correct)
                corr = np.array(corr, dtype=float)
                items[f"{d}_n{n}_k{k}_eps{eps:g}"] = corr.mean(axis=0)
                accs = corr.mean(axis=1)
                rows.append(dict(n=n, k=k, eps=eps, design=d, acc=accs.mean(),
                                 se=accs.std(ddof=1) / np.sqrt(len(accs)), nonprivate=ref.mean(), base=base,
                                 retained=(accs.mean() - base) / max(ref.mean() - base, 1e-9)))
            print(f"n={n} k={k} done", flush=True)
    pd.DataFrame(rows).to_csv(a.out + "_phase1.csv", index=False)
    np.savez_compressed(a.out + "_items.npz", **items)


if __name__ == "__main__":
    main()
