"""E4: does private utility track thicket density (RQ2)? E6: how much does the candidate pool matter?

E4 strengthens the coherent perturbation sensitivity (a stand-in for model
scale, which the paper links to higher density) and measures, at fixed (eps, n),
the gap between private and non-private RandOpt.

E6 redraws the whole candidate population (new seeds) several times and
compares between-population spread with selection-noise spread, i.e. how much
conditioning on one score table understates uncertainty.

Writes results/e4_scale.csv and results/e6_variability.md.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dprandopt import designs  # noqa: E402
from dprandopt.sim import SCENARIOS, simulate_table, strengthen  # noqa: E402

RES = os.path.join(os.path.dirname(__file__), "..", "results")


def mean_acc(t, design, k, n, reps, seed, **kw):
    rng = np.random.default_rng(seed)
    vals = []
    for r in range(reps):
        cols = t.subsample_pool(n, rng)
        vals.append(designs.select(t, design, k=k, pool_cols=cols, rng=rng, **kw).vote_acc)
    return float(np.mean(vals)), float(np.std(vals, ddof=1))


def e4(reps=20):
    rows = []
    for scen in ("countdown_like", "countdown_specialist"):
        for f in (0.6, 0.8, 1.0, 1.2, 1.4):
            sim = simulate_table(strengthen(SCENARIOS[scen], f), n_candidates=2000, pool=2000, public=50,
                                 test=1000, seed=0)
            t = sim.table
            for n in (200, 1000):
                for k in (1, 10):
                    np_acc, _ = mean_acc(t, "nonprivate", k, n, reps, 1)
                    pub, _ = mean_acc(t, "public", k, n, reps, 1, public_q=50)
                    for eps in (1, 4):
                        a, _ = mean_acc(t, "D1", k, n, reps, 2, eps=eps)
                        rows.append(dict(scenario=scen, factor=f, n=n, k=k, eps=eps, base=t.test_acc[0],
                                         density_2=t.density(0.02), density_5=t.density(0.05),
                                         density_10=t.density(0.10), nonprivate=np_acc, public_q50=pub,
                                         dp=a, private_gap=np_acc - a,
                                         retained=(a - t.test_acc[0]) / (np_acc - t.test_acc[0])))
            print(scen, f, "done", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "e4_scale.csv"), index=False)


def e6(pools=6, reps=20):
    lines = ["# E6: variability across candidate pools\n",
             "Six independently drawn candidate populations (N = 1000, same task items), 20 private "
             "subsamples x selection draws each. 'within' = SD over the 20 draws for one pool; "
             "'between' = SD of the six pool means.\n",
             "| scenario | design | eps | K | n | mean acc | within-pool SD | between-pool SD |",
             "|---|---|---|---|---|---|---|---|"]
    for scen in ("countdown_like", "gsm8k_like"):
        tables = [simulate_table(SCENARIOS[scen], n_candidates=1000, pool=1000, public=10, test=1000,
                                 seed=100 + p, item_seed=7) for p in range(pools)]
        for design, eps, k in (("nonprivate", np.inf, 10), ("D1", 4.0, 10), ("D1", 1.0, 10), ("D1", 4.0, 1)):
            means, within = [], []
            for t in (s.table for s in tables):
                kw = {} if design == "nonprivate" else dict(eps=eps)
                m, sd = mean_acc(t, design, k, 1000, reps, 3, **kw)
                means.append(m - t.test_acc[0])
                within.append(sd)
            lines.append(f"| {scen} | {design} | {eps:g} | {k} | 1000 | gain {100 * np.mean(means):.1f} | "
                         f"{100 * np.mean(within):.1f} | {100 * np.std(means, ddof=1):.1f} |")
    lines.append("\nAll pools share the same task items; only the perturbation seeds differ, as in the "
                 "GPU study. One score table is one draw of the candidate pool, so inference that treats "
                 "the pool as fixed is conditional on it.\n")
    open(os.path.join(RES, "e6_variability.md"), "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    e4()
    e6()
