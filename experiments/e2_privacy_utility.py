"""E2: privacy-utility of the ensemble designs on simulated score tables (Phase 1 dry run).

For each scenario one candidate population is drawn (N=2000, pool 2000,
public 200, test 1000); every (n, eps, K, design) cell is then post-processing
of that table, exactly as Phase 1 will run on the GPU table. 20 repetitions per
cell redraw the private subsample of size n and the selection noise; the same
subsample is shared by all designs within a repetition (paired comparison).

Writes results/e2_privacy_utility.csv and results/e2_references.csv.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dprandopt import designs  # noqa: E402
from dprandopt.sim import SCENARIOS, simulate_table  # noqa: E402

RES = os.path.join(os.path.dirname(__file__), "..", "results")


def main(reps: int, n_candidates: int, quick: bool, scenarios, tag: str):
    ns = (200, 1000) if quick else (200, 500, 1000, 2000)
    epss = (1, 2, 4, 8)
    ks = (1, 5, 10, 50)
    rows, refs = [], []
    t0 = time.time()
    for scen in scenarios:
        sim = simulate_table(SCENARIOS[scen], n_candidates=n_candidates, pool=2000, public=200, test=1000, seed=0)
        t = sim.table
        base_acc = t.test_acc[0]
        nb = {0.5: sim.neighbor_fn(0.5), 1.0: sim.neighbor_fn(1.0)}
        for n in ns:
            rng = np.random.default_rng(1000 + n)
            col_sets = [t.subsample_pool(n, rng) for _ in range(reps)]
            # references (no privacy or no private data)
            for k in ks:
                for name, kw in (("nonprivate", {}), ("public_q50", dict(design="public", public_q=50)),
                                 ("public_q200", dict(design="public", public_q=200)), ("random", {})):
                    design = kw.pop("design", name)
                    accs = [designs.select(t, design, k=k, pool_cols=c, rng=np.random.default_rng(r), **kw).vote_acc
                            for r, c in enumerate(col_sets)]
                    refs.append(dict(scenario=scen, n=n, k=k, design=name, acc=np.mean(accs),
                                     se=np.std(accs, ddof=1) / np.sqrt(reps), base=base_acc))
            for eps, k in itertools.product(epss, ks):
                cells = [("D1", {})] if k == 1 else [
                    ("D0", {}), ("D0_pre100", dict(prescreen_m=100, public_q=50)), ("D1", {}), ("D2", {}),
                    ("D3_s0.5", dict(neighbor_fn=nb[0.5])), ("D3_s1.0", dict(neighbor_fn=nb[1.0])), ("D7", {})]
                for name, kw in cells:
                    design = name.split("_")[0]
                    vals, mem = [], []
                    for r, c in enumerate(col_sets):
                        s = designs.select(t, design, k=k, pool_cols=c, rng=np.random.default_rng(10_000 + r),
                                           eps=eps, delta=1e-5, **kw)
                        vals.append(s.vote_acc)
                        mem.append(s.member_acc)
                    rows.append(dict(scenario=scen, n=n, eps=eps, k=k, design=name, acc=np.mean(vals),
                                     se=np.std(vals, ddof=1) / np.sqrt(reps), member_acc=np.mean(mem),
                                     base=base_acc))
            print(f"{scen} n={n} done ({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(RES, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, f"e2_privacy_utility{tag}.csv"), index=False)
    pd.DataFrame(refs).to_csv(os.path.join(RES, f"e2_references{tag}.csv"), index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--n-candidates", type=int, default=2000)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--scenarios", default="countdown_like,gsm8k_like")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    main(a.reps, a.n_candidates, a.quick, a.scenarios.split(","), a.tag)
