"""E3: selection temperature as a regulariser (the plan's 'side result').

For K = 1 the expected test accuracy of the EM pick is computed exactly,
sum_i p_i(eps0) acc_i, averaged over private subsamples. eps0 = inf is plain
RandOpt's argmax. Theorem 4 predicts an interior optimum at
eps0* = c sqrt(8 L / n) in the worst case; here we see what correlated, realistic
score tables do. Also reports the train-minus-test gap of the pick.

Writes results/e3_temperature.csv.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dprandopt import bounds  # noqa: E402
from dprandopt.mechanisms import em_probabilities  # noqa: E402
from dprandopt.sim import SCENARIOS, simulate_table  # noqa: E402

RES = os.path.join(os.path.dirname(__file__), "..", "results")
EPS0 = (0.25, 0.5, 1, 2, 4, 8, 16, 32, np.inf)


def main(reps=50):
    rows = []
    for scen in ("countdown_like", "gsm8k_like"):
        for N in (1000, 5000):
            sim = simulate_table(SCENARIOS[scen], n_candidates=N, pool=1000, public=10, test=1000, seed=1)
            t = sim.table
            acc = t.test_acc
            for n in (50, 100, 200, 500, 1000):
                rng = np.random.default_rng(n)
                res = {e: [] for e in EPS0}
                gap = {e: [] for e in EPS0}
                for _ in range(reps):
                    cols = t.subsample_pool(n, rng)
                    U = t.pool_sum(cols)
                    train = U / (n * t.reward_max)
                    for e in EPS0:
                        if np.isinf(e):
                            p = np.zeros(N)
                            p[np.argmax(U)] = 1.0
                        else:
                            p = em_probabilities(U, e, t.reward_max)
                        res[e].append(p @ acc)
                        # answer-only train accuracy to compare with test accuracy
                        tr_correct = (t.pool_reward[:, cols] >= 1.0).mean(axis=1)
                        gap[e].append(p @ (tr_correct - acc))
                for e in EPS0:
                    rows.append(dict(scenario=scen, N=N, n=n, eps0=e, test_acc=np.mean(res[e]),
                                     se=np.std(res[e], ddof=1) / np.sqrt(reps), train_test_gap=np.mean(gap[e]),
                                     base=acc[0], best_pop=acc.max(),
                                     gibbs_eps0_star=bounds.gibbs_optimal_eps0(n, 0.05)))
            print(scen, N, "done", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "e3_temperature.csv"), index=False)


if __name__ == "__main__":
    main()
