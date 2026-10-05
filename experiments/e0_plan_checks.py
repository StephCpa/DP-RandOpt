"""E0: re-derive every number in the research plan and apply the corrections.

Writes results/e0_plan_checks.md. Pure arithmetic; runs in seconds.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dprandopt import accounting as acc  # noqa: E402
from dprandopt import bounds  # noqa: E402
from dprandopt import stats as st  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "e0_plan_checks.md")
DELTA = 1e-5
lines: list[str] = []
w = lines.append


def table(header, rows):
    w("| " + " | ".join(header) + " |")
    w("| " + " | ".join("---" for _ in header) + " |")
    for r in rows:
        w("| " + " | ".join(str(x) for x in r) + " |")
    w("")


w("# E0: checks of the plan's arithmetic\n")

# ---------------------------------------------------------------- 1. slack table
w("## 1. The 'What the bound implies' table\n")
w("Slack t in accuracy points; density gamma = 0.05; 95% guarantee for all K members; delta' = 1e-5.\n")
rows = []
for eps in (4, 8):
    for k in (1, 10, 50):
        e0, _ = bounds.plan_table_slack(eps, k, 200)
        rows.append([eps, k, f"{e0:.2f}"] + [f"{bounds.plan_table_slack(eps, k, n)[1]:.1f}" for n in (200, 1000, 5000)])
w("**(a) Plan as written** (simple zCDP conversion, density unchanged across picks, reward range 1):\n")
table(["eps", "K", "eps0", "n=200", "n=1000", "n=5000"], rows)

w("**(b) Corrected**: tight CKS conversion; peeling correction with N = 1000 (the plan's experiment) "
  "and N = 5000; Countdown reward range R = 1.1. `inf` means gamma*N < K, so the guarantee is void.\n")
rows = []
for eps in (1, 2, 4, 8):
    for k in (1, 10, 50):
        e0 = acc.calibrate_eps0(eps, DELTA, k, "cks")
        cells = []
        for n in (200, 1000, 5000):
            for N in (1000, 5000):
                t = 100 * 1.1 * bounds.em_k_slack(e0, n, 0.05, k, N)
                cells.append("inf" if math.isinf(t) else f"{t:.1f}")
        rows.append([eps, k, f"{e0:.3f}"] + cells)
table(["eps", "K", "eps0 (CKS)", "n=200,N=1k", "n=200,N=5k", "n=1000,N=1k", "n=1000,N=5k",
       "n=5000,N=1k", "n=5000,N=5k"], rows)

w("**(c) Designs that escape the K bottleneck** (R = 1.1, N = 5000, gamma = 0.05, 95%): D1 = K Gumbel picks "
  "(CKS); D7 = one EM over K-subsets (pure eps, all members good simultaneously); D3 = one pick (pure eps), "
  "neighbours data-free. Add/remove adjacency with monotone scores halves every entry.\n")
rows = []
for eps in (1, 2, 4, 8):
    for k in (10, 50):
        for n in (200, 1000, 2000):
            e0 = acc.calibrate_eps0(eps, DELTA, k, "cks")
            d1 = 100 * 1.1 * bounds.em_k_slack(e0, n, 0.05, k, 5000)
            d7 = 100 * 1.1 * bounds.d7_slack(eps, n, 0.05, k, 5000)
            d3 = 100 * 1.1 * bounds.em_slack(eps, n, 0.05)
            rows.append([eps, k, n, f"{d1:.1f}", f"{d7:.1f}", f"{d3:.1f}"])
table(["eps", "K", "n", "D1 slack", "D7 slack", "D3 expert slack"], rows)

# ---------------------------------------------------------------- 2. per-pick budgets
w("## 2. Per-pick eps0 for K picks at (eps, 1e-5)\n")
rows = []
for eps in (1, 2, 4, 8):
    rows.append([eps] + [f"{acc.calibrate_eps0(eps, DELTA, k, 'simple'):.3f} / {acc.calibrate_eps0(eps, DELTA, k, 'cks'):.3f}"
                         for k in (1, 5, 10, 50)])
table(["eps", "K=1", "K=5", "K=10", "K=50"], rows)
w("Each cell: simple conversion / CKS conversion. K=1 uses pure DP (eps0 = eps).\n")

# ---------------------------------------------------------------- 3. population, temperature, lower bound
w("## 3. Population guarantee, optimal temperature, and lower bound\n")
rows = []
for n in (100, 200, 500, 1000, 2000):
    e_star = bounds.gibbs_optimal_eps0(n, 0.05)
    rows.append([n, f"{e_star:.2f}", f"{100 * bounds.gibbs_population_slack(e_star, n, 0.05):.1f}",
                 f"{100 * bounds.gibbs_population_slack(8.0, n, 0.05):.1f}",
                 f"{100 * bounds.argmax_population_slack(n, 1000):.1f}",
                 f"{100 * bounds.argmax_population_slack(n, 5000):.1f}"])
table(["n", "optimal eps0", "Gibbs slack at optimum", "Gibbs slack at eps0=8", "argmax slack N=1k",
       "argmax slack N=5k"], rows)
w("Population slack in points (gamma_pop = 0.05, beta = 0.05, replace-one). The Gibbs bound does not "
  "depend on N; the argmax union bound does.\n")

rows = []
for eps in (0.25, 0.5, 1.0, 2.0):
    for g in (0.001, 0.01, 0.05):
        rows.append([eps, g, f"{bounds.packing_lower_bound_n(eps, 0.05, g):.0f}",
                     f"{bounds.em_upper_n(eps, 0.05, g):.0f}"])
table(["eps", "gamma", "n needed (lower bound, any eps-DP selector)", "n sufficient (EM, prob 1/2)"], rows)
w("Target slack t = 5 points. The lower bound is vacuous once eps > ln(1/(2 gamma)); there the EM's "
  "remaining gap to optimal is at most the additive constant.\n")

# ---------------------------------------------------------------- 4. information ceiling
w("## 4. Information ceiling for plain RandOpt (seeds only)\n")
rows = []
for N, K, n in ((5000, 50, 200), (1000, 50, 1000), (1000, 1, 1000)):
    bits = bounds.log2_binom(N, K)
    rows.append([N, K, n, f"{bits:.0f}", f"{bits / n:.2f}", f"{bounds.max_confident_members(N, K, 0.9):.0f}",
                 f"{bounds.max_confident_members(N, K, 0.99):.0f}"])
table(["N", "K", "n", "log2 C(N,K) bits", "bits/example", "members inferable at 90%", "at 99%"], rows)
w("Releasing the top-K train rewards, base train reward and per-sigma means (as RandOpt's results.json "
  "does) voids this ceiling.\n")

# ---------------------------------------------------------------- 5. budget
w("## 5. Compute budget re-derivation\n")
tok = 3000 * 250
rows = []
total = 0
for name, tps, tasks, N in (("0.5B, two tasks", 10000, 2, 1000), ("1.5B, two tasks", 6000, 2, 1000),
                            ("3B, two tasks", 3500, 2, 1000), ("7B, Countdown", 1700, 1, 1000)):
    per = tok / tps
    gpuh = per * N * tasks / 3600
    total += gpuh
    rows.append([name, tps, f"{per:.0f} s", f"{gpuh:.0f}"])
table(["item", "tokens/s", "per candidate", "GPU-h"], rows)
other = 90 + 100 + 35 + 65
w(f"Scoring subtotal {total:.0f} GPU-h + other items {other} = {total + other:.0f} GPU-h (plan: ~645).\n")
w("Raising N to 2000 (needed for K = 50 at gamma = 0.05, Section 1b) doubles scoring: "
  f"{2 * total + other:.0f} GPU-h. Scoring the test set only for candidates the mechanisms can "
  "reach saves up to a third of scoring.\n")

# ---------------------------------------------------------------- 6. power
w("## 6. Power of the primary comparison\n")
rows = []
for disc in (0.2, 0.3, 0.4):
    rows.append([disc] + [f"{100 * st.min_detectable_diff(disc, 1000, a):.1f}" for a in (0.05, 0.05 / 2, 0.05 / 4)])
table(["discordant share", "MDE alpha=0.05", "alpha=0.025", "alpha=0.0125 (Holm first step)"], rows)
w("Minimum detectable accuracy difference (points) at 80% power, 1000 test items, McNemar normal "
  "approximation. Plan: ~5 points (6 after Holm) at 30% discordance -- confirmed.\n")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    f.write("\n".join(lines))
print("\n".join(lines))
