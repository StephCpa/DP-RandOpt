"""E5: what audits can and cannot show for selection mechanisms (RQ4 design).

(a) Natural membership inference: AUC of 'selected members' reward on example
    j minus the all-candidate mean', members vs held-out pool examples.
(b) Crafted canaries + many runs (cheap: selection is post-processing of the
    table): empirical eps lower bounds vs the accounted eps, and the exact
    Plackett-Luce privacy curve on the same worst-case pair.
(c) One-run audit (Steinke et al. 2023) of plain RandOpt and of D1: the
    largest eps it can certify when at most K outputs carry signal.

Writes results/e5_audit.md.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dprandopt import accounting as acc  # noqa: E402
from dprandopt import audit  # noqa: E402
from dprandopt import mechanisms as mech  # noqa: E402
from dprandopt.sim import SCENARIOS, simulate_table  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "e5_audit.md")
L = []
w = L.append


def auc(pos, neg):
    """Mann-Whitney AUC."""
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1.0
    # average ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / cnt)[inv]
    r_pos = ranks[: pos.size].sum()
    return (r_pos - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size)


sim = simulate_table(SCENARIOS["countdown_like"], n_candidates=2000, pool=2000, public=10, test=200, seed=3)
t = sim.table
R = t.reward_max
N = t.n_candidates
rng = np.random.default_rng(0)

# ----------------------------------------------------------------------------- (a)
w("# E5: auditing selection mechanisms\n")
w("Simulated Countdown-like table, N = 2000 candidates, rewards in [0, 1.1].\n")
w("## (a) Membership inference with natural examples\n")
w("Statistic for example j: mean reward of the selected members on j minus the mean over all "
  "candidates. Members = the n private examples; non-members = n held-out pool examples. "
  "AUC averaged over 10 private subsamples (0.5 = no signal).\n")
w("| n | K | plain RandOpt | D1 eps=8 | D1 eps=4 | D1 eps=1 |")
w("|---|---|---|---|---|---|")
centred = t.pool_reward - t.pool_reward.mean(axis=0, keepdims=True)
for n in (200, 1000):
    for k in (1, 50):
        res = {name: [] for name in ("plain", 8, 4, 1)}
        for rep in range(10):
            perm = rng.permutation(t.pool_size)
            mem, non = perm[:n], perm[n:2 * n]
            U = t.pool_sum(mem)
            sel = {"plain": np.argsort(-U)[:k]}
            for e in (8, 4, 1):
                sel[e] = mech.gumbel_top_k(U, k, acc.calibrate_eps0(e, 1e-5, k), R, rng=rng)
            for name, s in sel.items():
                stat = centred[s].mean(axis=0)
                res[name].append(auc(stat[mem], stat[non]))
        w(f"| {n} | {k} | " + " | ".join(f"{np.mean(res[x]):.3f}" for x in ("plain", 8, 4, 1)) + " |")
w("")

# ----------------------------------------------------------------------------- (b)
w("## (b) How stable is plain RandOpt's top-K set?\n")
w("g_K = U_(K) - U_(K+1), the sum-score gap at the selection boundary. One replaced record moves a "
  "score difference by at most 2R, so the top-K set can change with one record only if g_K < 2R; "
  "d = ceil(g_K / 2R) is the distance to instability. 'PTR release' is the probability that "
  "propose-test-release (Dwork & Lei 2009; Thakurta & Smith 2013) would publish the exact "
  "non-private top-K set: d + Lap(1/eps) > ln(1/delta)/eps, which is (eps, 1e-5)-DP. 200 private subsamples.\n")
w("| n | K | P(g_K < 2R) | median d | PTR release eps=1 | eps=4 | eps=8 | candidates within 2R of the boundary (median) |")
w("|---|---|---|---|---|---|---|---|")
for n in (200, 1000):
    for k in (1, 5, 10, 50):
        gaps, near = [], []
        for _ in range(200):
            U = t.pool_sum(rng.permutation(t.pool_size)[:n])
            srt = np.sort(U)[::-1]
            gaps.append(srt[k - 1] - srt[k])
            near.append(int(np.sum(np.abs(U - srt[k - 1]) < 2 * R)))
        gaps = np.array(gaps)
        d = np.ceil(gaps / (2 * R))
        rel = []
        for eps in (1, 4, 8):
            x = np.log(1 / 1e-5) / eps - d                  # release iff Lap(1/eps) > x
            p = np.where(x >= 0, 0.5 * np.exp(-eps * np.abs(x)), 1 - 0.5 * np.exp(-eps * np.abs(x)))
            rel.append(np.mean(p))
        w(f"| {n} | {k} | {np.mean(gaps < 2 * R):.2f} | {np.median(d):.0f} | {rel[0]:.2f} | {rel[1]:.2f} | "
          f"{rel[2]:.2f} | {np.median(near):.0f} |")
w("\nPTR release probability = P[Lap(1/eps) > thr - d] (exact Laplace tail). Plain RandOpt's set is "
  "usually one-record stable for small K, and unstable for K = 50, where many candidates crowd "
  "the boundary.\n")

# ----------------------------------------------------------------------------- (c)
w("## (c) Crafted worst-case canary, many runs\n")
w("A replace-one pair (D, D') moving the scores of the candidates ranked K+1..2K up by R and all "
  "others down by R. Test: at least h of the K selected members lie in that set (h chosen on half "
  "the runs, evaluated on the other half); Clopper-Pearson 95%. 'exact delta' is the "
  "Plackett-Luce privacy curve of the same pair at the accounted eps (Monte Carlo, 4000 samples); "
  "the accountant promises <= 1e-5. 'max loss' is the largest privacy loss sampled.\n")
w("| design | K | accounted eps | empirical eps lower bound (20k runs) | exact delta at accounted eps | max loss |")
w("|---|---|---|---|---|---|")
cols = rng.permutation(t.pool_size)[:1000]
U = t.pool_sum(cols)
for k in (1, 10, 50):
    order = np.argsort(-U)
    target = order[k:2 * k]
    U2 = audit.worst_case_pair(U, target, R)
    inset = np.zeros(N, dtype=bool)
    inset[target] = True
    changed = not np.array_equal(np.sort(np.argsort(-U)[:k]), np.sort(np.argsort(-U2)[:k]))
    w(f"| plain RandOpt | {k} | inf | {'inf (output changes deterministically)' if changed else '0 for this pair (gap > 2R); not DP in general'} | - | - |")
    for eps in (1, 4, 8):
        e0 = acc.calibrate_eps0(eps, 1e-5, k)
        runs = 20000
        c1 = np.array([inset[mech.gumbel_top_k(U2, k, e0, R, rng=rng)].sum() for _ in range(runs)])
        c0 = np.array([inset[mech.gumbel_top_k(U, k, e0, R, rng=rng)].sum() for _ in range(runs)])
        half = runs // 2
        best_h, best = 1, -1
        for h in range(1, k + 1):
            tpr, fpr = np.mean(c1[:half] >= h), np.mean(c0[:half] >= h)
            if fpr > 0 and tpr / fpr > best:
                best, best_h = tpr / fpr, h
        tp, fp = int(np.sum(c1[half:] >= best_h)), int(np.sum(c0[half:] >= best_h))
        lb = audit.multi_run_eps_lower_bound(tp, runs - half, fp, runs - half, delta=1e-5)
        _, d, losses = audit.exact_privacy_curve(U2, U, k, e0, [eps], R, samples=4000, rng=rng)
        w(f"| D1 | {k} | {eps} | {lb:.2f} | {d[0]:.1e} | {losses.max():.2f} |")
w("")

# ----------------------------------------------------------------------------- (d)
w("## (d) What a one-run audit can certify\n")
w("A one-run audit (Steinke, Nasr & Jagielski 2023) with r guesses certifies at most the eps "
  "below, even if every guess is right. Only canaries whose target candidate sits within 2R of "
  "the selection boundary can change a selection output, so r is capped by the last column of "
  "table (b) (single-target canaries), whatever the true leakage.\n")
w("| r (all correct) | certified eps (95%) |")
w("|---|---|")
for r in (5, 10, 20, 50, 100, 500):
    w(f"| {r} | {audit.one_run_eps_lower_bound(r, r):.2f} |")
w("\nPlain RandOpt is not DP for any finite eps, yet a one-run audit with a few dozen informative "
  "canaries certifies eps < 3. For RQ4 use many runs on the stored table (c) or the exact privacy "
  "curve, and report natural-example AUCs (a) separately as average-case leakage.\n")

with open(OUT, "w") as f:
    f.write("\n".join(L))
print("\n".join(L))
