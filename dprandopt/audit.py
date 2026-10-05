"""Auditing tools for selection mechanisms.

Three levels, from cheapest to strongest:

1. ``one_run_eps_lower_bound``: the Steinke, Nasr & Jagielski (NeurIPS 2023)
   one-run audit for pure eps-DP. With r guesses about independent
   Bernoulli(1/2) canary inclusions, an eps-DP mechanism gets at most
   Binomial(r, e^eps/(1+e^eps)) right; observing v correct rejects every eps
   whose tail probability is below alpha. (Their delta > 0 correction is not
   implemented; use it for the pure-DP designs D2, D3, D7 and K=1.)

2. ``multi_run_eps_lower_bound``: Clopper-Pearson bounds on the TPR/FPR of a
   membership test over many runs. Selection is post-processing of the stored
   score table, so thousands of runs with and without a canary cost seconds.

3. ``exact_privacy_curve``: for the exponential mechanism / Gumbel top-K the
   output probabilities are Plackett-Luce and can be evaluated exactly, so the
   privacy loss L(o) = log P_D(o) - log P_D'(o) of a given neighbouring pair is
   computable for every sampled output, and
   delta(eps) = E_{o ~ P_D}[(1 - exp(eps - L(o)))_+] is estimated without bias.
   A value above the accounted delta flags an accounting bug.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import optimize, stats

from . import mechanisms as mech


def one_run_eps_lower_bound(r: int, v: int, alpha: float = 0.05) -> float:
    """Largest eps rejected at level alpha after v correct out of r guesses (pure DP)."""
    if r <= 0 or v <= r / 2:
        return 0.0

    def tail(eps):
        p = 1.0 / (1.0 + math.exp(-eps))
        return stats.binom.sf(v - 1, r, p)  # P[W >= v]

    if tail(0.0) > alpha:
        return 0.0
    hi = 1.0
    while tail(hi) <= alpha and hi < 100:
        hi *= 2
    if tail(hi) <= alpha:
        return hi
    return optimize.brentq(lambda e: tail(e) - alpha, 0.0, hi)


def clopper_pearson(k: int, n: int, alpha: float):
    lo = stats.beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi


def multi_run_eps_lower_bound(tp: int, n_pos: int, fp: int, n_neg: int, delta: float = 0.0,
                              alpha: float = 0.05) -> float:
    """eps lower bound from a membership test's TPR/FPR (Clopper-Pearson, two-sided alpha)."""
    tpr_lo, _ = clopper_pearson(tp, n_pos, alpha)
    _, fpr_hi = clopper_pearson(fp, n_neg, alpha)
    num = tpr_lo - delta
    if num <= 0 or fpr_hi <= 0:
        return 0.0
    return max(0.0, math.log(num / fpr_hi))


def exact_privacy_curve(scores_d: np.ndarray, scores_d2: np.ndarray, k: int, eps0: float,
                        eps_grid, sensitivity: float = 1.0, monotone: bool = False,
                        samples: int = 20000, rng=None):
    """Monte-Carlo estimate of delta(eps) for Gumbel top-K on a fixed neighbouring pair.

    Returns (eps_grid, delta_estimates, privacy_loss_samples) for the direction D -> D'.
    """
    rng = np.random.default_rng(rng)
    lg1 = mech.em_logits(scores_d, eps0, sensitivity, monotone)
    lg2 = mech.em_logits(scores_d2, eps0, sensitivity, monotone)
    losses = np.empty(samples)
    for s in range(samples):
        noisy = lg1 + rng.gumbel(size=lg1.shape)
        o = np.argsort(-noisy)[:k]
        losses[s] = mech.plackett_luce_logprob(lg1, o) - mech.plackett_luce_logprob(lg2, o)
    eps_grid = np.asarray(eps_grid, dtype=float)
    deltas = np.array([np.mean(np.clip(1 - np.exp(e - losses), 0, None)) for e in eps_grid])
    return eps_grid, deltas, losses


def worst_case_pair(scores: np.ndarray, target: np.ndarray, sensitivity: float = 1.0):
    """A replace-one neighbour that moves ``target`` candidates up and all others down.

    Replacing a record with reward 0 on ``target`` and R elsewhere by one with
    R on ``target`` and 0 elsewhere changes every score by exactly +-R.
    """
    d2 = scores.astype(float).copy()
    mask = np.zeros(scores.size, dtype=bool)
    mask[target] = True
    d2[mask] += sensitivity
    d2[~mask] -= sensitivity
    return d2
