"""Statistics for the pre-registered comparison (test items are the unit)."""

from __future__ import annotations

import math

import numpy as np
from scipy import stats


def holm(pvalues, alpha: float = 0.05):
    """Holm step-down: returns (reject flags, adjusted p-values) in input order."""
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj <= alpha, adj


def paired_test(a: np.ndarray, b: np.ndarray):
    """Paired t-test on per-item scores (a, b may be fractional after averaging
    over selection-noise repeats). Returns (mean difference, two-sided p)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    if np.allclose(d, 0):
        return 0.0, 1.0
    t, p = stats.ttest_1samp(d, 0.0)
    return float(d.mean()), float(p)


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> float:
    """Exact McNemar test for binary per-item correctness."""
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    n01 = int(np.sum(a & ~b))
    n10 = int(np.sum(~a & b))
    n = n01 + n10
    if n == 0:
        return 1.0
    return float(min(1.0, 2 * stats.binom.cdf(min(n01, n10), n, 0.5)))


def mcnemar_power(diff: float, discordant: float, n_items: int, alpha: float) -> float:
    """Normal-approximation power of McNemar's test.

    diff = p10 - p01 (accuracy difference), discordant = p10 + p01.
    """
    nd = discordant * n_items
    d = diff / discordant
    z_a = stats.norm.ppf(1 - alpha / 2)
    return float(stats.norm.cdf((abs(d) * math.sqrt(nd) - z_a) / math.sqrt(1 - d * d)))


def min_detectable_diff(discordant: float, n_items: int, alpha: float, power: float = 0.8) -> float:
    lo, hi = 1e-4, discordant - 1e-6
    for _ in range(60):
        mid = (lo + hi) / 2
        if mcnemar_power(mid, discordant, n_items, alpha) < power:
            lo = mid
        else:
            hi = mid
    return hi
