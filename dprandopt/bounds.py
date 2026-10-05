"""Utility bounds and lower bounds for private selection over thickets.

Notation (all accuracies normalised to [0, 1]; rewards are divided by
``reward_max`` R, which therefore appears as the sensitivity):

* n       private-set size
* gamma   density: fraction of the N candidates whose score clears the
          threshold tau (the plan's delta(m) with tau = base + m)
* eps0    per-pick EM parameter; c = 2 under replace-one adjacency, c = 1
          under add/remove with monotone scores
* beta    failure probability

See docs/theory.md for statements and proofs.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import gammaln

from . import accounting as acc


def _c(monotone: bool) -> float:
    return 1.0 if monotone else 2.0


def em_slack(eps0: float, n: int, gamma: float, beta: float = 0.05, monotone: bool = False,
             reward_max: float = 1.0) -> float:
    """Theorem 1 (single pick, empirical): with prob >= 1 - beta the pick's
    empirical score is at least tau - t, with
    t = c R (ln(1/gamma) + ln(1/beta)) / (eps0 n)."""
    return _c(monotone) * reward_max * (math.log(1 / gamma) + math.log(1 / beta)) / (eps0 * n)


def em_k_slack(eps0: float, n: int, gamma: float, k: int, n_candidates: int | None = None,
               beta: float = 0.05, monotone: bool = False, reward_max: float = 1.0) -> float:
    """Corollary 2 (K peeling picks, all K good simultaneously w.p. >= 1 - beta).

    If ``n_candidates`` is given, the density seen by the last pick is
    (gamma N - K + 1)/(N - K + 1) (earlier picks may have removed good
    candidates); the bound is infinite when gamma N < K.  With
    ``n_candidates=None`` this reduces to the plan's formula (density
    unchanged across picks, union bound over the K picks).
    """
    if n_candidates is None:
        g = gamma
    else:
        good = gamma * n_candidates - (k - 1)
        if good <= 0:
            return math.inf
        g = good / (n_candidates - (k - 1))
    return _c(monotone) * reward_max * (math.log(1 / g) + math.log(k / beta)) / (eps0 * n)


def d7_slack(eps: float, n: int, gamma: float, k: int, n_candidates: int, beta: float = 0.05,
             monotone: bool = False, reward_max: float = 1.0) -> float:
    """Proposition 3 (threshold-subset EM, D7): all K members good w.p. >= 1 - beta with
    t = c R (ln[C(N,K)/C(gamma N,K)] + ln(1/beta)) / (eps n)."""
    good = math.floor(gamma * n_candidates)
    if good < k:
        return math.inf
    lr = (gammaln(n_candidates + 1) - gammaln(n_candidates - k + 1)
          - gammaln(good + 1) + gammaln(good - k + 1))
    return _c(monotone) * reward_max * (lr + math.log(1 / beta)) / (eps * n)


def plan_table_slack(eps: float, k: int, n: int, gamma: float = 0.05, delta: float = 1e-5,
                     beta: float = 0.05, method: str = "simple") -> tuple[float, float]:
    """Reproduce one cell of the plan's 'What the bound implies' table.

    Returns (eps0, slack in accuracy points). K=1 uses pure eps; K>1 uses
    BR-zCDP with the given conversion (the plan used the simple one).
    """
    eps0 = eps if k == 1 else math.sqrt(8 * acc.dp_to_zcdp_budget(eps, delta, method) / k)
    return eps0, 100 * em_k_slack(eps0, n, gamma, k, None, beta)


def gibbs_population_slack(eps0: float, n: int, gamma_pop: float, beta: float = 0.05,
                           monotone: bool = False) -> float:
    """Theorem 4 (population, N-free; PAC-Bayes/Gibbs argument).

    With probability >= 1 - beta over the private sample, the EM pick's
    EXPECTED population score (over the mechanism's coins) is at least
    tau* - t, where gamma_pop is the population density at tau* and

      t = sqrt(ln(4/beta)/(2n)) + (ln(2/gamma_pop) + ln(2/beta)) / lam + lam/(8n),
      lam = eps0 n / c  (the EM inverse temperature in mean-score units).

    N and the parameter count do not appear. The last term grows with eps0:
    a noiseless argmax (eps0 -> inf) has no N-free guarantee (winner's curse).
    """
    lam = eps0 * n / _c(monotone)
    return (math.sqrt(math.log(4 / beta) / (2 * n))
            + (math.log(2 / gamma_pop) + math.log(2 / beta)) / lam + lam / (8 * n))


def gibbs_optimal_eps0(n: int, gamma_pop: float, beta: float = 0.05, monotone: bool = False) -> float:
    """eps0 minimising :func:`gibbs_population_slack`: lam* = sqrt(8 n L)."""
    L = math.log(2 / gamma_pop) + math.log(2 / beta)
    lam = math.sqrt(8 * n * L)
    return lam * _c(monotone) / n


def argmax_population_slack(n: int, n_candidates: int, beta: float = 0.05) -> float:
    """Union-bound guarantee for the noiseless argmax: 2 sqrt(ln(2N/beta)/(2n))
    (uniform convergence over all N candidates; depends on ln N)."""
    return 2 * math.sqrt(math.log(2 * n_candidates / beta) / (2 * n))


def packing_lower_bound_n(eps: float, t: float, gamma: float) -> float:
    """Theorem 5 (pure eps-DP lower bound).

    Any eps-DP selector that, on every instance with density gamma, returns a
    candidate within t of the threshold with probability >= 1/2 needs
    n >= (ln(floor(1/gamma)/2)/eps - 1)/t.
    """
    m = math.floor(1 / gamma)
    return max(0.0, (math.log(m / 2) / eps - 1) / t)


def em_upper_n(eps: float, t: float, gamma: float, monotone: bool = False) -> float:
    """n sufficient for Theorem 1 with success probability 1/2 (beta = 1/2)."""
    return _c(monotone) * (math.log(1 / gamma) + math.log(2)) / (eps * t)


def log2_binom(n: int, k: int) -> float:
    return float((gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)) / math.log(2))


def binary_entropy(p: float) -> float:
    if p <= 0 or p >= 1:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def max_confident_members(n_candidates: int, k: int, accuracy: float, extra_bits: float = 0.0) -> float:
    """Fano ceiling for plain RandOpt that releases only K seeds.

    For independent uniform membership bits, sum_j I(S_j; output) <= H(output)
    <= log2 C(N, K) (+ extra_bits for other released choices).  A guesser that
    is right with probability p on a member's bit extracts at least 1 - h(p)
    bits, so at most H / (1 - h(p)) members can be inferred with accuracy p.
    """
    return (log2_binom(n_candidates, k) + extra_bits) / (1 - binary_entropy(accuracy))
