"""Private selection mechanisms over data-independent candidates.

All mechanisms take *sum* scores ``U_i = sum_j r(i, z_j)`` over the private
examples, with per-example rewards in ``[0, sensitivity]``.

Adjacency
---------
* replace-one (default, ``monotone=False``): replacing one example moves every
  ``U_i`` by at most ``sensitivity`` in either direction. The EM samples
  ``i`` with probability proportional to ``exp(eps0 * U_i / (2 * sensitivity))``.
* add/remove (``monotone=True``): adding an example raises every ``U_i`` by a
  value in ``[0, sensitivity]``, so all scores move in the same direction and
  the EM may use ``exp(eps0 * U_i / sensitivity)`` (McSherry & Talwar 2007).
  This halves the utility slack, but the private-set size n is then itself
  private and baselines must be accounted under the same adjacency.

Gumbel top-K (``gumbel_top_k``) returns the same ordered K-tuple distribution as
K rounds of the EM without replacement ("peeling"; Durfee & Rogers 2019); that
tuple distribution is the Plackett-Luce model with the EM logits, so its
probabilities can be computed exactly (``plackett_luce_logprob``), which the
tests and the exact audit use.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy.special import gammaln, logsumexp


def _rng(rng) -> np.random.Generator:
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


def em_logits(sum_scores: np.ndarray, eps0: float, sensitivity: float = 1.0,
              monotone: bool = False) -> np.ndarray:
    """Logits of the exponential mechanism for one pick."""
    c = eps0 / sensitivity if monotone else eps0 / (2.0 * sensitivity)
    return c * np.asarray(sum_scores, dtype=float)


def em_probabilities(sum_scores, eps0, sensitivity=1.0, monotone=False) -> np.ndarray:
    lg = em_logits(sum_scores, eps0, sensitivity, monotone)
    return np.exp(lg - logsumexp(lg))


def exponential_mechanism(sum_scores, eps0, sensitivity=1.0, monotone=False, rng=None) -> int:
    """One eps0-DP pick (Gumbel-max sampling of the EM)."""
    r = _rng(rng)
    lg = em_logits(sum_scores, eps0, sensitivity, monotone)
    return int(np.argmax(lg + r.gumbel(size=lg.shape)))


def gumbel_top_k(sum_scores, k: int, eps0: float, sensitivity=1.0, monotone=False, rng=None) -> np.ndarray:
    """K picks, each an eps0-DP EM pick without replacement, in one shot.

    Returns indices in pick order (first = most preferred).
    """
    r = _rng(rng)
    lg = em_logits(sum_scores, eps0, sensitivity, monotone)
    noisy = lg + r.gumbel(size=lg.shape)
    k = min(k, lg.size)
    top = np.argpartition(-noisy, k - 1)[:k]
    return top[np.argsort(-noisy[top])]


def peeling_em(sum_scores, k: int, eps0: float, sensitivity=1.0, monotone=False, rng=None) -> np.ndarray:
    """Reference implementation: K sequential EM picks without replacement."""
    r = _rng(rng)
    lg = em_logits(sum_scores, eps0, sensitivity, monotone)
    alive = np.ones(lg.size, dtype=bool)
    out = []
    for _ in range(min(k, lg.size)):
        idx = np.flatnonzero(alive)
        p = np.exp(lg[idx] - logsumexp(lg[idx]))
        j = idx[r.choice(idx.size, p=p)]
        out.append(j)
        alive[j] = False
    return np.array(out, dtype=int)


def plackett_luce_logprob_fast(logits: np.ndarray, ordered: np.ndarray) -> float:
    """Vectorised Plackett-Luce log-probability (O(N + K)); falls back to the exact
    loop when the picked items carry almost all of the mass (cancellation)."""
    lg = np.asarray(logits, dtype=float)
    m = lg.max()
    w = np.exp(lg - m)
    picked = w[ordered]
    remaining = w.sum() - np.concatenate([[0.0], np.cumsum(picked)[:-1]])
    if remaining.min() < 1e-9 * w.sum():
        return plackett_luce_logprob(lg, ordered)
    return float(np.sum(lg[ordered] - m - np.log(remaining)))


def plackett_luce_logprob(logits: np.ndarray, ordered: np.ndarray) -> float:
    """Exact log-probability that Gumbel top-K returns the ordered tuple ``ordered``."""
    lg = np.asarray(logits, dtype=float)
    alive = np.ones(lg.size, dtype=bool)
    total = 0.0
    for j in ordered:
        total += lg[j] - logsumexp(lg[alive])
        alive[j] = False
    return float(total)


# ---------------------------------------------------------------------------
# D7: threshold-subset exponential mechanism
# ---------------------------------------------------------------------------

def _log_comb(n, k):
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def threshold_subset_rank_logweights(sum_scores, k: int, eps: float, sensitivity=1.0,
                                     monotone=False):
    """Log-weights over the rank r of the subset minimum (0-indexed, r >= k-1).

    The EM over all K-subsets S with utility q(S) = min_{i in S} U_i has
    sensitivity ``sensitivity`` (and is monotone under add/remove). Sorting
    the scores in decreasing order (ties broken by index), the number of
    K-subsets whose lowest-ranked member sits at rank r is C(r, k-1).
    Returns (order, ranks, logweights).
    """
    u = np.asarray(sum_scores, dtype=float)
    order = np.lexsort((np.arange(u.size), -u))  # descending score, ties by index
    ranks = np.arange(k - 1, u.size)
    c = eps / sensitivity if monotone else eps / (2.0 * sensitivity)
    logw = _log_comb(ranks, k - 1) + c * u[order[ranks]]
    return order, ranks, logw


def threshold_subset_em(sum_scores, k: int, eps: float, sensitivity=1.0, monotone=False,
                        rng=None) -> np.ndarray:
    """D7: one eps-DP exponential mechanism over K-subsets, utility = subset minimum.

    Exact sampler: draw the rank r of the minimum with weight
    C(r, k-1) * exp(c * U_(r)), then k-1 members uniformly from ranks < r.
    The returned order is random (it must not depend on the scores).
    """
    r = _rng(rng)
    u = np.asarray(sum_scores, dtype=float)
    if k > u.size:
        raise ValueError("k larger than number of candidates")
    order, ranks, logw = threshold_subset_rank_logweights(u, k, eps, sensitivity, monotone)
    p = np.exp(logw - logsumexp(logw))
    rmin = ranks[r.choice(ranks.size, p=p)]
    others = r.choice(rmin, size=k - 1, replace=False) if k > 1 else np.array([], dtype=int)
    chosen = np.concatenate([order[others], [order[rmin]]]).astype(int)
    r.shuffle(chosen)
    return chosen


# ---------------------------------------------------------------------------
# D0: Gaussian noisy scores (DP-ES-style release), then top-K
# ---------------------------------------------------------------------------

def noisy_scores_top_k(sum_scores, k: int, sigma: float, rng=None):
    """Release U + N(0, sigma^2 I) and take the top K (post-processing).

    See :func:`dprandopt.accounting.gaussian_scores_sigma` for sigma.
    Returns (indices in decreasing noisy score, noisy scores).
    """
    r = _rng(rng)
    u = np.asarray(sum_scores, dtype=float)
    noisy = u + sigma * r.standard_normal(u.shape)
    k = min(k, u.size)
    top = np.argpartition(-noisy, k - 1)[:k]
    return top[np.argsort(-noisy[top])], noisy


# ---------------------------------------------------------------------------
# D2: sharded picks (parallel composition)
# ---------------------------------------------------------------------------

def sharded_picks(per_example_rewards: np.ndarray, k: int, eps: float, sensitivity=1.0,
                  monotone=False, rng=None) -> np.ndarray:
    """Split the private examples into K data-independent shards; one eps-DP pick per shard.

    Each record lives in exactly one shard, so the whole procedure is eps-DP
    by (adaptive) parallel composition. Earlier picks are excluded from later
    shards so the ensemble has K distinct members.
    """
    r = _rng(rng)
    R = np.asarray(per_example_rewards, dtype=float)
    n_cand, n = R.shape
    perm = r.permutation(n)
    shards = np.array_split(perm, k)
    alive = np.ones(n_cand, dtype=bool)
    out = []
    for sh in shards:
        u = R[:, sh].sum(axis=1)
        lg = em_logits(u, eps, sensitivity, monotone)
        lg[~alive] = -np.inf
        j = int(np.argmax(lg + r.gumbel(size=lg.shape)))
        out.append(j)
        alive[j] = False
    return np.array(out, dtype=int)


# ---------------------------------------------------------------------------
# D8: stability-based release of the exact top-K set (propose-test-release)
# ---------------------------------------------------------------------------

def distance_to_instability(sum_scores, k: int, sensitivity=1.0) -> int:
    """Records that must be replaced before the top-K *set* can change.

    One replaced record moves any score difference by at most 2R, so the set
    is fixed while fewer than ceil(g / 2R) records change, where g is the gap
    between the K-th and (K+1)-th scores.
    """
    u = np.sort(np.asarray(sum_scores, dtype=float))[::-1]
    if k >= u.size:
        return np.iinfo(np.int64).max
    gap = u[k - 1] - u[k]
    return int(math.ceil(gap / (2.0 * sensitivity))) if gap > 0 else 0


def stable_top_k(sum_scores, k: int, eps: float, delta: float, sensitivity=1.0, rng=None) -> Optional[np.ndarray]:
    """(eps, delta)-DP propose-test-release of the exact top-K set (Dwork & Lei 2009;
    Thakurta & Smith 2013). Returns the set in random order, or None (bottom)."""
    r = _rng(rng)
    d = distance_to_instability(sum_scores, k, sensitivity)
    if d + r.laplace(scale=1.0 / eps) <= math.log(1.0 / delta) / eps:
        return None
    u = np.asarray(sum_scores, dtype=float)
    top = np.argsort(-u, kind="stable")[:k]
    r.shuffle(top)
    return top
