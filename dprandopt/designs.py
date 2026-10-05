"""Ensemble designs for DP-RandOpt, run as post-processing of a ScoreTable.

Designs (privacy is spent only where marked):

* ``base``        the unperturbed model (no data).
* ``random``      K candidates uniformly at random (no data; the eps -> 0 limit).
* ``public``      plain top-K on the public proxy split (no private data).
* ``nonprivate``  plain RandOpt: deterministic top-K on the private pool (not DP).
* ``D0``          Gaussian noisy scores for all (pre-screened) candidates, then
                  top-K -- the DP-ES-style "privatize scores, select by
                  post-processing" route.  [Gaussian mechanism, sqrt(N) L2 sensitivity]
* ``D1``          Gumbel top-K = K exponential-mechanism picks.  [K picks, BR-zCDP]
* ``D2``          one pure pick per disjoint shard of the private set.  [parallel]
* ``D3``          one private pick + K-1 data-free neighbours of it.  [one pick]
* ``D7``          one EM over K-subsets with utility = subset minimum.  [one pick]
* ``D8``          propose-test-release of the exact top-K set when the gap at the
                  boundary is large; falls back to the base model otherwise.  [(eps, delta)]

Every private design may be restricted to a public pre-screen (top-M by the
public proxy score); the pre-screen uses no private data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from . import accounting as acc
from . import mechanisms as mech
from .scoretable import ScoreTable
from .vote import vote_correct

NeighborFn = Callable[[int, int, np.random.Generator], np.ndarray]
"""neighbor_fn(expert_index, count, rng) -> test answers [count, T] of data-free
perturbations around the expert (used by D3 only)."""

PRIVATE_DESIGNS = ("D0", "D1", "D2", "D3", "D7", "D8")


@dataclass
class Selection:
    design: str
    members: np.ndarray                      # candidate indices in vote order (D3: expert only)
    answers: np.ndarray                      # [K, T] test answers in vote order
    eps0: Optional[float] = None
    info: dict = field(default_factory=dict)

    @property
    def vote_correct(self) -> np.ndarray:
        return vote_correct(self.answers)

    @property
    def vote_acc(self) -> float:
        return float(self.vote_correct.mean())

    @property
    def member_acc(self) -> float:
        return float((self.answers == 0).mean())


def _prescreen(table: ScoreTable, m: Optional[int], q: Optional[int]) -> np.ndarray:
    """Candidate indices kept by the public pre-screen (always keeps the base)."""
    if m is None or m >= table.n_candidates:
        return np.arange(table.n_candidates)
    if table.public_reward is None:
        raise ValueError("pre-screen requires public_reward")
    pub = table.public_reward[:, : (q or table.public_reward.shape[1])].sum(axis=1)
    keep = np.argsort(-pub, kind="stable")[:m]
    if table.base_index not in keep:
        keep = np.append(keep[:-1], table.base_index)
    return keep


def select(table: ScoreTable, design: str, *, k: int, pool_cols: np.ndarray,
           rng: np.random.Generator, eps: float = np.inf, delta: float = 1e-5,
           monotone: bool = False, method: str = "cks", prescreen_m: Optional[int] = None,
           public_q: Optional[int] = None, neighbor_fn: Optional[NeighborFn] = None) -> Selection:
    """Run one design once (one draw of the selection randomness)."""
    T = table.test_size
    if design == "base":
        idx = np.array([table.base_index])
        return Selection(design, idx, table.test_answer[idx])

    cand = _prescreen(table, prescreen_m, public_q)
    if design == "random":
        idx = rng.choice(cand, size=min(k, cand.size), replace=False)
        return Selection(design, idx, table.test_answer[idx])
    if design == "public":
        pub = table.public_reward[cand, : (public_q or table.public_reward.shape[1])].sum(axis=1)
        idx = cand[np.argsort(-pub, kind="stable")[:k]]
        return Selection(design, idx, table.test_answer[idx])

    U = table.pool_reward[np.ix_(cand, pool_cols)].sum(axis=1, dtype=np.float64)
    R = table.reward_max
    if design == "nonprivate":
        idx = cand[np.argsort(-U, kind="stable")[:k]]
        return Selection(design, idx, table.test_answer[idx])

    if design not in PRIVATE_DESIGNS:
        raise ValueError(f"unknown design {design!r}")
    if not np.isfinite(eps):
        raise ValueError("private designs need a finite eps")

    if design == "D0":
        sigma = acc.gaussian_scores_sigma(eps, delta, cand.size, R, method)
        local, _ = mech.noisy_scores_top_k(U, k, sigma, rng)
        idx = cand[local]
        return Selection(design, idx, table.test_answer[idx], info={"sigma_sum": sigma})
    if design == "D1":
        eps0 = acc.calibrate_eps0(eps, delta, k, method)
        idx = cand[mech.gumbel_top_k(U, k, eps0, R, monotone, rng)]
        return Selection(design, idx, table.test_answer[idx], eps0=eps0)
    if design == "D2":
        rewards = table.pool_reward[np.ix_(cand, pool_cols)]
        idx = cand[mech.sharded_picks(rewards, k, eps, R, monotone, rng)]
        return Selection(design, idx, table.test_answer[idx], eps0=eps)
    if design == "D7":
        idx = cand[mech.threshold_subset_em(U, k, eps, R, monotone, rng)]
        return Selection(design, idx, table.test_answer[idx], eps0=eps)
    if design == "D8":
        local = mech.stable_top_k(U, k, eps, delta, R, rng)
        if local is None:
            idx = np.array([table.base_index])
            return Selection(design, idx, table.test_answer[idx], info={"released": False})
        idx = cand[local]
        return Selection(design, idx, table.test_answer[idx], info={"released": True})
    if design == "D3":
        expert = int(cand[mech.exponential_mechanism(U, eps, R, monotone, rng)])
        answers = table.test_answer[[expert]]
        if k > 1:
            if neighbor_fn is None:
                raise ValueError("D3 needs neighbor_fn for k > 1")
            answers = np.vstack([answers, neighbor_fn(expert, k - 1, rng)])
        return Selection(design, np.array([expert]), answers, eps0=eps)
    raise AssertionError


def expected_vote_correct(table: ScoreTable, design: str, *, reps: int, k: int, n: int,
                          rng: np.random.Generator, fixed_pool_cols: Optional[np.ndarray] = None,
                          **kw) -> np.ndarray:
    """Per-test-item vote correctness averaged over ``reps`` draws.

    Each repetition draws fresh selection noise; if ``fixed_pool_cols`` is
    None it also redraws which n pool examples form the private set.
    """
    acc_items = np.zeros(table.test_size)
    for _ in range(reps):
        cols = fixed_pool_cols if fixed_pool_cols is not None else table.subsample_pool(n, rng)
        acc_items += select(table, design, k=k, pool_cols=cols, rng=rng, **kw).vote_correct
    return acc_items / reps
