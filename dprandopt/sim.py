"""A linearised-Gaussian 'thicket' simulator for score tables.

This is a stylised stand-in for the GPU score table, used to test the code
paths, check the theory and size the experiments before any GPU time is spent.
It is NOT evidence about real LLMs.

Model
-----
Item j (pool, public or test) has a cluster c_j and a base logit b_j ~ N(mu_b, s_b^2);
the base model answers item j correctly iff b_j > 0. Perturbation i with relative
scale s_i (sigma_i / sigma_ref) shifts item logits by a linearised random direction

    Delta_ij = s_i * (g0 * z_i + gc * w_{i, c_j} + gi * e_ij) - s_i^2 * h,

with z_i, w_{i,c}, e_ij i.i.d. N(0, 1): a general component (whole-task skill),
a cluster component (specialists) and an item-level component (idiosyncratic
behaviour that does not transfer between items), plus a curvature penalty h.
Item j is answered correctly iff b_j + Delta_ij > 0 (greedy decoding is
deterministic). Wrong answers are invalid with probability p_invalid, else the
item's dominant distractor with probability q_dominant, else one of the
remaining distractors uniformly.

Because the perturbation is Gaussian and the response is linear in it, a data-free
neighbour of an expert (D3) at relative scale s' is the expert's direction plus a
fresh one: s* (.)* + s' (.)', with curvature penalty (s*^2 + s'^2) h. That makes
D3 testable inside the same model.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from .scoretable import ScoreTable


@dataclass(frozen=True)
class ThicketParams:
    mu_b: float = -1.5
    s_b: float = 1.0
    g0: float = 0.45          # general-skill sensitivity
    gc: float = 0.3           # cluster (specialist) sensitivity
    gi: float = 0.5           # item-level sensitivity
    h: float = 0.1            # curvature penalty per unit squared scale
    n_clusters: int = 8
    scales: tuple = (0.5, 1.0, 2.0)   # the sigma grid relative to sigma_ref
    p_invalid: float = 0.4
    q_dominant: float = 0.5
    n_distractors: int = 6
    format_credit: bool = False       # Countdown-style reward: answer + 0.1 * format


SCENARIOS = {
    # Calibrated by a coarse grid search (N=5000, n=200, plain RandOpt) to the
    # paper's Qwen2.5-1.5B Countdown row: base 6.7%, K=1 ~43-47%, K=50 ~52.7%.
    "countdown_like": ThicketParams(mu_b=-1.5, g0=0.28, gc=0.3, gi=0.6, h=0.12, p_invalid=0.35,
                                    q_dominant=0.6, format_credit=True),
    # Calibrated to the Qwen2.5-3B GSM8K row: base 79.8%, K=50 87.1%.
    "gsm8k_like": ThicketParams(mu_b=0.84, g0=0.12, gc=0.15, gi=0.45, h=0.05, p_invalid=0.15,
                                q_dominant=0.8),
}


@dataclass
class SimulatedTable:
    table: ScoreTable
    params: ThicketParams
    # latent state kept to generate D3 neighbours (test items only)
    scale: np.ndarray        # [N]
    z: np.ndarray            # [N]
    w: np.ndarray            # [N, C]
    e_test: np.ndarray       # [N, T]
    b_test: np.ndarray       # [T]
    c_test: np.ndarray       # [T]
    dominant_test: np.ndarray

    def neighbor_fn(self, rel_scale: float):
        """Return a D3 neighbour generator at relative scale ``rel_scale`` (data-free)."""
        p = self.params

        def fn(expert: int, count: int, rng: np.random.Generator) -> np.ndarray:
            s_e = self.scale[expert]
            s_n = rel_scale
            T = self.b_test.size
            z_n = rng.standard_normal(count)
            w_n = rng.standard_normal((count, p.n_clusters))
            e_n = rng.standard_normal((count, T))
            delta = (p.g0 * (s_e * self.z[expert] + s_n * z_n)[:, None]
                     + p.gc * (s_e * self.w[expert][self.c_test][None, :] + s_n * w_n[:, self.c_test])
                     + p.gi * (s_e * self.e_test[expert][None, :] + s_n * e_n)
                     - (s_e ** 2 + s_n ** 2) * p.h)
            correct = (self.b_test[None, :] + delta) > 0
            return _answers(correct, self.dominant_test, p, rng)

        return fn


def _answers(correct: np.ndarray, dominant: np.ndarray, p: ThicketParams, rng) -> np.ndarray:
    shape = correct.shape
    u = rng.random(shape)
    other = rng.integers(2, p.n_distractors + 1, size=shape)
    wrong = np.where(u < p.p_invalid, -1,
                     np.where(u < p.p_invalid + (1 - p.p_invalid) * p.q_dominant, dominant[None, :], other))
    return np.where(correct, 0, wrong).astype(np.int16)


def simulate_table(params: ThicketParams, n_candidates: int = 1000, pool: int = 2000, public: int = 200,
                   test: int = 1000, seed: int = 0, include_base: bool = True,
                   chunk: int = 250) -> SimulatedTable:
    """Draw one candidate population and its full score table."""
    rng = np.random.default_rng(seed)
    p = params
    J = pool + public + test
    b = rng.normal(p.mu_b, p.s_b, size=J)
    c = rng.integers(0, p.n_clusters, size=J)
    dominant = np.ones(test, dtype=np.int16)

    N = n_candidates
    scale = rng.choice(np.asarray(p.scales, dtype=float), size=N)
    z = rng.standard_normal(N)
    w = rng.standard_normal((N, p.n_clusters))
    if include_base:
        scale[0] = 0.0
        z[0] = 0.0
        w[0] = 0.0

    pool_reward = np.empty((N, pool), dtype=np.float32)
    public_reward = np.empty((N, public), dtype=np.float32)
    test_answer = np.empty((N, test), dtype=np.int16)
    e_test = np.empty((N, test), dtype=np.float32)
    for s in range(0, N, chunk):
        sl = slice(s, min(N, s + chunk))
        m = sl.stop - sl.start
        e = rng.standard_normal((m, J))
        if include_base and s == 0:
            e[0] = 0.0
        delta = (p.g0 * (scale[sl] * z[sl])[:, None]
                 + p.gc * scale[sl][:, None] * w[sl][:, c]
                 + p.gi * scale[sl][:, None] * e
                 - (scale[sl] ** 2)[:, None] * p.h)
        correct = (b[None, :] + delta) > 0
        reward = correct.astype(np.float32)
        if p.format_credit:
            fmt = np.where(correct, 1.0, rng.random(correct.shape) < 0.5)
            reward = reward + 0.1 * fmt.astype(np.float32)
        pool_reward[sl] = reward[:, :pool]
        public_reward[sl] = reward[:, pool:pool + public]
        test_answer[sl] = _answers(correct[:, pool + public:], dominant, p, rng)
        e_test[sl] = e[:, pool + public:]

    table = ScoreTable(pool_reward=pool_reward, test_answer=test_answer,
                       reward_max=1.1 if p.format_credit else 1.0, public_reward=public_reward,
                       sigma=scale, base_index=0, name=f"sim-{seed}")
    return SimulatedTable(table=table, params=p, scale=scale, z=z, w=w, e_test=e_test,
                          b_test=b[pool + public:], c_test=c[pool + public:], dominant_test=dominant)


def strengthen(params: ThicketParams, factor: float) -> ThicketParams:
    """A stand-in for model scale: scale the coherent (general + cluster) sensitivity."""
    return replace(params, g0=params.g0 * factor, gc=params.gc * factor)
