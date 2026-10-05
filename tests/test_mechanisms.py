import itertools
import math
from collections import Counter

import numpy as np
import pytest
from scipy import stats
from scipy.special import logsumexp

from dprandopt import mechanisms as mech


def em_logprob(u, eps0, mono=False):
    lg = mech.em_logits(u, eps0, 1.0, mono)
    return lg - logsumexp(lg)


@pytest.mark.parametrize("mono", [False, True])
def test_em_is_eps_dp_on_random_neighbours(mono):
    rng = np.random.default_rng(0)
    eps0 = 1.3
    worst = 0.0
    for _ in range(2000):
        u = rng.uniform(0, 50, size=6)
        if mono:  # add/remove: every score moves up by a value in [0, 1]
            u2 = u + rng.uniform(0, 1, size=6)
        else:     # replace-one: every score moves by a value in [-1, 1]
            u2 = u + rng.choice([-1.0, 1.0], size=6) * rng.uniform(0.5, 1, size=6)
        worst = max(worst, np.max(np.abs(em_logprob(u, eps0, mono) - em_logprob(u2, eps0, mono))))
    assert worst <= eps0 + 1e-9
    assert worst >= 0.5 * eps0  # the bound is not vacuous on these pairs


def test_em_extreme_pair_attains_eps():
    u = np.zeros(50)
    u2 = u.copy()
    u2[0] += 1
    u2[1:] -= 1
    ratio = em_logprob(u2, 2.0)[0] - em_logprob(u, 2.0)[0]
    assert ratio <= 2.0 + 1e-12
    assert ratio > 1.8


def test_gumbel_topk_matches_plackett_luce_and_peeling():
    rng = np.random.default_rng(1)
    u = np.array([5.0, 4.0, 3.5, 1.0, 0.0])
    eps0, k, n = 1.0, 2, 40000
    lg = mech.em_logits(u, eps0)
    outs = list(itertools.permutations(range(5), 2))
    p = np.array([math.exp(mech.plackett_luce_logprob(lg, np.array(o))) for o in outs])
    assert p.sum() == pytest.approx(1.0)
    for sampler in (mech.gumbel_top_k, mech.peeling_em):
        c = Counter(tuple(sampler(u, k, eps0, rng=rng)) for _ in range(n))
        obs = np.array([c[o] for o in outs])
        _, pval = stats.chisquare(obs, n * p)
        assert pval > 1e-3, sampler.__name__


def subsets_exact(u, k, eps, mono=False):
    c = eps if mono else eps / 2
    subs = list(itertools.combinations(range(len(u)), k))
    logw = np.array([c * min(u[i] for i in s) for s in subs])
    return subs, np.exp(logw - logsumexp(logw))


def test_threshold_subset_sampler_is_exact():
    rng = np.random.default_rng(2)
    u = np.array([9.0, 7.0, 7.0, 4.0, 2.0, 0.0])
    k, eps, n = 3, 1.5, 60000
    subs, p = subsets_exact(u, k, eps)
    c = Counter(tuple(sorted(mech.threshold_subset_em(u, k, eps, rng=rng))) for _ in range(n))
    obs = np.array([c[s] for s in subs])
    _, pval = stats.chisquare(obs, n * p)
    assert pval > 1e-3


def test_threshold_subset_is_eps_dp():
    rng = np.random.default_rng(3)
    eps, k = 1.0, 3
    worst = 0.0
    for _ in range(300):
        u = rng.uniform(0, 10, size=6)
        u2 = u + rng.uniform(-1, 1, size=6)
        _, p1 = subsets_exact(u, k, eps)
        _, p2 = subsets_exact(u2, k, eps)
        worst = max(worst, np.max(np.abs(np.log(p1) - np.log(p2))))
    assert worst <= eps + 1e-9


def test_large_eps_recovers_deterministic_topk():
    rng = np.random.default_rng(4)
    u = rng.permutation(100).astype(float)
    top = np.argsort(-u)[:10]
    assert np.array_equal(mech.gumbel_top_k(u, 10, 500.0, rng=rng), top)


def test_sharded_picks_are_distinct():
    rng = np.random.default_rng(5)
    R = rng.integers(0, 2, size=(30, 100)).astype(float)
    picks = mech.sharded_picks(R, 10, 2.0, rng=rng)
    assert len(set(picks.tolist())) == 10


def test_noisy_scores_topk_shapes():
    rng = np.random.default_rng(6)
    idx, noisy = mech.noisy_scores_top_k(np.arange(20.0), 5, 1.0, rng)
    assert idx.shape == (5,) and noisy.shape == (20,)
