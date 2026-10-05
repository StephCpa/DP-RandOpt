import numpy as np
import pytest

from dprandopt import accounting as acc
from dprandopt import audit


def test_one_run_bound_basic():
    assert audit.one_run_eps_lower_bound(100, 50) == 0.0
    assert audit.one_run_eps_lower_bound(100, 100) > 2.5
    a = audit.one_run_eps_lower_bound(1000, 600)
    b = audit.one_run_eps_lower_bound(1000, 700)
    assert 0 < a < b


def test_multi_run_bound():
    e = audit.multi_run_eps_lower_bound(1000, 1000, 0, 1000)
    assert 5 < e < 6.5


@pytest.mark.parametrize("k", [1, 5])
def test_exact_curve_respects_accounting(k):
    rng = np.random.default_rng(0)
    eps, delta = 2.0, 1e-5
    e0 = acc.calibrate_eps0(eps, delta, k)
    u = rng.uniform(0, 20, size=40)
    target = np.argsort(-u)[k:2 * k]          # push the runners-up into the top-K
    u2 = audit.worst_case_pair(u, target)
    _, d, _ = audit.exact_privacy_curve(u, u2, k, e0, [eps], samples=4000, rng=rng)
    assert d[0] <= delta + 1e-3


def test_exact_curve_catches_a_bug():
    # an implementation that forgot to split the budget across the K picks
    rng = np.random.default_rng(1)
    k, eps = 20, 1.0
    u = np.zeros(60)
    u2 = audit.worst_case_pair(u, np.arange(k))
    _, d_ok, _ = audit.exact_privacy_curve(u, u2, k, acc.calibrate_eps0(eps, 1e-5, k), [eps],
                                           samples=3000, rng=rng)
    _, d_bug, _ = audit.exact_privacy_curve(u, u2, k, eps, [eps], samples=3000, rng=rng)
    assert d_ok[0] <= 1e-5
    assert d_bug[0] > 0.1
