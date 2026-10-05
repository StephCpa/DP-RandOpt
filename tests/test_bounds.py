import numpy as np
import pytest

from dprandopt import bounds


PLAN = {  # (eps, K): slack in points for n = 200, 1000, 5000
    (4, 1): (1.5, 0.3, 0.1), (4, 10): (17.0, 3.4, 0.7), (4, 50): (45.4, 9.1, 1.8),
    (8, 1): (0.7, 0.1, 0.0), (8, 10): (9.1, 1.8, 0.4), (8, 50): (24.2, 4.8, 1.0),
}


@pytest.mark.parametrize("cell", list(PLAN))
def test_reproduces_plan_table(cell):
    eps, k = cell
    for n, want in zip((200, 1000, 5000), PLAN[cell]):
        _, got = bounds.plan_table_slack(eps, k, n)
        assert got == pytest.approx(want, abs=0.051)


def test_theorem1_holds_empirically():
    rng = np.random.default_rng(0)
    from dprandopt.mechanisms import exponential_mechanism
    N, n, gamma, eps0, beta = 2000, 300, 0.05, 1.0, 0.1
    t = bounds.em_slack(eps0, n, gamma, beta)
    fails = 0
    for _ in range(2000):
        u = rng.uniform(0, 0.5, size=N)
        u[: int(gamma * N)] = 0.6  # the good set at tau = 0.6
        i = exponential_mechanism(u * n, eps0, rng=rng)
        fails += u[i] < 0.6 - t
    assert fails / 2000 <= beta


def test_peeling_correction_matters():
    assert np.isinf(bounds.em_k_slack(0.4, 1000, 0.05, 50, n_candidates=900))
    assert bounds.em_k_slack(0.4, 1000, 0.05, 50, n_candidates=1000) > bounds.em_k_slack(0.4, 1000, 0.05, 50)


def test_lower_bound_below_upper_bound():
    for eps in [0.1, 0.5, 1.0]:
        for gamma in [0.001, 0.01, 0.05]:
            assert bounds.packing_lower_bound_n(eps, 0.05, gamma) <= bounds.em_upper_n(eps, 0.05, gamma)


def test_gibbs_optimum():
    e = bounds.gibbs_optimal_eps0(200, 0.05)
    f = lambda x: bounds.gibbs_population_slack(x, 200, 0.05)
    assert f(e) <= min(f(0.8 * e), f(1.25 * e))


def test_bits_ceiling():
    assert bounds.log2_binom(5000, 50) == pytest.approx(400, abs=10)
