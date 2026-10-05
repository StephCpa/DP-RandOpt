import math

import numpy as np
import pytest
from scipy import optimize, stats

from dprandopt import accounting as acc


def analytic_gaussian_eps(sigma, delta, sens=1.0):
    """Exact (eps, delta) curve of the Gaussian mechanism (Balle & Wang 2018)."""
    def d(eps):
        a = sens / (2 * sigma)
        b = eps * sigma / sens
        return stats.norm.cdf(a - b) - math.exp(eps) * stats.norm.cdf(-a - b)
    return optimize.brentq(lambda e: d(e) - delta, 0, 200)


def test_simple_conversion_formula():
    assert acc.zcdp_to_dp_simple(0.5, 1e-5) == pytest.approx(0.5 + 2 * math.sqrt(0.5 * math.log(1e5)))


@pytest.mark.parametrize("rho", [0.01, 0.1, 0.5, 1.0, 4.0])
def test_cks_is_valid_and_tighter(rho):
    delta = 1e-5
    cks = acc.zcdp_to_dp(rho, delta, "cks")
    simple = acc.zcdp_to_dp(rho, delta, "simple")
    assert cks <= simple + 1e-12
    # the Gaussian mechanism with sigma = 1/sqrt(2 rho) is exactly rho-zCDP; any valid
    # conversion must be at least its exact epsilon
    exact = analytic_gaussian_eps(1 / math.sqrt(2 * rho), delta)
    assert cks >= exact - 1e-6
    assert cks <= 1.2 * exact + 0.05


def test_budget_inverse():
    for eps in [0.5, 1, 4, 8]:
        for method in ["simple", "cks"]:
            rho = acc.dp_to_zcdp_budget(eps, 1e-5, method)
            assert acc.zcdp_to_dp(rho, 1e-5, method) == pytest.approx(eps, rel=1e-6)


def test_plan_per_pick_values():
    # "What the bound implies" table of the plan (simple conversion, delta' = 1e-5)
    expected = {(4, 10): 0.49, (4, 50): 0.22, (8, 10): 0.92, (8, 50): 0.41}
    for (eps, k), v in expected.items():
        rho = acc.dp_to_zcdp_budget(eps, 1e-5, "simple")
        assert math.sqrt(8 * rho / k) == pytest.approx(v, abs=0.006)


@pytest.mark.parametrize("eps", [1, 2, 4, 8])
@pytest.mark.parametrize("k", [1, 5, 10, 50])
def test_calibration_meets_target(eps, k):
    e0 = acc.calibrate_eps0(eps, 1e-5, k)
    assert acc.em_k_picks_dp(e0, k, 1e-5) <= eps + 1e-6
    assert acc.calibrate_eps0(eps, 1e-5, k, "cks") >= acc.calibrate_eps0(eps, 1e-5, k, "simple") - 1e-12


def test_single_pick_uses_pure_dp():
    assert acc.calibrate_eps0(8, 1e-5, 1) == pytest.approx(8)


def test_gaussian_scores_sigma_scales_with_sqrt_n():
    s1 = acc.gaussian_scores_sigma(4, 1e-5, 100)
    s2 = acc.gaussian_scores_sigma(4, 1e-5, 400)
    assert s2 / s1 == pytest.approx(2.0)
