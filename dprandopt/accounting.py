"""Privacy accounting for DP-RandOpt.

Everything here is mechanism-agnostic arithmetic on privacy parameters:

* pure epsilon-DP composition,
* bounded-range (BR) mechanisms -> zero-concentrated DP (zCDP),
  following Cesar & Rogers (ALT 2021): an eps-BR mechanism is eps^2/8-zCDP,
  and the exponential mechanism (EM) with privacy parameter eps is eps-BR,
* conversion of zCDP / Renyi DP to (eps, delta)-DP, either with the classic
  bound of Bun & Steinke (2016) or the tighter bound of Canonne, Kamath &
  Steinke (2020, Prop. 12),
* calibration: the largest per-pick EM parameter eps0 such that K adaptive
  picks satisfy a target (eps, delta).

Conventions
-----------
``eps0`` is always the privacy parameter of ONE exponential-mechanism pick,
i.e. the pick is eps0-DP. How eps0 maps to the sampling temperature depends on
the adjacency notion and lives in :mod:`dprandopt.mechanisms`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import optimize

_ALPHAS = np.concatenate(
    [1.0 + np.logspace(-3, 0, 60)[:-1], np.linspace(2.0, 64.0, 249), np.logspace(np.log10(64.5), 4, 80)]
)


# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------

def zcdp_to_dp_simple(rho: float, delta: float) -> float:
    """Bun & Steinke (2016): rho-zCDP => (rho + 2 sqrt(rho ln(1/delta)), delta)-DP."""
    if rho <= 0:
        return 0.0
    return rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta))


def rdp_to_dp(rdp_fn, delta: float, alphas: np.ndarray = _ALPHAS) -> float:
    """Tight RDP -> (eps, delta) conversion of Canonne, Kamath & Steinke (2020).

    For a mechanism with Renyi divergence at most ``rdp_fn(alpha)`` at order
    ``alpha``, it is (eps, delta)-DP with

        eps = tau + log(1 - 1/alpha) - (log(delta) + log(alpha)) / (alpha - 1),

    minimised over alpha > 1.  This is the same formula used by Google's
    ``dp_accounting`` library.
    """
    a = np.asarray(alphas, dtype=float)
    tau = np.array([rdp_fn(x) for x in a], dtype=float)
    eps = tau + np.log1p(-1.0 / a) - (math.log(delta) + np.log(a)) / (a - 1.0)
    best = float(np.min(eps))
    return max(best, 0.0)


def zcdp_to_dp(rho: float, delta: float, method: str = "cks") -> float:
    """Convert rho-zCDP to (eps, delta)-DP.

    method="simple" uses Bun & Steinke; method="cks" uses the tighter RDP
    conversion (zCDP means RDP(alpha) <= rho * alpha for all alpha > 1).
    The returned value is never larger than the simple bound.
    """
    if rho <= 0:
        return 0.0
    simple = zcdp_to_dp_simple(rho, delta)
    if method == "simple":
        return simple
    if method != "cks":
        raise ValueError(f"unknown method {method!r}")
    return min(simple, rdp_to_dp(lambda a: rho * a, delta))


def dp_to_zcdp_budget(eps: float, delta: float, method: str = "cks") -> float:
    """Largest rho such that rho-zCDP converts to at most (eps, delta)-DP."""
    if eps <= 0:
        return 0.0
    f = lambda r: zcdp_to_dp(r, delta, method) - eps
    hi = 1.0
    while f(hi) < 0:
        hi *= 2.0
    return optimize.brentq(f, 1e-12, hi, xtol=1e-12, rtol=1e-10)


# ---------------------------------------------------------------------------
# Exponential mechanism accounting
# ---------------------------------------------------------------------------

def em_zcdp(eps0: float, k: int = 1) -> float:
    """zCDP of k adaptive EM picks, each eps0-DP (bounded range): k * eps0^2 / 8."""
    return k * eps0 ** 2 / 8.0


def em_k_picks_dp(eps0: float, k: int, delta: float, method: str = "cks") -> float:
    """(eps, delta)-DP of k adaptive eps0-EM picks: min of pure and BR-zCDP bounds."""
    pure = k * eps0
    if delta <= 0:
        return pure
    return min(pure, zcdp_to_dp(em_zcdp(eps0, k), delta, method))


def calibrate_eps0(eps: float, delta: float, k: int, method: str = "cks") -> float:
    """Largest per-pick eps0 such that k adaptive EM picks are (eps, delta)-DP.

    Uses the better of pure composition (k * eps0 <= eps) and BR-zCDP
    composition with the chosen conversion.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    pure = eps / k
    if delta <= 0:
        return pure
    rho = dp_to_zcdp_budget(eps, delta, method)
    br = math.sqrt(8.0 * rho / k)
    return max(pure, br)


# ---------------------------------------------------------------------------
# Gaussian mechanism on the full score vector (DP-ES-style "noisy scores")
# ---------------------------------------------------------------------------

def gaussian_scores_sigma(eps: float, delta: float, n_candidates: int, sensitivity: float = 1.0,
                          method: str = "cks") -> float:
    """Noise std for releasing all N sum-scores with the Gaussian mechanism.

    Changing one private example moves every candidate's sum-score by at most
    ``sensitivity``, so the L2 sensitivity of the score vector is
    ``sensitivity * sqrt(N)``. The Gaussian mechanism with std sigma is
    (Delta_2^2 / (2 sigma^2))-zCDP.
    """
    rho = dp_to_zcdp_budget(eps, delta, method)
    l2 = sensitivity * math.sqrt(n_candidates)
    return l2 / math.sqrt(2.0 * rho)


# ---------------------------------------------------------------------------
# Convenience record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SelectionBudget:
    """Privacy plan for one DP-RandOpt selection."""

    eps: float
    delta: float
    k: int
    design: str = "D1"
    method: str = "cks"

    @property
    def eps0(self) -> float:
        """Per-pick EM parameter for this design."""
        if self.design in ("D1",):
            return calibrate_eps0(self.eps, self.delta, self.k, self.method)
        if self.design in ("D2", "D3", "D7"):
            # D2: parallel composition (one pure pick per disjoint shard);
            # D3 and D7: a single exponential mechanism.
            return self.eps
        raise ValueError(f"unknown design {self.design!r}")

    def describe(self) -> str:
        return (f"{self.design}: target ({self.eps}, {self.delta})-DP, K={self.k}, "
                f"per-pick eps0={self.eps0:.4f} [{self.method}]")
