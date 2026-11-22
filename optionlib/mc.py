# Monte Carlo + variance reduction
"""Monte Carlo pricing of European options under GBM, with variance reduction.

S_T = S0 exp((r - sigma^2/2) T + sigma sqrt(T) Z),  Z ~ N(0, 1)
Price = e^{-rT} mean(payoff),  SE = std(discounted payoffs) / sqrt(n)

Cost convention: `n` is always the number of payoff evaluations, so methods can
be compared at equal cost (antithetic uses n // 2 pairs).
"""
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm, qmc


@dataclass(frozen=True)
class MCResult:
    price: float
    se: float

    @property
    def ci(self) -> tuple[float, float]:
        """95% confidence interval: price +/- 1.96 SE."""
        return (self.price - 1.96 * self.se, self.price + 1.96 * self.se)


def _terminal(S: float, r: float, T: float, sigma: float, Z: np.ndarray) -> np.ndarray:
    return S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)


def _payoff(ST: np.ndarray, K: float, kind: str) -> np.ndarray:
    return np.maximum(ST - K, 0.0) if kind == "call" else np.maximum(K - ST, 0.0)


def mc_plain(S, K, r, T, sigma, n, kind="call", seed=None) -> MCResult:
    """Plain Monte Carlo."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n)
    x = np.exp(-r * T) * _payoff(_terminal(S, r, T, sigma, Z), K, kind)
    return MCResult(float(x.mean()), float(x.std(ddof=1) / np.sqrt(n)))


def mc_antithetic(S, K, r, T, sigma, n, kind="call", seed=None) -> MCResult:
    """Antithetic variates: average the payoffs of Z and -Z (n // 2 pairs)."""
    rng = np.random.default_rng(seed)
    half = n // 2
    Z = rng.standard_normal(half)
    pay = 0.5 * (_payoff(_terminal(S, r, T, sigma, Z), K, kind)
                 + _payoff(_terminal(S, r, T, sigma, -Z), K, kind))
    x = np.exp(-r * T) * pay
    return MCResult(float(x.mean()), float(x.std(ddof=1) / np.sqrt(half)))


def mc_control_variate(S, K, r, T, sigma, n, kind="call", seed=None) -> MCResult:
    """Control variate Y = e^{-rT} S_T, whose expectation is exactly S0.

    Estimator: X - b (Y - S0), with b = Cov(X, Y) / Var(Y) estimated on the sample.
    (Estimating b from the same sample adds a bias of order 1/n, negligible here.)
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n)
    ST = _terminal(S, r, T, sigma, Z)
    X = np.exp(-r * T) * _payoff(ST, K, kind)
    Y = np.exp(-r * T) * ST
    C = np.cov(X, Y)
    b = C[0, 1] / C[1, 1]
    adj = X - b * (Y - S)
    return MCResult(float(adj.mean()), float(adj.std(ddof=1) / np.sqrt(n)))


def mc_qmc(S, K, r, T, sigma, m, kind="call", seed=None) -> float:
    """Randomized quasi-MC with a scrambled Sobol sequence, n = 2**m points.

    Returns only the price: the error estimate comes from repeating with
    different scramblings (seeds), not from a single run.
    """
    u = qmc.Sobol(d=1, scramble=True, seed=seed).random_base2(m).ravel()
    Z = norm.ppf(u)
    x = np.exp(-r * T) * _payoff(_terminal(S, r, T, sigma, Z), K, kind)
    return float(x.mean())


def mc_delta_pathwise(S, K, r, T, sigma, n, seed=None) -> MCResult:
    """Pathwise call delta: e^{-rT} 1{S_T > K} S_T / S0."""
    rng = np.random.default_rng(seed)
    ST = _terminal(S, r, T, sigma, rng.standard_normal(n))
    x = np.exp(-r * T) * (ST > K) * ST / S
    return MCResult(float(x.mean()), float(x.std(ddof=1) / np.sqrt(n)))