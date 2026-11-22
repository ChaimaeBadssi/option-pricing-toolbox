# Black-Scholes price, Greeks, implied vol for European call and put options.

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def _d1_d2(S: float, K: float, r: float, T: float, sigma: float) -> tuple[float, float]:
    """d1 = [ln(S/K) + (r + sigma^2/2) T] / (sigma sqrt(T)),  d2 = d1 - sigma sqrt(T)."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return d1, d1 - sigma * np.sqrt(T)


def bs_call(S: float, K: float, r: float, T: float, sigma: float) -> float:
    """Call = S N(d1) - K e^{-rT} N(d2)."""
    d1, d2 = _d1_d2(S, K, r, T, sigma)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put(S: float, K: float, r: float, T: float, sigma: float) -> float:
    """Put = K e^{-rT} N(-d2) - S N(-d1)."""
    d1, d2 = _d1_d2(S, K, r, T, sigma)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_delta_call(S: float, K: float, r: float, T: float, sigma: float) -> float:
    """Call delta = N(d1)."""
    d1, _ = _d1_d2(S, K, r, T, sigma)
    return norm.cdf(d1)

def bs_delta_put(S: float, K: float, r: float, T: float, sigma: float) -> float:
    """Put delta = N(d1) - 1."""
    d1, _ = _d1_d2(S, K, r, T, sigma)
    return norm.cdf(d1) - 1

def bs_gamma(S: float, K: float, r: float, T: float, sigma: float) -> float:
    """Gamma = phi(d1) / (S sigma sqrt(T)). same for put and call"""
    d1, _ = _d1_d2(S, K, r, T, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def bs_vega(S: float, K: float, r: float, T: float, sigma: float) -> float:
    """Vega = S phi(d1) sqrt(T). same for put and call"""
    d1, _ = _d1_d2(S, K, r, T, sigma)
    return S * norm.pdf(d1) * np.sqrt(T)

def bs_theta_call(S: float, K: float, r: float, T: float, sigma: float) -> float:
    """Call theta = -S phi(d1) sigma / (2 sqrt(T)) - r K e^{-rT} N(d2)."""
    d1, d2 = _d1_d2(S, K, r, T, sigma)
    return -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)


def bs_theta_put(S: float, K: float, r: float, T: float, sigma: float) -> float:
    """Put theta = -S phi(d1) sigma / (2 sqrt(T)) + r K e^{-rT} N(-d2).
    Convention: theta is the change in value per day,here we use as convention per year so divide by 365."""
    d1, d2 = _d1_d2(S, K, r, T, sigma)
    return -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)

def bs_rho_call(S: float, K: float, r: float, T: float, sigma: float) -> float:
    """Call rho = K T e^{-rT} N(d2)."""
    d1, d2 = _d1_d2(S, K, r, T, sigma)
    return K * T * np.exp(-r * T) * norm.cdf(d2)

def bs_rho_put(S: float, K: float, r: float, T: float, sigma: float) -> float:
    """Put rho = -K T e^{-rT} N(-d2)."""
    d1, d2 = _d1_d2(S, K, r, T, sigma)
    return -K * T * np.exp(-r * T) * norm.cdf(-d2)

def implied_vol(price, S, K, r, T, kind: str = "call") -> float:
    """Implied volatility by Brent root-finding in [1e-6, 5].
    kind: 'call' or 'put'."""
    f = bs_call if kind == "call" else bs_put
    return brentq(lambda s: f(S, K, r, T, s) - price, 1e-6, 5.0)