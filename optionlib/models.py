# GBM, Variance Gamma: char. functions, simulators
"""Models: characteristic functions of ln S_T and path simulators (GBM, Variance Gamma).

All characteristic functions are those of X = ln S_T under the risk-neutral measure:
    phi(u) = E[exp(i u ln S_T)]
so that phi(-i) = E[S_T] = S0 e^{rT} (martingale check).
"""
from typing import Callable

import numpy as np

CharFn = Callable[[complex], complex]


#GBM
def gbm_cf(S0: float, r: float, T: float, sigma: float) -> CharFn:
    """GBM: phi(u) = exp( i u (ln S0 + (r - sigma^2/2) T) - sigma^2 u^2 T / 2 )."""

    def phi(u):
        return np.exp(1j * u * (np.log(S0) + (r - 0.5 * sigma**2) * T)
                      - 0.5 * sigma**2 * u**2 * T)

    return phi


def simulate_gbm(S0: float, r: float, T: float, sigma: float, n: int,
                 seed: int | None = None) -> np.ndarray:
    """Terminal prices S_T = S0 exp((r - sigma^2/2) T + sigma sqrt(T) Z)."""
    Z = np.random.default_rng(seed).standard_normal(n)
    return S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)


#Variance Gamma
def vg_omega(sigma: float, theta: float, nu: float) -> float:
    """Martingale correction omega = (1/nu) ln(1 - theta nu - sigma^2 nu / 2).

    Requires 1 - theta nu - sigma^2 nu / 2 > 0.
    """
    arg = 1.0 - theta * nu - 0.5 * sigma**2 * nu
    if arg <= 0.0:
        raise ValueError("Need 1 - theta*nu - sigma^2*nu/2 > 0")
    return float(np.log(arg) / nu)


def vg_alpha_max(sigma: float, theta: float, nu: float) -> float:
    """Largest damping alpha keeping E[S_T^(alpha+1)] finite under VG.
    The moment E[S_T^a] is finite iff 1 - a theta nu - sigma^2 nu a^2 / 2 > 0, a = alpha + 1.
    """
    a_max = (-theta * nu + np.sqrt((theta * nu) ** 2 + 2.0 * sigma**2 * nu)) / (sigma**2 * nu)
    return float(a_max - 1.0)


def vg_cf(S0: float, r: float, T: float, sigma: float, theta: float, nu: float) -> CharFn:
    """Variance Gamma:
        phi(u) = exp( i u (ln S0 + (r + omega) T) ) * (1 - i u theta nu + sigma^2 nu u^2 / 2)^(-T/nu)

    The complex power is computed as exp(-(T/nu) * log(z)) with the principal branch.
    For u = v - i(alpha+1) with Re z > 0 (i.e. alpha below vg_alpha_max) z never
    crosses the negative real axis, so the principal branch is continuous in v.
    """
    omega = vg_omega(sigma, theta, nu)

    def phi(u):
        z = 1.0 - 1j * u * theta * nu + 0.5 * sigma**2 * nu * u**2
        return np.exp(1j * u * (np.log(S0) + (r + omega) * T) - (T / nu) * np.log(z))

    return phi


def simulate_vg(S0: float, r: float, T: float, sigma: float, theta: float, nu: float,
                n: int, seed: int | None = None) -> np.ndarray:
    """Terminal prices under VG via gamma time change.
    G ~ Gamma(shape=T/nu, scale=nu),  X = theta G + sigma sqrt(G) Z,
    S_T = S0 exp((r + omega) T + X).
    """
    rng = np.random.default_rng(seed)
    omega = vg_omega(sigma, theta, nu)
    G = rng.gamma(shape=T / nu, scale=nu, size=n)
    Z = rng.standard_normal(n)
    X = theta * G + sigma * np.sqrt(G) * Z
    return S0 * np.exp((r + omega) * T + X)