# Carr-Madan
"""Carr-Madan Fourier pricing of European calls (and puts by parity).

With k = ln K and damping alpha > 0, c_alpha(k) = e^{alpha k} C(k) is integrable and

    C(K) = (e^{-alpha k} / pi) * Int_0^inf Re[ e^{-i v k} psi(v) ] dv
    psi(v) = e^{-rT} phi(v - (alpha+1) i) / (alpha^2 + alpha - v^2 + i (2 alpha + 1) v)

where phi is the characteristic function of ln S_T.
"""
from typing import Callable

import numpy as np
from scipy.integrate import quad

CharFn = Callable[[complex], complex]


def _psi(cf: CharFn, v, r: float, T: float, alpha: float):
    """Damped transform psi(v) of the call price (v may be a scalar or an array)."""
    return (np.exp(-r * T) * cf(v - (alpha + 1.0) * 1j)
            / (alpha**2 + alpha - v**2 + 1j * (2.0 * alpha + 1.0) * v))


def fourier_call(cf: CharFn, K: float, r: float, T: float, alpha: float = 1.5,
                 v_max: float = 200.0, n_points: int | None = None) -> float:
    """Carr-Madan call price.
    n_points=None : adaptive integration (scipy.integrate.quad) on [0, v_max].
    n_points=int  : fixed Gauss-Legendre rule with n_points nodes on [0, v_max]
                    (used in the alpha / v_max / N experiments).
    """
    k = np.log(K)

    if n_points is None:
        def integrand(v: float) -> float:
            return float((np.exp(-1j * v * k) * _psi(cf, v, r, T, alpha)).real)

        val, _ = quad(integrand, 0.0, v_max, limit=500, epsabs=1e-12, epsrel=1e-12)
    else:
        x, w = np.polynomial.legendre.leggauss(n_points)
        v = 0.5 * v_max * (x + 1.0)
        w = 0.5 * v_max * w
        val = float(np.sum(w * (np.exp(-1j * v * k) * _psi(cf, v, r, T, alpha)).real))

    return float(np.exp(-alpha * k) / np.pi * val)


def fourier_put(cf: CharFn, S0: float, K: float, r: float, T: float, alpha: float = 1.5,
                v_max: float = 200.0, n_points: int | None = None) -> float:
    """Put by put-call parity: P = C - S0 + K e^{-rT}."""
    c = fourier_call(cf, K, r, T, alpha, v_max, n_points)
    return c - S0 + K * np.exp(-r * T)