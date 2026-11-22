# CRR binomial tree
"""Cox-Ross-Rubinstein binomial tree for European options."""
import numpy as np


def crr_price(S: float, K: float, r: float, T: float, sigma: float,
              N: int, kind: str = "call") -> float:
    """Price a European option on an N-step CRR tree.

    dt = T/N, u = exp(sigma sqrt(dt)), d = 1/u, p = (exp(r dt) - d) / (u - d).
    Backward step: V = exp(-r dt) [p V_up + (1 - p) V_down].
    """
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    # Terminal nodes: j up-moves and N-j down-moves  ->  S u^(2j - N)
    j = np.arange(N + 1)
    ST = S * u ** (2 * j - N)
    V = np.maximum(ST - K, 0.0) if kind == "call" else np.maximum(K - ST, 0.0)

    # Roll back one time level per iteration, vectorized over all nodes.
    # V[j+1] is the "up" child of node j, V[j] the "down" child.
    for _ in range(N):
        V = disc * (p * V[1:] + (1.0 - p) * V[:-1])
    return float(V[0])


def crr_price_smoothed(S: float, K: float, r: float, T: float, sigma: float,
                       N: int, kind: str = "call") -> float:
    """Average of the N-step and (N+1)-step prices: damps the odd/even oscillation."""
    return 0.5 * (crr_price(S, K, r, T, sigma, N, kind)
                  + crr_price(S, K, r, T, sigma, N + 1, kind))