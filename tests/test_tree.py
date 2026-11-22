import numpy as np
from optionlib.bs import bs_call, bs_put
from optionlib.tree import crr_price, crr_price_smoothed

S, K, r, T, sigma = 100.0, 100.0, 0.05, 1.0, 0.2


def _slow_tree(S, K, r, T, sigma, N):
    """Reference: plain nested-loop backward induction (call).
    our code has a complexity of O(N) while this one has O(N^2) and is much slower, but easier to read."""
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt)); d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    V = [max(S * u**j * d**(N - j) - K, 0.0) for j in range(N + 1)]
    for n in range(N, 0, -1):
        V = [np.exp(-r * dt) * (p * V[j + 1] + (1 - p) * V[j]) for j in range(n)]
    return V[0]


def test_matches_slow_loop():
    assert np.isclose(crr_price(S, K, r, T, sigma, 50), _slow_tree(S, K, r, T, sigma, 50))


def test_converges_to_bs():
    assert abs(crr_price(S, K, r, T, sigma, 1000) - bs_call(S, K, r, T, sigma)) < 1e-2


def test_put_converges_to_bs():
    assert abs(crr_price(S, K, r, T, sigma, 1000, "put") - bs_put(S, K, r, T, sigma)) < 1e-2


def test_put_call_parity_on_tree():
    c = crr_price(S, K, r, T, sigma, 200, "call")
    p = crr_price(S, K, r, T, sigma, 200, "put")
    assert np.isclose(c - p, S - K * np.exp(-r * T))


def test_error_decreases_with_N():
    ref = bs_call(S, K, r, T, sigma)
    assert abs(crr_price(S, K, r, T, sigma, 500) - ref) < abs(crr_price(S, K, r, T, sigma, 10) - ref)


def test_smoothed_beats_plain():
    ref = bs_call(S, K, r, T, sigma)
    assert abs(crr_price_smoothed(S, K, r, T, sigma, 101) - ref) < abs(crr_price(S, K, r, T, sigma, 101) - ref)