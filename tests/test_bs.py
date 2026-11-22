import numpy as np
from optionlib.bs import (
    bs_call, bs_put, bs_delta_call, bs_delta_put, bs_gamma, bs_vega,
    bs_theta_call, bs_theta_put, bs_rho_call, bs_rho_put, implied_vol,
)

S, K, r, T, sigma = 100.0, 100.0, 0.05, 1.0, 0.2


def test_reference_call():
    assert round(bs_call(S, K, r, T, sigma), 4) == 10.4506


def test_put_call_parity():
    c, p = bs_call(S, K, r, T, sigma), bs_put(S, K, r, T, sigma)
    assert np.isclose(c - p, S - K * np.exp(-r * T))


def test_implied_vol_roundtrip():
    price = bs_call(S, K, r, T, 0.2)
    assert np.isclose(implied_vol(price, S, K, r, T), 0.2)


def test_delta_call_finite_difference():
    h = 1e-4
    fd = (bs_call(S + h, K, r, T, sigma) - bs_call(S - h, K, r, T, sigma)) / (2 * h)
    assert np.isclose(bs_delta_call(S, K, r, T, sigma), fd, atol=1e-6)


def test_delta_put_finite_difference():
    h = 1e-4
    fd = (bs_put(S + h, K, r, T, sigma) - bs_put(S - h, K, r, T, sigma)) / (2 * h)
    assert np.isclose(bs_delta_put(S, K, r, T, sigma), fd, atol=1e-6)


def test_gamma_finite_difference():
    h = 1e-3
    fd = (bs_delta_call(S + h, K, r, T, sigma) - bs_delta_call(S - h, K, r, T, sigma)) / (2 * h)
    assert np.isclose(bs_gamma(S, K, r, T, sigma), fd, atol=1e-6)


def test_vega_finite_difference():
    h = 1e-5
    fd = (bs_call(S, K, r, T, sigma + h) - bs_call(S, K, r, T, sigma - h)) / (2 * h)
    assert np.isclose(bs_vega(S, K, r, T, sigma), fd, atol=1e-4)


def test_theta_finite_difference():
    # theta = -dV/dT
    h = 1e-5
    fd_c = -(bs_call(S, K, r, T + h, sigma) - bs_call(S, K, r, T - h, sigma)) / (2 * h)
    fd_p = -(bs_put(S, K, r, T + h, sigma) - bs_put(S, K, r, T - h, sigma)) / (2 * h)
    assert np.isclose(bs_theta_call(S, K, r, T, sigma), fd_c, atol=1e-4)
    assert np.isclose(bs_theta_put(S, K, r, T, sigma), fd_p, atol=1e-4)


def test_rho_finite_difference():
    h = 1e-6
    fd_c = (bs_call(S, K, r + h, T, sigma) - bs_call(S, K, r - h, T, sigma)) / (2 * h)
    fd_p = (bs_put(S, K, r + h, T, sigma) - bs_put(S, K, r - h, T, sigma)) / (2 * h)
    assert np.isclose(bs_rho_call(S, K, r, T, sigma), fd_c, atol=1e-4)
    assert np.isclose(bs_rho_put(S, K, r, T, sigma), fd_p, atol=1e-4)


def test_delta_parity():
    # delta_call - delta_put = 1
    assert np.isclose(bs_delta_call(S, K, r, T, sigma) - bs_delta_put(S, K, r, T, sigma), 1.0)