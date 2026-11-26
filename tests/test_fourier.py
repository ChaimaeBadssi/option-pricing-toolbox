import numpy as np
from optionlib.bs import bs_call, bs_put
from optionlib.fourier import fourier_call, fourier_put
from optionlib.models import gbm_cf, vg_cf, simulate_vg

S0, K, r, T, sigma = 100.0, 100.0, 0.05, 1.0, 0.2
THETA, NU = -0.1, 0.2


def test_gbm_matches_black_scholes():
    price = fourier_call(gbm_cf(S0, r, T, sigma), K, r, T, alpha=1.5)
    assert abs(price - bs_call(S0, K, r, T, sigma)) < 1e-8


def test_gbm_matches_black_scholes_several_strikes():
    cf = gbm_cf(S0, r, T, sigma)
    for strike in (70.0, 90.0, 110.0, 140.0):
        assert abs(fourier_call(cf, strike, r, T) - bs_call(S0, strike, r, T, sigma)) < 1e-8


def test_gbm_put_by_parity():
    price = fourier_put(gbm_cf(S0, r, T, sigma), S0, K, r, T)
    assert abs(price - bs_put(S0, K, r, T, sigma)) < 1e-8


def test_price_does_not_depend_on_alpha():
    cf = gbm_cf(S0, r, T, sigma)
    prices = [fourier_call(cf, K, r, T, alpha=a) for a in (0.5, 1.5, 3.0)]
    assert max(prices) - min(prices) < 1e-7


def test_vg_alpha_independence():
    cf = vg_cf(S0, r, T, sigma, THETA, NU)
    p1 = fourier_call(cf, K, r, T, alpha=1.0)
    p2 = fourier_call(cf, K, r, T, alpha=2.0)
    assert abs(p1 - p2) < 1e-6


def test_vg_within_mc_confidence_interval():
    price = fourier_call(vg_cf(S0, r, T, sigma, THETA, NU), K, r, T, alpha=1.0)
    ST = simulate_vg(S0, r, T, sigma, THETA, NU, 1_000_000, seed=7)
    x = np.exp(-r * T) * np.maximum(ST - K, 0.0)
    mc, se = x.mean(), x.std(ddof=1) / np.sqrt(x.size)
    assert abs(price - mc) < 3 * se


def test_fixed_rule_converges_to_quad():
    cf = gbm_cf(S0, r, T, sigma)
    exact = fourier_call(cf, K, r, T)
    assert abs(fourier_call(cf, K, r, T, v_max=100.0, n_points=256) - exact) < 1e-8