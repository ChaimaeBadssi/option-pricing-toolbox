import numpy as np
from optionlib.bs import bs_call, bs_put, bs_delta_call
from optionlib.mc import (mc_plain, mc_antithetic, mc_control_variate,
                          mc_qmc, mc_delta_pathwise)

S, K, r, T, sigma = 100.0, 100.0, 0.05, 1.0, 0.2
REF = bs_call(S, K, r, T, sigma)
N = 100_000


def test_plain_within_4_se():
    res = mc_plain(S, K, r, T, sigma, N, seed=1)
    assert abs(res.price - REF) < 4 * res.se


def test_antithetic_within_4_se():
    res = mc_antithetic(S, K, r, T, sigma, N, seed=1)
    assert abs(res.price - REF) < 4 * res.se


def test_control_variate_within_4_se():
    res = mc_control_variate(S, K, r, T, sigma, N, seed=1)
    assert abs(res.price - REF) < 4 * res.se


def test_put_within_4_se():
    ref = bs_put(S, K, r, T, sigma)
    res = mc_control_variate(S, K, r, T, sigma, N, kind="put", seed=2)
    assert abs(res.price - ref) < 4 * res.se


def test_variance_reduction_lowers_se():
    plain = mc_plain(S, K, r, T, sigma, N, seed=3).se
    assert mc_antithetic(S, K, r, T, sigma, N, seed=3).se < plain
    assert mc_control_variate(S, K, r, T, sigma, N, seed=3).se < plain


def test_ci_coverage_over_seeds():
    # a 95% CI should contain the true price about 19 times out of 20
    hits = sum(mc_plain(S, K, r, T, sigma, 20_000, seed=s).ci[0] <= REF <= mc_plain(S, K, r, T, sigma, 20_000, seed=s).ci[1]
               for s in range(20))
    assert hits >= 15


def test_reproducible_with_seed():
    a = mc_plain(S, K, r, T, sigma, 1000, seed=42)
    b = mc_plain(S, K, r, T, sigma, 1000, seed=42)
    assert a.price == b.price


def test_qmc_close_to_bs():
    assert abs(mc_qmc(S, K, r, T, sigma, m=14, seed=0) - REF) < 2e-2


def test_pathwise_delta_within_4_se():
    res = mc_delta_pathwise(S, K, r, T, sigma, N, seed=5)
    assert abs(res.price - bs_delta_call(S, K, r, T, sigma)) < 4 * res.se