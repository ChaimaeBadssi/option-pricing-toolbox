import numpy as np
import pytest

from optionlib.models import (gbm_cf, vg_cf, vg_omega, vg_alpha_max,
                              simulate_gbm, simulate_vg)

S0, r, T, sigma = 100.0, 0.05, 1.0, 0.2
THETA, NU = -0.1, 0.2


def test_cf_at_zero_is_one():
    assert np.isclose(gbm_cf(S0, r, T, sigma)(0.0), 1.0)
    assert np.isclose(vg_cf(S0, r, T, sigma, THETA, NU)(0.0), 1.0)


def test_gbm_cf_martingale():
    # phi(-i) = E[S_T] = S0 e^{rT}
    assert np.isclose(gbm_cf(S0, r, T, sigma)(-1j), S0 * np.exp(r * T))


def test_vg_cf_martingale():
    # to catche a forgotten or wrong omega.
    assert np.isclose(vg_cf(S0, r, T, sigma, THETA, NU)(-1j), S0 * np.exp(r * T))


def test_vg_omega_condition():
    assert np.isfinite(vg_omega(0.2, -10.0, 1.0))   # 1 + 10 - 0.02 > 0: fine
    with pytest.raises(ValueError):
        vg_omega(0.2, 10.0, 1.0)                    # 1 - 10 - 0.02 < 0: invalid


def test_vg_alpha_max_is_the_moment_limit():
    a = vg_alpha_max(0.2, THETA, NU)
    def bound(alpha):
        A = alpha + 1.0
        return 1.0 - A * THETA * NU - 0.5 * 0.2**2 * NU * A**2
    assert bound(a - 1e-6) > 0 > bound(a + 1e-6)


def test_gbm_simulator_is_martingale():
    ST = simulate_gbm(S0, r, T, sigma, 400_000, seed=1)
    se = ST.std(ddof=1) / np.sqrt(ST.size)
    assert abs(ST.mean() - S0 * np.exp(r * T)) < 4 * se


def test_vg_simulator_is_martingale():
    ST = simulate_vg(S0, r, T, sigma, THETA, NU, 400_000, seed=1)
    se = ST.std(ddof=1) / np.sqrt(ST.size)
    assert abs(ST.mean() - S0 * np.exp(r * T)) < 4 * se