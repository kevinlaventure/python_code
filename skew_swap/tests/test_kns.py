import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from hedge import Variant, run, run_path
from sabr import black_call, hagan_vol, sabr_call, sabr_price
from simulate import simulate_paths
from strip import (G, entropy_strip, kns_quantities, log_strip, otm_from_call, parity_relabel,
                   recenter_trade)

CFG = Config()
K = CFG.strikes


def test_put_call_parity():
    F, T, a = 97.3, 0.37, 0.23
    c = sabr_price(F, K, T, a, 1.0, -0.7, 0.8, True)
    p = sabr_price(F, K, T, a, 1.0, -0.7, 0.8, False)
    np.testing.assert_allclose(c - p, F - K, atol=1e-10)
    assert np.all(p >= -1e-12) and np.all(c >= -1e-12)


def test_hagan_atm_and_lognormal_limit():
    # nu = 0, beta = 1: flat vol alpha
    np.testing.assert_allclose(hagan_vol(K, 100.0, 0.5, 0.2, 1.0, -0.7, 0.0), 0.2, atol=1e-12)
    # smooth through ATM (small-z branch)
    v = hagan_vol(100.0 * np.exp([-1e-9, 0.0, 1e-9]), 100.0, 0.5, 0.2, 1.0, -0.7, 0.8)
    assert np.ptp(v) < 1e-8


def test_vE_equals_vL_when_nu_zero():
    F = 100.0
    otm = otm_from_call(sabr_call(F, K, 0.5, 0.2, 1.0, -0.7, 0.0), K, F)
    # Black-Scholes: both equal sigma^2 T up to strike truncation / discretisation
    vL, vE = log_strip(otm, K), entropy_strip(otm, K, F)
    assert abs(vE - vL) < 1e-5
    assert abs(vL - 0.2**2 * 0.5) < 2e-4


def test_strip_value_is_3_vE_minus_vL():
    for F in (80.0, 100.0, 123.4):
        otm = otm_from_call(sabr_call(F, K, 0.5, 0.2, 1.0, -0.7, 0.8), K, F)
        val = np.sum(kns_quantities(K, F) * otm)
        np.testing.assert_allclose(val, 3 * (entropy_strip(otm, K, F) - log_strip(otm, K)), rtol=1e-12)
    assert val < 0  # negative implied skew with rho = -0.7


@pytest.mark.parametrize("F_old, F_new", [(100.0, 104.0), (100.0, 95.0), (60.0, 61.0)])
def test_recenter_trade_one_signed(F_old, F_new):
    tr = recenter_trade(K, F_new, F_old)
    assert np.all(np.sign(tr) == np.sign(1 / F_new - 1 / F_old))
    np.testing.assert_allclose(tr, kns_quantities(K, F_new) - kns_quantities(K, F_old), atol=1e-15)


def test_parity_relabel_value_neutral():
    F_old, F_new, T, a = 100.0, 103.5, 0.3, 0.2
    q = kns_quantities(K, F_old)[None, :]
    was_call = (K >= F_old)[None, :]
    C = sabr_call(F_new, K, T, a, 1.0, -0.7, 0.8)[None, :]
    held = lambda ic: np.sum(q * np.where(ic, C, C - (F_new - K)))
    now_call, fwd, cash = parity_relabel(q, K, was_call, np.array([[F_new]]))
    np.testing.assert_allclose(held(was_call), held(now_call) + cash[0], atol=1e-15)
    # the forward leg exactly offsets the delta change of the relabel (call->put delta drops by 1 per unit)
    changed = now_call != was_call
    np.testing.assert_allclose(fwd[0], np.sum(q[changed]))
    assert np.all(now_call == (K >= F_new))


def test_G_expansion():
    x = np.array([1e-3, -2e-3, 5e-3])
    np.testing.assert_allclose(G(x), x**3 / 6 + x**4 / 12, rtol=1e-4)


def test_black_intrinsic_at_expiry():
    np.testing.assert_allclose(black_call(100.0, K, 0.0, 0.2), np.maximum(100.0 - K, 0.0))


def test_hedged_book_tracks_realised_skew_on_wide_grid():
    """With a wide/fine strike grid, step rebalancing + sticky-alpha delta replicates RS - S0."""
    cfg = Config(n_paths=40, n_steps=40, K_min=5.0, K_max=1000.0, K_step=0.5)
    F, s = simulate_paths(cfg)
    res = run(cfg, F, s, [Variant("step", "sticky_alpha")], [0])
    b = res["books"]["step/sticky_alpha"]
    te = b.value[:, -1] - (res["RS"][:, -1] - res["S0"])
    assert np.max(np.abs(te)) < 1e-5 < np.std(b.value[:, -1])
    # accounting: value = sum of P&L components
    np.testing.assert_allclose(b.value[:, -1], b.opt_mtm.sum(1) + b.fwd_pnl.sum(1) - b.cost.sum(1), atol=1e-14)


def test_run_path_reproduces_full_run_row():
    """The notebook explorer re-runs one path; it must match that row of the full run exactly."""
    cfg = Config(n_paths=12, n_steps=20)
    F, s = simulate_paths(cfg)
    variants = [Variant(r, "sticky_alpha") for r in ("step", "threshold", "none")]
    res = run(cfg, F, s, variants, [5])
    one = run_path(res, 5)
    assert one["path_ids"] == [5]
    for name, b in res["books"].items():
        o = one["books"][name]
        np.testing.assert_array_equal(o.value[0], b.value[5])
        np.testing.assert_array_equal(o.opt_trades[0], b.opt_trades[0])
        np.testing.assert_array_equal(o.dvdrho[0], b.dvdrho[0])
