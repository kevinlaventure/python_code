"""Tables of results and checks (pandas), plus a disk cache for simulation runs.

Shared by main.py (printed report) and kns_skew_swap.ipynb (inline display).
"""
import hashlib
import os
import pickle

import numpy as np
import pandas as pd

from config import Config
from hedge import Variant, run
from sabr import hagan_vol, sabr_call
from simulate import realized_spot_vol_corr, simulate_paths
from strip import entropy_strip, kns_quantities, log_strip, otm_from_call

REBALANCES = ("step", "threshold", "none")
DELTAS = ("sticky_alpha", "bartlett")


# ----------------------------------------------------------------------------- cache
def cached(tag, key, fn, cache_dir="cache", refresh=False):
    """Return fn(), pickled under cache_dir/<tag>_<hash(key)>.pkl; recomputed if missing or refresh."""
    h = hashlib.sha1(repr(key).encode()).hexdigest()[:12]
    path = os.path.join(cache_dir, f"{tag}_{h}.pkl")
    if not refresh and os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    out = fn()
    os.makedirs(cache_dir, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    return out


def simulate_and_run(cfg: Config, variants, log_idx=(0,), progress=None):
    F, sig = simulate_paths(cfg)
    return run(cfg, F, sig, list(variants), list(log_idx), progress=progress)


# ----------------------------------------------------------------------------- step 0
def strip_values(cfg: Config, K):
    otm = otm_from_call(sabr_call(cfg.F0, K, cfg.T, cfg.alpha0, cfg.beta, cfg.rho_imp, cfg.nu), K, cfg.F0)
    vL, vE = log_strip(otm, K), entropy_strip(otm, K, cfg.F0)
    return dict(vL=vL, vE=vE, skew=3 * (vE - vL), strip_value=np.sum(kns_quantities(K, cfg.F0) * otm))


def initial_portfolio_table(cfg: Config) -> pd.DataFrame:
    """v^L, v^E, 3(v^E - v^L) and sum(q * OTM) on the config grid and on wider grids (truncation check)."""
    grids = {f"config {cfg.K_min:g}..{cfg.K_max:g} step {cfg.K_step:g}": cfg.strikes,
             "wide 5..1000 step 1": np.arange(5.0, 1000.5, 1.0),
             "wide 1..3000 step 0.25": np.arange(1.0, 3000.01, 0.25)}
    df = pd.DataFrame({k: strip_values(cfg, K) for k, K in grids.items()}).T
    df.columns = ["v^L", "v^E", "3(v^E − v^L)", "Σ q·OTM"]
    ref = df.iloc[-1]
    df["skew error vs widest"] = df["3(v^E − v^L)"] - ref["3(v^E − v^L)"]
    df["skew error %"] = 100 * df["skew error vs widest"] / ref["3(v^E − v^L)"]
    return df


def atm_vol(cfg: Config) -> float:
    return float(hagan_vol(cfg.F0, cfg.F0, cfg.T, cfg.alpha0, cfg.beta, cfg.rho_imp, cfg.nu))


# ----------------------------------------------------------------------------- MC tables
def variant_table(res) -> pd.DataFrame:
    """Final P&L and tracking error TE = P&L − (RS − S0) per variant."""
    target = res["RS"][:, -1] - res["S0"]
    corr = realized_spot_vol_corr(res["F"], res["sig"])
    n = len(target)
    rows = {}
    for name, b in res["books"].items():
        p = b.value[:, -1]
        te = p - target
        rows[name] = {"mean P&L": p.mean(), "s.e.": p.std() / np.sqrt(n), "std P&L": p.std(),
                      "TE mean": te.mean(), "TE std": te.std(),
                      "rebalances/path": b.rebalanced[:, 1:].sum(1).mean(),
                      "corr(P&L, realised ρ)": np.corrcoef(p, corr)[0, 1]}
    return pd.DataFrame(rows).T


def benchmark_summary(res) -> pd.Series:
    n = res["F"].shape[0]
    rs = res["RS"][:, -1]
    corr = realized_spot_vol_corr(res["F"], res["sig"])
    return pd.Series({"S0 = 3(vE−vL)_0": res["S0"][0], "mean RS": rs.mean(), "s.e. RS": rs.std() / np.sqrt(n),
                      "mean RS − S0": rs.mean() - res["S0"][0],
                      "mean realised spot-vol corr": corr.mean(), "std realised spot-vol corr": corr.std()})


def vega_table(res, delta_mode) -> pd.DataFrame:
    """Book vega in units of dv^E/dα, minus the fresh strip's own skew vega 3(dvE − dvL)/dvE,
    regressed on 3(F_t/F0 − 1). Expected: step ≈ 0, threshold small, none slope ≈ 1."""
    cfg = res["cfg"]
    live = slice(0, -1)
    x = (3 * (res["F"][:, live] / cfg.F0 - 1)).ravel()
    vE_v, vL_v = res["vE_vega"][:, live], res["vL_vega"][:, live]
    own = 3 * (vE_v - vL_v) / vE_v
    A = np.column_stack([np.ones(x.size), x])
    rows = {}
    for rb in REBALANCES:
        name = f"{rb}/{delta_mode}"
        if name not in res["books"]:
            continue
        excess = (res["books"][name].vega[:, live] / vE_v - own).ravel()
        (c0, c1), *_ = np.linalg.lstsq(A, excess, rcond=None)
        r2 = 1 - np.var(excess - A @ [c0, c1]) / max(np.var(excess), 1e-300)
        rows[name] = {"intercept": c0, "slope on 3(F_t/F0−1)": c1, "R²": r2, "std(excess vega)": excess.std(),
                      "std 3(F_t/F0−1)": x.std()}
    return pd.DataFrame(rows).T


def convergence_table(cfg: Config, steps=(63, 126, 252), n_paths=500, progress=None) -> pd.DataFrame:
    """Tracking-error std vs n_steps for each rebalance mode (config delta convention)."""
    variants = [Variant(r, cfg.delta_mode) for r in REBALANCES]
    rows = {}
    for m in steps:
        c = cfg.with_(n_steps=m, n_paths=n_paths)
        res = simulate_and_run(c, variants, progress=progress)
        tgt = res["RS"][:, -1] - res["S0"]
        rows[m] = {f"{b.var.rebalance} TE std": np.std(b.value[:, -1] - tgt) for b in res["books"].values()}
    df = pd.DataFrame(rows).T
    df.index.name = "n_steps"
    return df


def grid_table(cfg: Config, n_paths=500, progress=None) -> pd.DataFrame:
    """Tracking error of step rebalancing vs strike grid: isolates truncation/discretisation."""
    rows = {}
    for kw in (dict(), dict(K_min=5.0, K_max=1000.0), dict(K_min=5.0, K_max=1000.0, K_step=0.25)):
        c = cfg.with_(n_paths=n_paths, **kw)
        res = simulate_and_run(c, [Variant("step", cfg.delta_mode)], progress=progress)
        te = next(iter(res["books"].values())).value[:, -1] - (res["RS"][:, -1] - res["S0"])
        rows[f"K = {c.K_min:g}..{c.K_max:g} step {c.K_step:g}"] = {"TE mean": te.mean(), "TE std": te.std()}
    return pd.DataFrame(rows).T
