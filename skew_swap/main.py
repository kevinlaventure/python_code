"""KNS skew swap replication under SABR: initial strip, MC hedging, checks, plots.

    python main.py --path-id 7 --rebalance threshold --threshold 0.03 --rho-sim -0.5 --n-paths 2000
"""
import argparse
import os
import time

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

import analysis as an
from config import Config
from hedge import Variant
from plots import plot_distribution, plot_path, plot_weights, plotly_paths


class Report:
    def __init__(self):
        self.lines = []

    def __call__(self, s=""):
        s = s.to_string(float_format=lambda x: f"{x:+.4e}") if isinstance(s, (pd.DataFrame, pd.Series)) else s
        print(s)
        self.lines.append(s)

    def save(self, path):
        with open(path, "w") as f:
            f.write("\n".join(self.lines) + "\n")


def header(R, title):
    R("\n" + "=" * 78)
    R(title)
    R("=" * 78)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path-id", type=int, default=0)
    ap.add_argument("--rebalance", choices=an.REBALANCES, default="step")
    ap.add_argument("--threshold", type=float, default=0.03)
    ap.add_argument("--rho-sim", type=float, default=-0.5)
    ap.add_argument("--n-paths", type=int, default=2000)
    ap.add_argument("--delta-mode", choices=an.DELTAS, default="sticky_alpha")
    ap.add_argument("--bid-ask-vol", type=float, default=0.0, help="full bid-ask in vol (0.01 = 1 vol pt)")
    ap.add_argument("--out", default="output")
    ap.add_argument("--quick", action="store_true", help="skip the sanity and convergence re-runs")
    a = ap.parse_args()

    cfg = Config(path_id=a.path_id, rebalance=a.rebalance, threshold=a.threshold, rho_sim=a.rho_sim,
                 n_paths=a.n_paths, delta_mode=a.delta_mode, bid_ask_vol=a.bid_ask_vol, out_dir=a.out)
    os.makedirs(cfg.out_dir, exist_ok=True)
    R = Report()
    t0 = time.time()

    header(R, f"STEP 0 — initial KNS strip (SABR, rho_imp = {cfg.rho_imp:.2f}); ATM vol {an.atm_vol(cfg):.4f}, "
              f"E_Q[∫σ²dt] = {cfg.alpha0**2 * np.expm1(cfg.nu**2 * cfg.T) / cfg.nu**2:.6f}")
    R(an.initial_portfolio_table(cfg))
    plot_weights(cfg, os.path.join(cfg.out_dir, "step0_weights.png"))

    variants = [Variant(r, d) for d in an.DELTAS for r in an.REBALANCES]
    log_idx = [cfg.path_id] + [p for p in range(cfg.n_paths) if p != cfg.path_id][: cfg.n_log_paths - 1]
    header(R, f"STEP 1 — MC hedging: {cfg.n_paths} paths, {cfg.n_steps} steps, rho_sim = {cfg.rho_sim}, "
              f"rho_imp = {cfg.rho_imp}, threshold = {cfg.threshold:.1%}, bid-ask = {cfg.bid_ask_vol * 100:.2f} vol pts")
    res = an.simulate_and_run(cfg, variants, log_idx)
    R(an.benchmark_summary(res))
    R("\nFinal P&L (long strip / receive realised) and tracking error TE = P&L − (RS − S0):")
    R(an.variant_table(res))
    R("\nVega check (expected: step ≈ 0, threshold small, none slope ≈ 1):")
    R(an.vega_table(res, cfg.delta_mode))

    if not a.quick:
        R("\nSanity: rho_sim = rho_imp (MC measure = pricing model up to Hagan approximation error)")
        res_eq = an.simulate_and_run(cfg.with_(rho_sim=cfg.rho_imp), [Variant(r, cfg.delta_mode) for r in an.REBALANCES])
        R(an.benchmark_summary(res_eq))
        R(an.variant_table(res_eq))
        R("\nConvergence of tracking error vs n_steps (500 paths, config strike grid):")
        R(an.convergence_table(cfg))
        R("\nTracking error vs strike grid (step rebalance, 500 paths):")
        R(an.grid_table(cfg))

    primary = f"{cfg.rebalance}/{cfg.delta_mode}"
    plot_path(res, primary, 0, os.path.join(cfg.out_dir, f"path_{cfg.path_id}_{cfg.rebalance}.png"))
    plotly_paths(res, primary, 0, os.path.join(cfg.out_dir, f"paths_{cfg.rebalance}.html"))
    plot_distribution(res, [f"{r}/{cfg.delta_mode}" for r in an.REBALANCES],
                      os.path.join(cfg.out_dir, "distribution.png"))
    R(f"\nOutputs written to {os.path.abspath(cfg.out_dir)}  ({time.time() - t0:.0f}s)")
    R.save(os.path.join(cfg.out_dir, "report.txt"))


if __name__ == "__main__":
    main()
