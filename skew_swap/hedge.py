"""Rebalancing engine for the KNS skew-swap replicating book.

Price/greek grids are computed once per time step for all paths (they only depend on the market
state F_t, sigma_t, t) and shared by every hedging variant. Books are vectorised across paths and
strikes; the loop is over time.

Book accounting (zero rates):
    value = sum(q * price of held option) + cash
    forwards are marked daily into cash (variation margin), so they carry no value of their own.
    P&L = option MTM + forward P&L - costs; parity relabels and option trades are value-neutral.
"""
from dataclasses import dataclass

import numpy as np

from config import Config
from sabr import sabr_call, hagan_vol, black_vega
from strip import (trap_dk, is_call_otm, otm_from_call, log_strip, entropy_strip,
                   kns_quantities, parity_relabel, G)


@dataclass(frozen=True)
class Variant:
    rebalance: str      # "step" | "threshold" | "none"
    delta_mode: str     # "sticky_alpha" | "bartlett"

    @property
    def name(self) -> str:
        return f"{self.rebalance}/{self.delta_mode}"


class Market:
    """Call-price grids and finite-difference greeks at one time step, shape (n_paths, nK)."""

    def __init__(self, cfg: Config, K, F, sig, tau, log_idx):
        self.F, self.sig, self.tau = F, sig, tau
        Fc, ac = F[:, None], sig[:, None]
        b, rho, nu = cfg.beta, cfg.rho_imp, cfg.nu
        self.C = sabr_call(Fc, K, tau, ac, b, rho, nu)
        n = len(F)
        if tau <= 0:  # expiry: intrinsic only, no greeks
            z = np.zeros_like(self.C)
            self.dC_dF = self.gamma = self.dC_da = self.bvega = z
            self.vanna = self.dC_drho = np.zeros((len(log_idx), len(K)))
            return
        hF, ha = cfg.fd_F * Fc, cfg.fd_alpha * ac
        Cu = sabr_call(Fc + hF, K, tau, ac, b, rho, nu)
        Cd = sabr_call(Fc - hF, K, tau, ac, b, rho, nu)
        self.dC_dF = (Cu - Cd) / (2 * hF)                         # sticky-alpha delta
        self.gamma = (Cu - 2 * self.C + Cd) / hF**2
        self.dC_da = (sabr_call(Fc, K, tau, ac + ha, b, rho, nu)
                      - sabr_call(Fc, K, tau, ac - ha, b, rho, nu)) / (2 * ha)
        if cfg.bid_ask_vol > 0:
            self.bvega = black_vega(Fc, K, tau, hagan_vol(K, Fc, tau, ac, b, rho, nu))
        else:
            self.bvega = np.zeros((n, len(K)))
        # extra greeks only for the logged paths (plots)
        Fl, al, hFl, hal = Fc[log_idx], ac[log_idx], hF[log_idx], ha[log_idx]
        c = lambda f, a, r=rho: sabr_call(f, K, tau, a, b, r, nu)
        self.vanna = (c(Fl + hFl, al + hal) - c(Fl - hFl, al + hal)
                      - c(Fl + hFl, al - hal) + c(Fl - hFl, al - hal)) / (4 * hFl * hal)
        self.dC_drho = c(Fl, al, rho + cfg.fd_rho) - self.C[log_idx]   # price change per +fd_rho


class Book:
    def __init__(self, var: Variant, cfg: Config, K, log_idx):
        self.var, self.cfg, self.K, self.log_idx = var, cfg, K, np.asarray(log_idx)
        n, m, nK, nl = cfg.n_paths, cfg.n_steps + 1, len(K), len(log_idx)
        rec = lambda: np.zeros((n, m))
        self.value, self.opt_mtm, self.fwd_pnl, self.cost = rec(), rec(), rec(), rec()
        self.n_fwd, self.hedge_trade, self.parity_trade = rec(), rec(), rec()
        self.vega, self.cash_gamma, self.delta_pre = rec(), rec(), rec()
        self.rebalanced = np.zeros((n, m), dtype=bool)
        self.opt_trades = np.zeros((nl, m, nK))       # strip trades in OTM-option units (logged paths)
        self.opt_is_call = np.zeros((nl, m, nK), dtype=bool)
        self.vanna, self.dvdrho = np.zeros((nl, m)), np.zeros((nl, m))
        for a in (self.vega, self.cash_gamma, self.vanna, self.dvdrho):
            a[..., -1] = np.nan

    def _held_prices(self, mkt):
        return np.where(self.is_call, mkt.C, mkt.C - (mkt.F[:, None] - self.K))

    def step(self, i: int, mkt: Market):
        cfg, K, var = self.cfg, self.K, self.var
        F = mkt.F
        if i == 0:
            self.q = np.zeros_like(mkt.C)
            self.is_call = is_call_otm(K, F[:, None])
            self.cash = np.zeros(len(F))
            self.fwd = np.zeros(len(F))
            self.F_last = F.copy()
            self.F_prev = F.copy()
            rebal = np.ones(len(F), dtype=bool)          # initial strip purchase
        else:
            val_pre = np.sum(self.q * self._held_prices(mkt), axis=1)
            self.opt_mtm[:, i] = val_pre - self.val_post
            self.fwd_pnl[:, i] = self.fwd * (F - self.F_prev)
            self.cash += self.fwd_pnl[:, i]
            self.is_call, ptrade, pcash = parity_relabel(self.q, K, self.is_call, F[:, None])
            self.fwd += ptrade
            self.cash += pcash
            self.parity_trade[:, i] = ptrade
            if i == cfg.n_steps:
                rebal = np.zeros(len(F), dtype=bool)
            elif var.rebalance == "step":
                rebal = np.ones(len(F), dtype=bool)
            elif var.rebalance == "threshold":
                rebal = np.abs(F / self.F_last - 1.0) > cfg.threshold
            else:
                rebal = np.zeros(len(F), dtype=bool)

        if i < cfg.n_steps:
            # option rebalance: move to the strip centred on the current forward
            trade = np.where(rebal[:, None], kns_quantities(K, F[:, None]) - self.q, 0.0)
            prices = self._held_prices(mkt)
            cost = np.sum(np.abs(trade) * mkt.bvega, axis=1) * 0.5 * cfg.bid_ask_vol
            self.cash -= np.sum(trade * prices, axis=1) + cost
            self.cost[:, i] = cost
            self.q = self.q + trade
            self.F_last = np.where(rebal, F, self.F_last)
            self.rebalanced[:, i] = rebal
            self.opt_trades[:, i] = trade[self.log_idx]

            # forward delta hedge to zero total book delta
            opt_delta = np.sum(self.q * (mkt.dC_dF - (~self.is_call)), axis=1)
            vega = np.sum(self.q * mkt.dC_da, axis=1)
            if var.delta_mode == "bartlett":   # dalpha ~ rho nu / F^beta dF under the pricing model
                opt_delta = opt_delta + vega * cfg.rho_imp * cfg.nu / F**cfg.beta
            self.delta_pre[:, i] = opt_delta + self.fwd
            self.hedge_trade[:, i] = -self.delta_pre[:, i]
            self.fwd = -opt_delta

            self.vega[:, i] = vega
            self.cash_gamma[:, i] = np.sum(self.q * mkt.gamma, axis=1) * F**2
            ql = self.q[self.log_idx]
            self.vanna[:, i] = np.sum(ql * mkt.vanna, axis=1)
            self.dvdrho[:, i] = np.sum(ql * mkt.dC_drho, axis=1)

        self.opt_is_call[:, i] = self.is_call[self.log_idx]
        self.val_post = np.sum(self.q * self._held_prices(mkt), axis=1)
        self.value[:, i] = self.val_post + self.cash
        self.n_fwd[:, i] = self.fwd
        self.F_prev = F.copy()

    def cum_pnl(self):
        return {"option MTM": np.cumsum(self.opt_mtm, 1),
                "forward hedge": np.cumsum(self.fwd_pnl, 1),
                "costs": -np.cumsum(self.cost, 1)}


def run(cfg: Config, F, sig, variants, log_idx, progress=None):
    """Run all variants along simulated paths. Returns dict with books and per-path strip values.

    progress: optional iterable wrapper over time steps, e.g. tqdm.
    """
    K = cfg.strikes
    dk = trap_dk(K)
    n, m = F.shape
    books = {v.name: Book(v, cfg, K, log_idx) for v in variants}
    vE, vL, vE_vega, vL_vega = (np.zeros((n, m)) for _ in range(4))
    steps = enumerate(cfg.T - cfg.times)
    if progress is not None:
        steps = progress(steps, total=m, desc=f"{n} paths")
    for i, tau in steps:
        mkt = Market(cfg, K, F[:, i], sig[:, i], max(tau, 0.0), log_idx)
        otm = otm_from_call(mkt.C, K, F[:, i, None])
        vE[:, i] = entropy_strip(otm, K, F[:, i])
        vL[:, i] = log_strip(otm, K)
        vE_vega[:, i] = 2 * np.sum(mkt.dC_da * dk / K, axis=1) / F[:, i]
        vL_vega[:, i] = 2 * np.sum(mkt.dC_da * dk / K**2, axis=1)
        for b in books.values():
            b.step(i, mkt)

    df = np.diff(np.log(F), axis=1)
    rs_inc = 3 * np.diff(vE, axis=1) * np.expm1(df) + 6 * G(df)
    RS = np.concatenate([np.zeros((n, 1)), np.cumsum(rs_inc, 1)], axis=1)
    S0 = 3 * (vE[:, 0] - vL[:, 0])
    # running benchmark: realized leg to date + implied skew of the remaining period - initial implied
    bench = RS + 3 * (vE - vL) - S0[:, None]
    return dict(cfg=cfg, K=K, F=F, sig=sig, books=books, vE=vE, vL=vL, vE_vega=vE_vega,
                vL_vega=vL_vega, RS=RS, S0=S0, bench=bench, log_idx=np.asarray(log_idx))


def run_path(res, p, variants=None):
    """Re-run the engine on path p of `res` with full per-strike logs (paths are independent rows,
    so the result is identical to that row of the full run). Used by the path explorer."""
    cfg = res["cfg"].with_(n_paths=1)
    variants = variants or [b.var for b in res["books"].values()]
    out = run(cfg, res["F"][p:p + 1], res["sig"][p:p + 1], variants, [0])
    out["path_ids"] = [p]
    return out
