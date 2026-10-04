import warnings
from typing import Dict, List, NamedTuple, Optional, Sequence, Union

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.optimize import nnls

from .pricing_model import BSMModel


Floors = Union[List[str], Dict[str, float]]


class ReductionResult(NamedTuple):
    weights: pd.Series        # non-zero multipliers of the original positions (full book = 1)
    r2: float                 # fit of X @ weights to y
    risk_check: pd.DataFrame  # full / reduced / diff / ok per risk metric


def constrained_lasso(
    X: pd.DataFrame, y: pd.Series, risk: Optional[pd.DataFrame], max_positions: int,
    floors: Optional[Floors] = None, bands: Optional[Dict[str, float]] = None,
    eps_scale: float = 0.5, n_iter: int = 10, n_grid: int = 30, n_refine: int = 15, alpha_max: float = 1.0,
) -> ReductionResult:
    """
    Selects at most max_positions positions whose weighted sum tracks y, under risk constraints vs the full book.

    Iteratively reweighted long-only lasso (Candes, Wakin & Boyd 2008), no least squares refit: each strike's penalty
    is 1 / (w_j + eps) from the previous fit, eps = eps_scale * mean active weight, and the lasso alpha is the best
    fit with at most max_positions positions on a log grid (refined around the best point).
    n_iter = 1 is the standard lasso (single pass, same penalty on every strike). The risk constraints then keep more
    than max_positions strikes active at the top of the default grid, so start it higher with alpha_max (e.g. 100).

    Args:
        X: (n_obs, n_pos) contribution of each position (e.g. hedged P&L per path). The full book is weights = 1.
        y: (n_obs,) target (e.g. the full book, X.sum(axis=1)).
        risk: (n_pos, n_metrics) risk of each position, indexed like X.columns. Full book risk = risk.sum().
              None for an unconstrained fit.
        max_positions: maximum number of positions kept.
        floors: metrics constrained to reduced >= full, as a list, or {metric: ratio} for reduced >= ratio * full.
        bands: {metric: tol} constraining |reduced - full| <= tol * |full|.
        alpha_max: start of the alpha grid, as a multiple of the alpha where the first strike enters.

    Returns:
        ReductionResult(weights, r2, risk_check)
    """
    floors, bands = _floor_ratios(floors), bands or {}
    risk = pd.DataFrame(index=X.columns) if risk is None else risk
    n, p = X.shape
    R, z, r2 = _scaled_qr(X, y)

    # The penalty vector is a parameter so the problem is compiled once and re-solved for each alpha / strike penalty
    w, penalty = cp.Variable(p, nonneg=True), cp.Parameter(p, nonneg=True)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(R @ w - z) / (2 * n) + penalty @ w),
                         _risk_constraints(w, X.columns, risk, floors, bands))

    def fit(alpha, strike_penalty):
        penalty.value = alpha * strike_penalty
        coef = _solve(problem, w)
        return None if coef is None else {"alpha": alpha, "n": int((coef > 0).sum()), "r2": r2(coef), "coef": coef}

    def best_alpha(strike_penalty):
        top = (R.T @ z / n / strike_penalty).max()                            # alpha where the first strike enters
        grid = [f for a in np.geomspace(top * alpha_max, top * 1e-4, n_grid) if (f := fit(a, strike_penalty))]
        best = max([f for f in grid if f["n"] <= max_positions] or [min(grid, key=lambda f: f["n"])], key=lambda f: f["r2"])
        smaller = [f["alpha"] for f in grid if f["alpha"] < best["alpha"]]
        if smaller:
            refined = [f for a in np.geomspace(best["alpha"], smaller[0], n_refine) if (f := fit(a, strike_penalty))]
            best = max([best, *[f for f in refined if f["n"] <= max_positions]], key=lambda f: f["r2"])
        return best

    strike_penalty, best = np.ones(p), None
    for _ in range(n_iter):
        f = best_alpha(strike_penalty)
        if f["n"] <= max_positions and (best is None or f["r2"] > best["r2"]):
            best = f
        strike_penalty = 1.0 / (f["coef"] + eps_scale * f["coef"][f["coef"] > 0].mean())

    weights = pd.Series(best["coef"], index=X.columns).loc[lambda s: s > 0]
    return ReductionResult(weights, best["r2"], _risk_check(weights, risk, floors, bands))


def omp(
    X: pd.DataFrame, y: pd.Series, risk: Optional[pd.DataFrame], max_positions: int,
    floors: Optional[Floors] = None, bands: Optional[Dict[str, float]] = None, n_candidates: int = 20,
) -> ReductionResult:
    """
    Orthogonal Matching Pursuit (long-only, no penalty) under the same risk constraints as constrained_lasso.

    1. Selection: each step screens the n_candidates positions most correlated with the current residual and adds the
       one giving the best long-only least squares fit (risk constraints are not used here: with few positions they
       are often infeasible and would force poor early choices).
    2. Constrained fit: least squares on the selected positions under the risk constraints.
    3. Swaps: each selected position is replaced by a screened candidate whenever that improves the constrained fit,
       until no swap helps.
    If no feasible selection is found, the long-only fit is returned and risk_check shows the breach.
    Same inputs and output as constrained_lasso.
    """
    floors, bands = _floor_ratios(floors), bands or {}
    risk = pd.DataFrame(index=X.columns) if risk is None else risk
    p = X.shape[1]
    R, z, r2 = _scaled_qr(X, y)

    def expand(selection, coef_selected):
        coef = np.zeros(p)
        coef[selection] = coef_selected
        return coef

    def free_fit(selection):
        return expand(selection, nnls(R[:, selection], z)[0])

    def constrained_fit(selection):
        w = cp.Variable(len(selection), nonneg=True)
        problem = cp.Problem(cp.Minimize(cp.sum_squares(R[:, selection] @ w - z)),
                             _risk_constraints(w, X.columns[selection], risk, floors, bands))
        coef = _solve(problem, w)
        return None if coef is None else expand(selection, coef)

    def screened(coef, selection):
        correlation = R.T @ (z - R @ coef)
        return [j for j in np.argsort(-correlation) if j not in selection][:n_candidates]

    selection, coef = [], np.zeros(p)
    for _ in range(max_positions):
        coef, j = max(((free_fit([*selection, j]), j) for j in screened(coef, selection)), key=lambda t: r2(t[0]))
        selection.append(j)

    best = constrained_fit(selection)
    best_r2 = -np.inf if best is None else r2(best)
    improved = True
    while improved:
        improved = False
        for i in range(len(selection)):
            for j in screened(coef if best is None else best, selection):
                trial = [*selection[:i], j, *selection[i + 1:]]
                candidate = constrained_fit(trial)
                if candidate is not None and r2(candidate) > best_r2 + 1e-9:
                    selection, best, best_r2, improved = trial, candidate, r2(candidate), True

    coef = coef if best is None else best
    weights = pd.Series(coef, index=X.columns).loc[lambda s: s > 0]
    return ReductionResult(weights, r2(coef), _risk_check(weights, risk, floors, bands))


def _scaled_qr(X: pd.DataFrame, y: pd.Series):
    """QR factor of the scaled data, ||y - X w||^2 = ||z - R w||^2 + ||y_perp||^2, and the R2 of weights w."""
    scale = np.sqrt(np.mean(np.square(y)))
    q, R = np.linalg.qr(X.to_numpy() / scale)
    y_scaled = np.asarray(y) / scale
    z = q.T @ y_scaled
    sse_perp = y_scaled @ y_scaled - z @ z
    sst = np.sum((y_scaled - y_scaled.mean()) ** 2)
    return R, z, lambda coef: 1 - (np.sum((z - R @ coef) ** 2) + sse_perp) / sst


def _floor_ratios(floors: Optional[Floors]) -> Dict[str, float]:
    """Floors as {metric: ratio}: a list means ratio 1 (reduced >= full)."""
    return dict(floors) if isinstance(floors, dict) else dict.fromkeys(floors or [], 1.0)


def _risk_constraints(w, columns, risk: pd.DataFrame, floors: Dict[str, float], bands: Dict[str, float]) -> list:
    """Risk constraints on (reduced - full) / |full| for the weights w of the positions `columns`."""
    full = risk.sum()
    rel = {m: risk.loc[columns, m].to_numpy() / abs(full[m]) @ w - np.sign(full[m]) for m in [*floors, *bands]}
    return ([rel[m] + (1 - ratio) * np.sign(full[m]) >= 0 for m, ratio in floors.items()]      # reduced >= ratio * full
            + [cp.abs(rel[m]) <= tol for m, tol in bands.items()])


def _solve(problem: cp.Problem, w: cp.Variable) -> Optional[np.ndarray]:
    """Solves the problem, returns the weights with solver zeros cleaned, or None if not solved."""
    with warnings.catch_warnings():                                           # optimal_inaccurate is accepted below
        warnings.filterwarnings("ignore", message="Solution may be inaccurate")
        try:
            problem.solve(solver=cp.CLARABEL)
        except cp.SolverError:
            return None
    if problem.status not in ("optimal", "optimal_inaccurate"):
        return None
    coef = np.maximum(w.value, 0.0)
    coef[coef < 1e-6 * coef.max()] = 0.0                                      # solver zeros
    return coef


def _risk_check(weights: pd.Series, risk: pd.DataFrame, floors: Dict[str, float], bands: Dict[str, float]) -> pd.DataFrame:
    """full / reduced / diff per risk metric, and ok = whether each constraint holds (NaN if unconstrained)."""
    full = risk.sum()
    reduced = risk.loc[weights.index].mul(weights, axis=0).sum()
    risk_check = pd.DataFrame({"full": full, "reduced": reduced, "diff": reduced - full})
    tol = 1e-5 * full.abs()                                                   # solver precision
    risk_check["ok"] = pd.Series({m: risk_check.loc[m, "reduced"] - ratio * full[m] >= -tol[m] for m, ratio in floors.items()}
                                 | {m: abs(risk_check.loc[m, "diff"]) <= b * abs(full[m]) + tol[m] for m, b in bands.items()},
                                 dtype=object)
    return risk_check


def option_risk_table(
    positions: pd.DataFrame, S0: float, T: float, r: float, sigma: float, slide_shocks: Sequence[float] = (-0.05, -0.10),
) -> pd.DataFrame:
    """
    t0 Black-76 risk of each option position (unit risk x quantity), indexed like positions.

    Columns: delta (spot), gamma_cash (Gamma * S^2 / 100), theta (per day), vega (per vol point) and one delta hedged
    slide per shock x, named slide_{x:+.0%}: PV(S0 * (1 + x)) - PV(S0) - delta * S0 * x.
    """
    F0 = BSMModel.compute_forward(S0, T, r, 0.0, 0.0)
    rows = []
    for strike, option_type in zip(positions["strike"], positions["option_type"]):
        base = BSMModel.compute_option_with_forward(F0, strike, T, r, sigma, option_type, compute_greeks=True)
        row = {
            "delta": base["delta"] * np.exp(r * T),
            "gamma_cash": base["gamma"] * np.exp(2 * r * T) * S0 ** 2 / 100,
            "theta": base["theta"],
            "vega": base["vega"],
        }
        for shock in slide_shocks:
            shocked = BSMModel.compute_option_with_forward(F0 * (1 + shock), strike, T, r, sigma, option_type)
            row[f"slide_{shock:+.0%}"] = shocked - base["price"] - row["delta"] * S0 * shock
        rows.append(row)
    return pd.DataFrame(rows, index=positions.index).mul(positions["quantity"].to_numpy(), axis=0)


def terminal_payoff_lasso(
    portfolio_terminal_payoff: pd.Series,
    option_terminal_payoff: pd.DataFrame,
    max_position: int,
    t0_risk: pd.DataFrame,
    risk_floor: List[str],
    risk_tolerance: Dict[str, float],
    **lasso_kwargs,
) -> ReductionResult:
    """
    Reduces an option book to at most max_position options whose terminal payoff tracks the portfolio terminal
    payoff, under t0 risk constraints vs the full book (weights = 1, risk = t0_risk.sum()), with constrained_lasso.

    Args:
        portfolio_terminal_payoff: payoff to replicate, indexed by terminal spot level (e.g. the full book's payoff).
                                   Sampling the spot levels from Monte Carlo weights each region by its probability.
        option_terminal_payoff: terminal payoff of each option position (payoff x quantity), index = terminal spot
                                level (same as portfolio_terminal_payoff), columns = option strike.
        max_position: maximum number of options kept.
        t0_risk: t0 risk of each option position, indexed by option strike, one column per metric.
        risk_floor: metrics constrained to reduced - full >= 0 (e.g. gamma_cash, theta, slides).
        risk_tolerance: {metric: tol} constraining |reduced - full| <= tol * |full| (e.g. {"vega": 0.05}).
        lasso_kwargs: eps_scale, n_iter, n_grid, n_refine, alpha_max passed to constrained_lasso.

    Returns:
        ReductionResult(weights, r2, risk_check), weights indexed by strike as multipliers of the option positions.
    """
    _check_terminal_inputs(portfolio_terminal_payoff, option_terminal_payoff)
    return constrained_lasso(
        option_terminal_payoff, portfolio_terminal_payoff, t0_risk.loc[option_terminal_payoff.columns],
        max_position, risk_floor, risk_tolerance, **lasso_kwargs,
    )


def quick_terminal_payoff_omp(
    portfolio_weight: pd.Series,
    option_terminal_payoff: pd.DataFrame,
    risk_t0: pd.DataFrame,
    risk_floor: List[str],
    risk_tolerance: Dict[str, float],
    max_position: int = 10,
    option_bid_offer: Optional[pd.Series] = None,
    pnl_haircut: Optional[float] = None,
) -> ReductionResult:
    """
    Reduces a portfolio of options to at most max_position options whose terminal payoff tracks the portfolio's,
    with the constrained OMP (see omp) under t0 risk constraints vs the full portfolio.

    Args:
        portfolio_weight: quantity of each option in the full portfolio, indexed by option name (e.g. "put_95").
        option_terminal_payoff: unit intrinsic value of each option (unweighted), index = terminal spot level,
                                columns = option name. Monte Carlo spot levels weight the fit by probability.
        risk_t0: unit t0 risk of each option (unweighted), index = option name, one column per risk.
        risk_floor: risks constrained to sum(reduced weight x risk) - sum(portfolio weight x risk) >= 0.
        risk_tolerance: {risk: tol} constraining |reduced - portfolio| <= tol * |portfolio| (e.g. {"vega": 0.05}).
        max_position: maximum number of options kept.
        option_bid_offer: optional bid-offer P&L per unit of each option, indexed by option name, negative for a loss.
                          The full portfolio P&L loss is pnl_loss = sum(portfolio_weight * option_bid_offer).
        pnl_haircut: with option_bid_offer, constrains sum(reduced weight * option_bid_offer) >= pnl_haircut * pnl_loss,
                     e.g. 0.8: the reduced portfolio loses at most 80% of the full one. Shown as the "bid_offer" row of
                     risk_check.

    Returns:
        ReductionResult(weights, r2, risk_check): weights = reduced quantity per kept option (same units as
        portfolio_weight), r2 = fit of the reduced to the portfolio terminal payoff, risk_check = portfolio / reduced /
        diff / ok per risk.
    """
    return _quick_terminal_payoff(omp, portfolio_weight, option_terminal_payoff, risk_t0, risk_floor, risk_tolerance,
                                  max_position, option_bid_offer, pnl_haircut)


def quick_terminal_payoff_lasso(
    portfolio_weight: pd.Series,
    option_terminal_payoff: pd.DataFrame,
    risk_t0: pd.DataFrame,
    risk_floor: List[str],
    risk_tolerance: Dict[str, float],
    max_position: int = 10,
    option_bid_offer: Optional[pd.Series] = None,
    pnl_haircut: Optional[float] = None,
) -> ReductionResult:
    """Same as quick_terminal_payoff_omp with the constrained lasso (see constrained_lasso) instead of OMP."""
    return _quick_terminal_payoff(constrained_lasso, portfolio_weight, option_terminal_payoff, risk_t0, risk_floor,
                                  risk_tolerance, max_position, option_bid_offer, pnl_haircut)


def _quick_terminal_payoff(solver, portfolio_weight, option_terminal_payoff, risk_t0, risk_floor, risk_tolerance,
                           max_position, option_bid_offer=None, pnl_haircut=None) -> ReductionResult:
    """Adds the optional bid-offer floor, scales the unit payoffs and risks by the portfolio weights, runs the solver
    against the portfolio's own terminal payoff and converts the multipliers back to quantities."""
    if (option_bid_offer is None) != (pnl_haircut is None):
        raise ValueError("option_bid_offer and pnl_haircut must be given together")
    floors = dict.fromkeys(risk_floor, 1.0)
    if option_bid_offer is not None:
        risk_t0 = risk_t0.assign(bid_offer=option_bid_offer.reindex(risk_t0.index))
        floors["bid_offer"] = pnl_haircut

    names = portfolio_weight.index
    X = option_terminal_payoff[names].mul(portfolio_weight, axis=1)          # payoff of each position
    risk = risk_t0.loc[names].mul(portfolio_weight, axis=0)                   # risk of each position
    result = solver(X, X.sum(axis=1), risk, max_position, floors, risk_tolerance)
    reduced_weight = result.weights * portfolio_weight[result.weights.index]
    risk_check = result.risk_check.rename(columns={"full": "portfolio"})
    return ReductionResult(reduced_weight, result.r2, risk_check)


def _check_terminal_inputs(portfolio_terminal_payoff: pd.Series, option_terminal_payoff: pd.DataFrame) -> None:
    if not option_terminal_payoff.index.equals(portfolio_terminal_payoff.index):
        raise ValueError("option_terminal_payoff and portfolio_terminal_payoff must share the same terminal spot index")
