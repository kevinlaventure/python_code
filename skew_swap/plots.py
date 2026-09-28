"""Matplotlib (PNG) and plotly (HTML) views of the KNS replication."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from simulate import realized_spot_vol_corr
from strip import kns_weights, otm_from_call, trap_dk
from sabr import hagan_vol, sabr_call

# reference palette (validated categorical order), ink and diverging pair
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
DIV = ["#104281", "#2a78d6", "#f0efec", "#e34948", "#9c2a29"]
DIV_CMAP = LinearSegmentedColormap.from_list("kns_div", DIV[::-1])   # red = sold, blue = bought
OUT = dict(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
PLOTLY_DIV = [[0.0, DIV[4]], [0.25, DIV[3]], [0.5, DIV[2]], [0.75, DIV[1]], [1.0, DIV[0]]]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": INK2,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 1.6, "font.size": 9, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
    "axes.formatter.limits": (-3, 4),
})


# ----------------------------------------------------------------------------- step 0
def fig_initial(cfg):
    """SABR smile at t=0, KNS weights w(K) and value density w(K)·OTM(K)."""
    K = cfg.strikes
    vol = hagan_vol(K, cfg.F0, cfg.T, cfg.alpha0, cfg.beta, cfg.rho_imp, cfg.nu)
    otm = otm_from_call(sabr_call(cfg.F0, K, cfg.T, cfg.alpha0, cfg.beta, cfg.rho_imp, cfg.nu), K, cfg.F0)
    w = kns_weights(K, cfg.F0)
    fig, ax = plt.subplots(3, 1, figsize=(8, 8.5), sharex=True)
    ax[0].plot(K, vol, color=BLUE)
    ax[0].set_title(f"SABR implied vol at t=0  (ρ_imp = {cfg.rho_imp}, ν = {cfg.nu}, β = {cfg.beta})")
    ax[1].plot(K, w, color=BLUE)
    ax[1].set_title("KNS strip weight  w(K) = 6(K − F₀)/(K²F₀)")
    ax[1].axhline(0, color=INK2, lw=0.8)
    ax[2].plot(K, w * otm, color=BLUE)
    ax[2].set_title(f"Value density  w(K)·OTM(K)   (integral = implied skew {np.sum(w * otm * trap_dk(K)):.3e})")
    ax[2].axhline(0, color=INK2, lw=0.8)
    for a in ax:
        a.axvline(cfg.F0, color=INK2, lw=0.8, ls=":")
    ax[2].set_xlabel("strike K")
    fig.tight_layout()
    return fig


def plot_weights(cfg, path):
    fig = fig_initial(cfg)
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------- per-path data
def path_series(res, primary, j):
    """Everything the path panels need for logged path j (index into res['log_idx']).

    `p` indexes rows of res; `label` is the path id shown (differs for single-path reruns, see hedge.run_path).
    """
    p = int(res["log_idx"][j])
    label = int(res.get("path_ids", res["log_idx"])[j])
    cfg, t = res["cfg"], res["cfg"].times
    bp = res["books"][primary]
    bn = res["books"]["none/" + bp.var.delta_mode]
    trades = bp.opt_trades[j].copy()
    trades[0] = np.nan                                   # initial strip purchase dwarfs rebalances
    cum = bp.cum_pnl()
    return dict(
        p=label, t=t, F=res["F"][p], sig=res["sig"][p], K=res["K"],
        rebal_t=t[1:-1][bp.rebalanced[p, 1:-1]],
        trades=trades, is_call=bp.opt_is_call[j],
        hedge=bp.hedge_trade[p], parity=bp.parity_trade[p], pos=bp.n_fwd[p],
        greeks={name: (getattr(bp, a)[p if a in ("vega", "cash_gamma") else j],
                       getattr(bn, a)[p if a in ("vega", "cash_gamma") else j])
                for name, a in [("cash gamma  Γ·F²", "cash_gamma"), ("vega  dV/dα", "vega"),
                                ("vanna  d²V/dF dα", "vanna"), (f"skew exposure  ΔV per +{cfg.fd_rho} ρ_imp", "dvdrho")]},
        total=bp.value[p], comps={k: v[p] for k, v in cum.items()},
        bench=res["bench"][p], te=bp.value[p] - res["bench"][p],
        variant=primary, other=bn.var.name,
    )


def plot_path(res, primary, j, path):
    d = path_series(res, primary, j)
    t = d["t"]
    fig = plt.figure(figsize=(11, 28))
    gs = fig.add_gridspec(11, 1, height_ratios=[1, 1, 2.2, 1, 1, 1, 1, 1, 1, 1.4, 0.9], hspace=0.45)
    axs = [fig.add_subplot(gs[0])]
    axs += [fig.add_subplot(gs[k], sharex=axs[0]) for k in range(1, 11)]
    few = len(d["rebal_t"]) <= 40

    def mark(ax, y):
        if few:
            for x in d["rebal_t"]:
                ax.axvline(x, color=INK2, lw=0.6, ls=":", alpha=0.7)
        ax.plot(d["rebal_t"], np.interp(d["rebal_t"], t, y), "o", ms=3, color=ORANGE, label="option rebalance")

    ax = axs[0]; ax.plot(t, d["F"], color=BLUE, label="F_t"); mark(ax, d["F"])
    ax.set_title(f"Path {d['p']} — forward F_t   [{d['variant']}]"); ax.legend(**OUT)
    ax = axs[1]; ax.plot(t, d["sig"], color=BLUE); mark(ax, d["sig"]); ax.set_title("SABR vol σ_t")

    ax = axs[2]
    z = d["trades"].T
    lim = np.nanpercentile(np.abs(z[z != 0]), 99) if np.any(np.nan_to_num(z) != 0) else 1.0
    im = ax.pcolormesh(t, d["K"], z, cmap=DIV_CMAP, norm=TwoSlopeNorm(0, -lim, lim), shading="nearest")
    ax.plot(t, d["F"], color=INK, lw=1.2, label="F_t")
    ax.set_ylim(d["K"][0], d["K"][-1]); ax.set_ylabel("strike"); ax.grid(False)
    ax.set_title("Option strip trades (OTM units; blue = bought, red = sold; t=0 purchase omitted)")
    fig.colorbar(im, ax=ax, pad=0.01, fraction=0.03)

    ax = axs[3]
    w = (t[1] - t[0]) * 0.8
    ax.bar(t, d["hedge"], width=w, color=BLUE, label="delta-hedge trade")
    ax.bar(t, d["parity"], width=w, color=ORANGE, label="parity relabel")
    ax.set_title("Forward trades"); ax.legend(**OUT)
    ax = axs[4]; ax.plot(t, d["pos"], color=BLUE); ax.set_title("Forward position (after trades)")

    for ax, (name, (a, b)) in zip(axs[5:9], d["greeks"].items()):
        ax.plot(t, a, color=BLUE, label=d["variant"])
        ax.plot(t, b, color=ORANGE, ls="--", label=d["other"])
        ax.set_title(f"Book {name}")
    axs[5].legend(**OUT)

    ax = axs[9]
    ax.plot(t, d["total"], color=BLUE, lw=2, label="book P&L")
    ax.plot(t, d["comps"]["option MTM"], color=ORANGE, label="option MTM")
    ax.plot(t, d["comps"]["forward hedge"], color=AQUA, label="forward hedge")
    ax.plot(t, d["comps"]["costs"], color=YELLOW, label="costs")
    ax.plot(t, d["bench"], color=VIOLET, ls="--", label="RS_t + 3(vᴱ−vᴸ)_t − 3(vᴱ−vᴸ)₀")
    ax.axhline(0, color=INK2, lw=0.8); ax.set_title("Cumulative P&L"); ax.legend(**OUT)
    ax = axs[10]; ax.plot(t, d["te"], color=BLUE); ax.axhline(0, color=INK2, lw=0.8)
    ax.set_title("Tracking error = book P&L − benchmark"); ax.set_xlabel("time (years)")
    for ax in axs[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


PLOTLY_ROWS = ["Forward F_t", "SABR vol σ_t", "Option strip trades (hover: strike, qty, type)",
               "Forward trades", "Forward position", "Cash gamma Γ·F²", "Vega dV/dα", "Vanna d²V/dF dα",
               "Skew exposure ΔV per +0.01 ρ_imp", "Cumulative P&L", "Tracking error"]
PLOTLY_HEIGHTS = [1, 1, 2.2, 1, 1, 1, 1, 1, 1, 1.4, 0.9]


def _plotly_canvas():
    return make_subplots(rows=len(PLOTLY_ROWS), cols=1, shared_xaxes=True, subplot_titles=PLOTLY_ROWS,
                         row_heights=[h / sum(PLOTLY_HEIGHTS) for h in PLOTLY_HEIGHTS], vertical_spacing=0.018)


def _add_path_traces(fig, d, vis=True):
    """Add the 11-panel traces for one path (dict from path_series)."""
    t, rt = d["t"], d["rebal_t"]
    line = lambda y, name, color, row, dash=None, legend=False: fig.add_trace(go.Scatter(
        x=t, y=y, name=name, line=dict(color=color, width=2, dash=dash), visible=vis,
        legendgroup=name, showlegend=legend and vis, hovertemplate="%{y:.4g}"), row=row, col=1)
    line(d["F"], "F_t", BLUE, 1)
    fig.add_trace(go.Scatter(x=rt, y=np.interp(rt, t, d["F"]), mode="markers", name="option rebalance",
                             marker=dict(color=ORANGE, size=6), visible=vis, showlegend=vis), row=1, col=1)
    line(d["sig"], "σ_t", BLUE, 2)
    fig.add_trace(go.Scatter(x=rt, y=np.interp(rt, t, d["sig"]), mode="markers", showlegend=False,
                             marker=dict(color=ORANGE, size=6), visible=vis, hoverinfo="skip"), row=2, col=1)
    z = d["trades"].T
    lim = np.nanpercentile(np.abs(z[z != 0]), 99) if np.any(np.nan_to_num(z) != 0) else 1.0
    cd = np.where(d["is_call"].T, "call", "put")
    zr = [[None if np.isnan(v) else float(f"{v:.4g}") for v in row] for row in z]   # 4 s.f. keeps HTML small
    fig.add_trace(go.Heatmap(x=t, y=d["K"], z=zr, customdata=cd, colorscale=PLOTLY_DIV, zmid=0,
                             zmin=-lim, zmax=lim, visible=vis, showscale=False,
                             hovertemplate="t=%{x:.3f}<br>K=%{y}<br>qty=%{z:.3e}<br>%{customdata}<extra></extra>"),
                  row=3, col=1)
    line(d["F"], "F_t", INK, 3)
    fig.add_trace(go.Bar(x=t, y=d["hedge"], name="delta-hedge trade", marker_color=BLUE, visible=vis,
                         showlegend=vis), row=4, col=1)
    fig.add_trace(go.Bar(x=t, y=d["parity"], name="parity relabel", marker_color=ORANGE, visible=vis,
                         showlegend=vis), row=4, col=1)
    line(d["pos"], "forward position", BLUE, 5)
    for r, (a, b) in zip(range(6, 10), d["greeks"].values()):
        line(a, d["variant"], BLUE, r, legend=(r == 6))
        line(b, d["other"], ORANGE, r, dash="dash", legend=(r == 6))
    line(d["total"], "book P&L", BLUE, 10, legend=True)
    line(d["comps"]["option MTM"], "option MTM", ORANGE, 10, legend=True)
    line(d["comps"]["forward hedge"], "forward hedge", AQUA, 10, legend=True)
    line(d["comps"]["costs"], "costs", YELLOW, 10, legend=True)
    line(d["bench"], "RS_t + 3(vE−vL)_t − 3(vE−vL)_0", VIOLET, 10, dash="dash", legend=True)
    line(d["te"], "tracking error", BLUE, 11)


def _plotly_layout(fig, title, height):
    fig.update_layout(
        height=height, template="plotly_white", barmode="relative", paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE, font=dict(color=INK, size=11), hovermode="x unified", title=title,
        legend=dict(orientation="h", y=1.005, yanchor="bottom", x=0))
    fig.update_xaxes(title_text="time (years)", row=len(PLOTLY_ROWS), col=1)
    return fig


def plotly_path_figure(res, primary, j=0, height=2400):
    """Interactive 11-panel figure for one logged path."""
    d = path_series(res, primary, j)
    fig = _plotly_canvas()
    _add_path_traces(fig, d)
    return _plotly_layout(fig, f"Path {d['p']} — {primary}", height)


def plotly_paths(res, primary, j0, path):
    """Standalone HTML with a dropdown over all logged paths."""
    fig = _plotly_canvas()
    groups = []
    for j in range(len(res["log_idx"])):
        n0 = len(fig.data)
        d = path_series(res, primary, j)
        _add_path_traces(fig, d, vis=(j == j0))
        groups.append((d["p"], range(n0, len(fig.data))))

    n = len(fig.data)
    buttons = []
    for p, idx in groups:
        v = [False] * n
        for k in idx:
            v[k] = True
        buttons.append(dict(label=f"path {p}", method="update",
                            args=[{"visible": v, "showlegend": [v[k] and fig.data[k].showlegend is not False
                                                                for k in range(n)]}]))
    _plotly_layout(fig, f"KNS skew swap replication — {primary}", 2600)
    fig.update_layout(updatemenus=[dict(buttons=buttons, active=j0, x=1.0, xanchor="right", y=1.02,
                                        yanchor="bottom")])
    fig.write_html(path, include_plotlyjs="cdn")


# ----------------------------------------------------------------------------- distribution
def fig_distribution(res, variants):
    """Final P&L histograms per variant and P&L vs realised spot-vol correlation."""
    corr = realized_spot_vol_corr(res["F"], res["sig"])
    target = res["RS"][:, -1] - res["S0"]
    colors = [BLUE, ORANGE, AQUA]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    pnls = [res["books"][v].value[:, -1] for v in variants]
    bins = np.linspace(np.percentile(np.concatenate(pnls), 0.5), np.percentile(np.concatenate(pnls), 99.5), 60)
    for v, p, c in zip(variants, pnls, colors):
        ax[0].hist(p, bins=bins, histtype="step", lw=1.8, color=c, label=f"{v}  mean {p.mean():+.2e}")
        ax[0].axvline(p.mean(), color=c, lw=1, ls="--")
    ax[0].hist(target, bins=bins, histtype="stepfilled", color=GRID, alpha=0.7, label="RS − 3(vᴱ−vᴸ)₀", zorder=0)
    ax[0].axvline(0, color=INK2, lw=0.8)
    ax[0].set_title("Final P&L per variant"); ax[0].set_xlabel("P&L (per unit strip notional)"); ax[0].legend()
    for v, p, c in zip(variants, pnls, colors):
        ax[1].scatter(corr, p, s=9, color=c, alpha=0.45, edgecolors="none", label=v)
    cfg = res["cfg"]
    for r, lab in [(cfg.rho_imp, "ρ_imp"), (cfg.rho_sim, "ρ_sim")]:
        ax[1].axvline(r, color=INK2, lw=0.8, ls=":")
        ax[1].text(r, ax[1].get_ylim()[1], f" {lab}", va="top", color=INK2)
    ax[1].axhline(0, color=INK2, lw=0.8)
    ax[1].set_title("Final P&L vs realised spot–vol correlation"); ax[1].set_xlabel("corr(d log F, d log σ) on path")
    ax[1].legend(markerscale=2)
    fig.tight_layout()
    return fig


def plot_distribution(res, variants, path):
    fig = fig_distribution(res, variants)
    fig.savefig(path, dpi=130)
    plt.close(fig)
