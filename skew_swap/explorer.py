"""ipywidgets single-path explorer for the notebook.

Detailed per-strike logs are not kept for every MC path (≈0.5 GB per variant); the selected path is
re-run through the engine on demand (hedge.run_path, identical to its row in the full run).
"""
import ipywidgets as W
import numpy as np
from IPython.display import clear_output

from hedge import run_path
from plots import plotly_path_figure
from simulate import realized_spot_vol_corr


def path_explorer(res, variant=None, height=2400, seed=None):
    n = res["F"].shape[0]
    names = list(res["books"])
    corr = realized_spot_vol_corr(res["F"], res["sig"])
    target = res["RS"][:, -1] - res["S0"]

    idx = W.BoundedIntText(value=0, min=0, max=n - 1, description="path", layout=W.Layout(width="160px"))
    slider = W.IntSlider(value=0, min=0, max=n - 1, continuous_update=False, readout=False,
                         layout=W.Layout(width="320px"))
    W.link((idx, "value"), (slider, "value"))
    var = W.Dropdown(options=names, value=variant if variant in names else names[0], description="variant")
    btn = W.Button(description="random path", icon="random")
    info = W.HTML()
    out = W.Output()
    rng = np.random.default_rng(seed)
    runs = {}

    def draw(*_):
        p, v = idx.value, var.value
        if p not in runs:
            runs[p] = run_path(res, p)
        pnl = res["books"][v].value[p, -1]
        info.value = (f"<b>path {p}</b> · final P&L {pnl:+.3e} · RS − S0 {target[p]:+.3e} · "
                      f"tracking error {pnl - target[p]:+.3e} · realised spot-vol corr {corr[p]:+.3f} · "
                      f"option rebalances {res['books'][v].rebalanced[p, 1:].sum()}")
        fig = plotly_path_figure(runs[p], v, 0, height=height)
        with out:
            clear_output(wait=True)
            fig.show()

    idx.observe(draw, "value")
    var.observe(draw, "value")
    btn.on_click(lambda _: setattr(idx, "value", int(rng.integers(n))))
    draw()
    return W.VBox([W.HBox([idx, slider, var, btn]), info, out])
