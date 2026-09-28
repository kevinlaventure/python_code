# KNS skew swap replication under SABR

This project builds the Kozhan–Neuberger–Schneider (RFS 2013) skew swap strip from European options and prices it with the Hagan (2002) SABR vol plus Black‑76. It then hedges the strip along SABR Monte Carlo paths, where the real‑world ρ differs from the pricing ρ, and plots every rebalance along a chosen path.

```
python main.py                                   # defaults, all checks (~3 min)
python main.py --path-id 7 --rebalance threshold --threshold 0.03 --rho-sim -0.5 --n-paths 2000
python main.py --quick --delta-mode bartlett --bid-ask-vol 0.005   # skip sanity/convergence re-runs
python -m pytest -q tests
```

Or open `kns_skew_swap.ipynb` (in this folder) and run all cells. It only imports the modules, sets parameters in one `Config` cell, and shows everything inline: the initial strip, the MC run with a progress bar, an ipywidgets single‑path explorer (plotly), cross‑path results and check tables. Monte Carlo results are pickled to `cache/`, keyed on the full `Config`, so re‑running a plotting cell does not re‑simulate. Set `REFRESH = True` to force a new run. On a cold cache the main run and the ρ_sim = ρ_imp sanity run take about 20 s each; the convergence and strike‑grid tables take about 2 min more.

| file | content |
|---|---|
| `config.py` | `Config` dataclass: every parameter (F0, T, n_steps, β, α0, ν, ρ_imp, ρ_sim, strike grid, delta convention, threshold, bid‑ask, FD bumps) |
| `sabr.py` | Hagan lognormal vol (stable x(z)), Black‑76, Black vega; puts via parity |
| `strip.py` | trapezoid weights, OTM split, v^L, v^E, KNS weights/quantities, recentre trade, parity relabel, G(x) |
| `simulate.py` | SABR paths (log‑Euler on σ and F), realised spot–vol correlation |
| `hedge.py` | `Market` (price and greek grids per step, shared by all variants) and `Book` (holdings, relabel, strip rebalance, delta hedge, logs); `run()` also computes RS |
| `plots.py` | figure factories: Step 0 smile and weights, 11‑panel path figure (PNG / plotly), plotly HTML with a path dropdown, P&L distribution |
| `analysis.py` | pandas tables for results and checks (initial strip, variants, vega drift, convergence, strike grid) and the pickle cache |
| `explorer.py` | ipywidgets path explorer (path box/slider, variant dropdown, random path) used by the notebook |
| `main.py` | CLI and printed report (saved to `output/report.txt`) |
| `kns_skew_swap.ipynb` | notebook front end over the modules above |

Outputs go to `output/`: `step0_weights.png`, `path_<id>_<mode>.png`, `paths_<mode>.html` (dropdown over 10 logged paths), `distribution.png` and `report.txt`. The HTML loads plotly.js from the CDN.

## Mechanics

* **Strip.** Hold q(K) = 6(K−F)/(K²F)·dK OTM options (OTM means a put if K < F, else a call). Then Σ q·OTM = 3(v^E − v^L) exactly on the grid.
* **Rebalance.** Move to the strip centred on F_t. The log leg is static. The trade is 6(1/F_t − 1/F_last)/K·dK, which has the same sign at every strike (see `recenter_trade` and its test). Three modes:
  * `step`: rebalance every step.
  * `threshold`: rebalance when |F_t/F_last − 1| > x.
  * `none`: never rebalance, which gives the static 6G(x_T) payoff, roughly a gamma swap minus a var swap.
* **Relabel.** When F crosses a held strike, the ITM option becomes the same‑strike OTM option via parity. The ±1 forward per option is logged as a *parity* trade, separate from the delta‑hedge trade. The relabel cash keeps it value‑neutral (tested).
* **Delta.**
  * `sticky_alpha`: central finite difference in F with α fixed.
  * `bartlett`: adds (dV/dα)·ρ_imp·ν/F^β.
  * Forwards are marked to market daily through cash.
* **Accounting.** book value = Σ q·price + cash, and it equals cumulative (option MTM + forward P&L − costs) to 1e‑16. Costs use half the bid‑ask (in vol points) times the Black vega on each option trade; the default spread is 0.

## Checking the realised leg

The realised leg is

RS = Σ [3·δv^E_i·(e^{δf_i} − 1) + 6·G(δf_i)],  G(x) = x·e^x − 2e^x + x + 2 ≈ x³/6 + x⁴/12.

It was checked in two ways:

* **Derivation (Neuberger 2012 aggregation property).** v^E_t and v^L_t are re‑referenced to F_t. Take one step of the strip centred at F_i and replicate its payoff model‑free around F_{i+1}. Its option P&L is then exactly 3(e^δ v^E_{i+1} − v^E_i) − 3δv^L + 6G(δ) = [RS increment] + 3(δv^E − δv^L) + 3v^E_i(e^δ − 1).
  * The last term is cancelled by a short of 3v^E_i/F_i forwards. With β = 1, SABR is scale invariant, so this is exactly the sticky‑α delta.
  * The middle term telescopes to −3(v^E − v^L)_0.
  * Therefore the hedged P&L equals RS − 3(v^E − v^L)_0 at every step, whatever the model, up to strike discretisation.
* **Numerics.** With step rebalancing and sticky‑α hedging, the tracking error std is:

  | strike grid | TE std |
  |---|---|
  | 30–300, step 1 | 2.5e‑4 |
  | 5–1000, step 1 | 6.3e‑6 |
  | 5–1000, step 0.25 | 1.5e‑6 |

  For comparison, the P&L std is about 3.8e‑3. The residual therefore comes from the strike grid alone, and the formula is confirmed.

The running benchmark in the path plots is RS_t + 3(v^E − v^L)_t − 3(v^E − v^L)_0: the realised leg so far plus the implied value of what remains. It equals RS − S0 at T.

## Results (defaults: 2000 paths, 126 steps, ρ_imp = −0.70, ρ_sim = −0.50, seed 42)

**Step 0.** v^L = 0.023893, v^E = 0.022177, implied skew 3(v^E − v^L) = **−5.148e‑3** (ATM vol 19.86%). A wide, fine grid (1–3000, step 0.25) gives −5.242e‑3, so truncating at 30–300 loses **1.8%** of the skew, almost all of it in the low‑strike put wing, where weights grow like 1/K².

**Monte Carlo.** P&L is for the long strip, receiving the realised leg. TE = P&L − (RS − S0). "s.e." is the standard error of the mean.

| variant | mean P&L | s.e. | std P&L | TE mean | TE std | rebalances per path |
|---|---|---|---|---|---|---|
| step / sticky‑α | +1.757e‑3 | 8.4e‑5 | 3.75e‑3 | −6e‑6 | 8.0e‑4 | 125 |
| threshold 3% / sticky‑α | +1.762e‑3 | 8.6e‑5 | 3.83e‑3 | −4e‑7 | 1.05e‑3 | 14 |
| none / sticky‑α | +1.600e‑3 | 2.2e‑4 | 1.01e‑2 | −1.6e‑4 | 8.8e‑3 | 0 |
| step / Bartlett | +1.745e‑3 | 8.9e‑5 | 3.97e‑3 | −2e‑5 | 3.5e‑3 | 125 |
| threshold 3% / Bartlett | +1.745e‑3 | 9.1e‑5 | 4.06e‑3 | −2e‑5 | 3.5e‑3 | 14 |
| none / Bartlett | +1.630e‑3 | 2.0e‑4 | 8.8e‑3 | −1.3e‑4 | 7.6e‑3 | 0 |

* **Sign.** With ρ_sim = −0.5 > ρ_imp = −0.7, realised skew is less negative than implied. E[RS] = −3.38e‑3 against S0 = −5.15e‑3, so the long position earns **+1.76e‑3 ± 0.08e‑3**, positive as expected (about 34% of |S0|). Final P&L correlates +0.16 with the path's realised spot–vol correlation for the rebalanced books and about 0 for the static book, whose P&L is dominated by its entropy exposure.
* **Sanity with ρ_sim = ρ_imp: mean P&L is *not* exactly 0.** It is +4.1e‑4 ± 1.1e‑4 (3.7 s.e.). E[RS] = −4.72e‑3 against S0 = −5.15e‑3. The cause is that Hagan's expansion is not a martingale price for the SABR diffusion at ν = 0.8, T = 0.5: a direct Monte Carlo of E[6G(x_T)] (200k paths) gives −4.98e‑3 ± 0.11e‑3, against the Hagan strip value of −5.24e‑3. The Hagan surface overstates the skew. Of the +1.76e‑3 at ρ_sim = −0.5, roughly **+1.35e‑3 is the correlation premium**; the rest is Hagan bias.
* **Vega** (book vega in units of dv^E/dα, after subtracting the fresh strip's own skew vega 3(dv^E − dv^L)/dv^E, which goes from −0.345 at t = 0 to ≈ 0 at T):

  | variant | excess vega regressed on 3(F_t/F0 − 1) |
  |---|---|
  | step | 0.000 (std 0.000) |
  | threshold 3% | slope 0.018, R² 0.02, std 0.041 |
  | none | slope **1.000**, R² **1.000** |

  Without rebalancing the book carries 3(F_t/F0 − 1) entropy contracts, as expected. The recentred strip also has near‑zero cash gamma, while the static book's cash gamma wanders (path plot, panel "cash gamma").
* **Delta convention.** Bartlett delta hedges part of the spot–vol comovement that the swap is meant to capture: the recentred strip still has skew vega, about −0.35 in v^E units at t = 0. It therefore does not track RS (TE std 3.5e‑3) and adds variance. Sticky‑α is the delta the identity requires when β = 1.
* **Convergence** (500 paths, 30–300 grid), TE std by variant:

  | n_steps | step | threshold | none |
  |---|---|---|---|
  | 63 | 3.1e‑4 | 8.6e‑4 | 6.2e‑3 |
  | 126 | 2.5e‑4 | 7.6e‑4 | 5.8e‑3 |
  | 252 | 1.5e‑4 | 6.7e‑4 | 4.3e‑3 |

  For `step`, the identity holds at each step, so what remains is the grid error, which varies with how far F wanders from the grid centre. The `threshold` error is set by the 3% lag, and the `none` error by the unhedged entropy exposure. Neither goes to 0 as dt → 0.

## Caveats

* Hagan's lognormal vol is an asymptotic expansion. At ν = 0.8 its wings are not model‑consistent (see the sanity bias above). The 30–300 grid shows no butterfly arbitrage at t = 0, but this is not checked along paths.
* The finite grid means relabelled strikes and wing truncation leave a few 1e‑4 of TE per path. Use `K_min=5, K_max=1000` for near‑exact replication at about 4× the runtime.
* Relabels are free and costs apply only to strip rebalances. Forwards cost nothing.
* Units are per unit strip notional (third‑moment units). Multiply by your notional.
