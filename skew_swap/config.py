from dataclasses import dataclass, field, replace

import numpy as np


@dataclass(frozen=True)
class Config:
    # market / diffusion
    F0: float = 100.0
    T: float = 0.5
    n_steps: int = 126
    beta: float = 1.0
    alpha0: float = 0.20
    nu: float = 0.8
    rho_imp: float = -0.70          # pricing model (market) correlation
    rho_sim: float = -0.50          # real-world diffusion correlation
    # strike grid
    K_min: float = 30.0
    K_max: float = 300.0
    K_step: float = 1.0
    # Monte Carlo
    n_paths: int = 2000
    seed: int = 42
    # hedging
    delta_mode: str = "sticky_alpha"  # "sticky_alpha" | "bartlett"
    rebalance: str = "step"           # primary variant for path plots: "step" | "threshold" | "none"
    threshold: float = 0.03           # re-centre when |F_t / F_last - 1| > threshold
    bid_ask_vol: float = 0.0          # full bid-ask in vol points (0.01 = 1 vol pt); half paid per trade
    # finite-difference bumps (relative)
    fd_F: float = 1e-3
    fd_alpha: float = 1e-3
    fd_rho: float = 0.01              # absolute bump of rho_imp for skew exposure
    # outputs
    path_id: int = 0
    n_log_paths: int = 10             # paths with full per-strike logs (plotly dropdown)
    out_dir: str = "output"

    @property
    def strikes(self) -> np.ndarray:
        n = int(round((self.K_max - self.K_min) / self.K_step)) + 1
        return self.K_min + self.K_step * np.arange(n)

    @property
    def dt(self) -> float:
        return self.T / self.n_steps

    @property
    def times(self) -> np.ndarray:
        return np.linspace(0.0, self.T, self.n_steps + 1)

    def with_(self, **kw) -> "Config":
        return replace(self, **kw)
