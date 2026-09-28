"""SABR Monte Carlo: dF = sigma F^beta dW1, dsigma = nu sigma dW2, corr(dW1, dW2) = rho_sim.

Log-Euler for sigma (exact given the Brownian increment) and log-Euler for F with local vol
sigma F^(beta-1); for beta = 1 this is exact conditional on sigma_i and keeps F > 0 and E[F_{i+1} | i] = F_i.
"""
import numpy as np

from config import Config


def simulate_paths(cfg: Config, rho: float | None = None):
    """Returns F, sigma of shape (n_paths, n_steps + 1)."""
    rho = cfg.rho_sim if rho is None else rho
    rng = np.random.default_rng(cfg.seed)
    n, m, dt = cfg.n_paths, cfg.n_steps, cfg.dt
    z1 = rng.standard_normal((n, m))
    z2 = rho * z1 + np.sqrt(1.0 - rho**2) * rng.standard_normal((n, m))

    F = np.empty((n, m + 1))
    sig = np.empty((n, m + 1))
    F[:, 0], sig[:, 0] = cfg.F0, cfg.alpha0
    sq = np.sqrt(dt)
    for i in range(m):
        loc = sig[:, i] * F[:, i] ** (cfg.beta - 1.0)
        F[:, i + 1] = F[:, i] * np.exp(-0.5 * loc**2 * dt + loc * sq * z1[:, i])
        sig[:, i + 1] = sig[:, i] * np.exp(-0.5 * cfg.nu**2 * dt + cfg.nu * sq * z2[:, i])
    return F, sig


def realized_spot_vol_corr(F, sig):
    """Per-path sample correlation of d log F and d log sigma."""
    a = np.diff(np.log(F), axis=1)
    b = np.diff(np.log(sig), axis=1)
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))
