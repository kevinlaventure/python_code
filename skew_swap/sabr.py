"""SABR (Hagan 2002 lognormal expansion) + Black-76 pricing, vectorised with numpy broadcasting.

Zero rates, no dividends: spot = forward F, undiscounted prices.
"""
import numpy as np
from scipy.special import ndtr


def hagan_vol(K, F, T, alpha, beta, rho, nu):
    """Hagan et al. (2002) lognormal implied vol. All inputs broadcast together."""
    K, F, T, alpha = np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in (K, F, T, alpha)))
    one_b = 1.0 - beta
    log_fk = np.log(F / K)
    fk_pow = (F * K) ** (0.5 * one_b)

    denom = fk_pow * (1.0 + one_b**2 / 24.0 * log_fk**2 + one_b**4 / 1920.0 * log_fk**4)
    z = nu / alpha * fk_pow * log_fk

    # x(z) = log((sqrt(1-2 rho z + z^2) + z - rho) / (1 - rho)); rationalised form when z - rho < 0
    # avoids cancellation for very negative z (high strikes with negative rho).
    sq = np.sqrt(1.0 - 2.0 * rho * z + z * z)
    num = np.where(z - rho >= 0.0, sq + z - rho, (1.0 - rho * rho) / np.maximum(sq - z + rho, 1e-300))
    x = np.log(num / (1.0 - rho))
    small = np.abs(z) < 1e-7
    z_over_x = np.where(small, 1.0 - 0.5 * rho * z, z / np.where(small, 1.0, x))

    corr = 1.0 + (one_b**2 / 24.0 * alpha**2 / fk_pow**2
                  + 0.25 * rho * beta * nu * alpha / fk_pow
                  + (2.0 - 3.0 * rho**2) / 24.0 * nu**2) * T
    return alpha / denom * z_over_x * corr


def black_call(F, K, T, vol):
    """Undiscounted Black-76 call. T <= 0 returns intrinsic."""
    F, K, T, vol = np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in (F, K, T, vol)))
    sd = vol * np.sqrt(np.maximum(T, 0.0))
    live = sd > 1e-14
    sd_safe = np.where(live, sd, 1.0)
    d1 = (np.log(F / K) + 0.5 * sd_safe**2) / sd_safe
    price = F * ndtr(d1) - K * ndtr(d1 - sd_safe)
    return np.where(live, price, np.maximum(F - K, 0.0))


def black_price(F, K, T, vol, is_call):
    call = black_call(F, K, T, vol)
    return np.where(is_call, call, call - (F - K))


def black_vega(F, K, T, vol):
    """dPrice/dvol (Black), used for vol-point bid-ask costs."""
    F, K, T, vol = np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in (F, K, T, vol)))
    sqt = np.sqrt(np.maximum(T, 0.0))
    sd = np.maximum(vol * sqt, 1e-14)
    d1 = (np.log(F / K) + 0.5 * sd**2) / sd
    return np.where(T > 0, F * np.exp(-0.5 * d1**2) / np.sqrt(2 * np.pi) * sqt, 0.0)


def sabr_call(F, K, T, alpha, beta, rho, nu):
    """Undiscounted call priced with Hagan vol + Black. Puts via parity: P = C - (F - K)."""
    T_safe = np.maximum(T, 1e-12)
    vol = hagan_vol(K, F, T_safe, alpha, beta, rho, nu)
    return black_call(F, K, T, vol)


def sabr_price(F, K, T, alpha, beta, rho, nu, is_call):
    call = sabr_call(F, K, T, alpha, beta, rho, nu)
    return np.where(is_call, call, call - (F - K))
