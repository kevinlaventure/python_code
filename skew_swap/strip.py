"""KNS (Kozhan-Neuberger-Schneider 2013) option strips on a discrete strike grid.

Conventions (zero rates, forward = spot):
    OTM(K, F) = put if K < F else call
    v^L = 2 * int OTM(K) / K^2   dK      = E[-2 ln(F_T/F)]
    v^E = 2 * int OTM(K) / (K F) dK      = E[ 2 (F_T/F) ln(F_T/F)]
    implied skew = 3 (v^E - v^L)         = E[6 G(ln F_T/F)] ~ E[x^3]
All integrals use trapezoidal quadrature on the strike grid (`trap_dk`).
"""
import numpy as np


def trap_dk(K):
    """Trapezoid quadrature weights for a (possibly non-uniform) strike grid."""
    K = np.asarray(K, dtype=float)
    dk = np.empty_like(K)
    dk[1:-1] = 0.5 * (K[2:] - K[:-2])
    dk[0] = 0.5 * (K[1] - K[0])
    dk[-1] = 0.5 * (K[-1] - K[-2])
    return dk


def is_call_otm(K, F):
    """OTM option type at each strike: True = call (K >= F), False = put. F broadcasts (e.g. shape (m, 1))."""
    return np.asarray(K) >= np.asarray(F)


def otm_from_call(call, K, F):
    """OTM prices from call prices via put-call parity P = C - (F - K)."""
    return np.where(is_call_otm(K, F), call, call - (F - K))


def log_strip(otm, K, F=None):
    """v^L = 2 sum OTM / K^2 dK. `otm` has strikes on the last axis."""
    return 2.0 * np.sum(otm * (trap_dk(K) / K**2), axis=-1)


def entropy_strip(otm, K, F):
    """v^E = 2 sum OTM / (K F) dK, with F the reference forward (scalar or shape (m,))."""
    return 2.0 * np.sum(otm * (trap_dk(K) / K), axis=-1) / np.asarray(F, dtype=float)


def kns_weights(K, F):
    """KNS strip density w(K) = 6 (K - F) / (K^2 F) = 6/(K F) - 6/K^2."""
    F = np.asarray(F, dtype=float)
    return 6.0 * (K - F) / (K**2 * F)


def kns_quantities(K, F):
    """Number of OTM options held per grid strike = w(K) dK. Sum(q * OTM) = 3 (v^E - v^L)."""
    return kns_weights(K, F) * trap_dk(K)


def recenter_trade(K, F_new, F_old):
    """Option trade (in OTM units) to move the strip centre from F_old to F_new.

    The log (1/K^2) leg is static; only the entropy leg changes: 6 (1/F_new - 1/F_old) / K * dK,
    which has the same sign at every strike.
    """
    F_new = np.asarray(F_new, dtype=float)
    F_old = np.asarray(F_old, dtype=float)
    return 6.0 * (1.0 / F_new - 1.0 / F_old) / K * trap_dk(K)


def parity_relabel(q, K, was_call, F):
    """Convert held options to the OTM type w.r.t. the current forward F (put-call parity).

    q, was_call: (m, nK); F: (m, 1). Returns (now_call, fwd_trade (m,), cash (m,)).
    call -> put:  C = P + (F - K)  => hold P, buy  1 forward, receive (F - K) cash.
    put  -> call: P = C - (F - K)  => hold C, sell 1 forward, receive (K - F) cash.
    The conversion is value-neutral; the forward leg is logged separately from hedge trades.
    """
    now_call = is_call_otm(K, F)
    changed = now_call != was_call
    sign = np.where(was_call, 1.0, -1.0) * changed          # +1 call->put, -1 put->call, 0 unchanged
    fwd_trade = np.sum(q * sign, axis=-1)
    cash = np.sum(q * sign * (F - K), axis=-1)
    return now_call, fwd_trade, cash


def G(x):
    """G(x) = x e^x - 2 e^x + x + 2  (~ x^3/6 + x^4/12)."""
    x = np.asarray(x, dtype=float)
    ex = np.exp(x)
    return x * ex - 2.0 * ex + x + 2.0
