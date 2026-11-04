"""Credit as a jump to default (Phase 5).

Model.  Pre-default stock S_t follows the path generator; default arrives at the first time the cumulative hazard
exceeds an independent Exp(1) variable, with stock-dependent intensity (Carr-Linetsky jump-to-default CEV)

    lambda_t = min(lam_max, lam0 * (S_t / S0)^(-beta)),     beta >= 0   (beta = 0: constant hazard).

At default the stock is worthless and the bond pays recovery * face. Given the pre-default path, survival to step k has
probability w_k = exp(-dt * sum_{j<k} lambda_j), so instead of simulating default times we use survival weights
(Cox process; no extra Monte Carlo variance):

    cash_k = dfr * [ q_k cash_{k+1} + (1 - q_k) * recovery * face ],   eq_k = dfr * q_k * eq_{k+1},   q_k = exp(-lambda_k dt).

Risk-neutral drift.  E[S_k 1{tau > k}] = E[S_k w_k] must equal S0 e^{(r-q)t_k}. We rescale pre-default paths per time step,
S_k <- c_k S_k, solving the fixed point (c_k enters the hazard) by damped iteration. For beta = 0 this is the usual
S e^{lam t} drift uplift.

Calibration.  lam0 is chosen so that the risky zero-coupon bond (same recovery) matches a flat-spread bond
face * exp(-(r+s) T). Any difference to the Phase 4 flat-spread convertible price is then purely credit-model risk.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from src.contracts import Convertible
from src.pricer.lsm import price_lsm

LAM_MAX = 5.0


def hazard(S: np.ndarray, S0: float, lam0: float, beta: float, lam_max: float = LAM_MAX) -> np.ndarray:
    if beta == 0.0:
        return np.full_like(S, min(lam0, lam_max))
    return np.minimum(lam_max, lam0 * (S / S0) ** (-beta))


def survival(lam: np.ndarray, dt: float) -> np.ndarray:
    """w[:, k] = P(tau > t_k | path), shape like lam."""
    w = np.ones_like(lam)
    w[:, 1:] = np.exp(-dt * np.cumsum(lam[:, :-1], axis=1))
    return w


def compensated_paths(Z: np.ndarray, S0: float, r: float, q: float, T: float, weekly_vol: float,
                      lam0: float, beta: float, iters: int = 40, tol: float = 1e-10):
    """Unit-variance weekly returns -> (S, lam, w, max_rel_err) with E[S_k w_k] = S0 e^{(r-q) t_k} for all k."""
    Z = Z - Z.mean()
    X = np.concatenate([np.zeros((Z.shape[0], 1)), np.cumsum(weekly_vol * Z, axis=1)], axis=1)
    S_raw = S0 * np.exp(X)
    n = S_raw.shape[1] - 1
    dt = T / n
    target = S0 * np.exp((r - q) * dt * np.arange(n + 1))
    c = np.ones(n + 1)
    err = np.inf
    for _ in range(iters):
        S = S_raw * c
        lam = hazard(S, S0, lam0, beta)
        w = survival(lam, dt)
        m = (S * w).mean(axis=0)
        ratio = target / m
        err = float(np.max(np.abs(ratio - 1.0)))
        if err < tol:
            break
        c = c * ratio ** (0.7 if beta > 0 else 1.0)
    S = S_raw * c
    lam = hazard(S, S0, lam0, beta)
    return S, lam, survival(lam, dt), err


def zcb_value(lam: np.ndarray, r: float, T: float, recovery: float, face: float = 100.0) -> float:
    """Risky zero-coupon bond value, recovery paid at the end of the default step (mean over paths)."""
    n = lam.shape[1] - 1
    dt = T / n
    dfr = np.exp(-r * dt)
    V = np.full(lam.shape[0], face)
    for k in range(n - 1, -1, -1):
        qk = np.exp(-lam[:, k] * dt)
        V = dfr * (qk * V + (1.0 - qk) * recovery * face)
    return float(V.mean())


def calibrate_lam0(Z: np.ndarray, S0: float, r: float, q: float, T: float, weekly_vol: float,
                   beta: float, recovery: float, spread: float, face: float = 100.0) -> float:
    """lam0 such that the risky ZCB matches face * exp(-(r + spread) T)."""
    target = face * np.exp(-(r + spread) * T)

    def f(l0):
        _, lam, _, _ = compensated_paths(Z, S0, r, q, T, weekly_vol, l0, beta)
        return zcb_value(lam, r, T, recovery, face) - target
    return float(brentq(f, 1e-6, 2.5, xtol=1e-7))


def price_jd(Z: np.ndarray, S0: float, r: float, q: float, T: float, weekly_vol: float, bond: Convertible,
             lam0: float, beta: float, recovery: float, keep_pv: bool = False) -> dict:
    S, lam, _, err = compensated_paths(Z, S0, r, q, T, weekly_vol, lam0, beta)
    res = price_lsm(S, bond, r, 0.0, hazard=lam, recovery=recovery, keep_pv=keep_pv)
    res["comp_err"] = err
    return res
