"""Synthetic 'market' with known stylized facts, used to develop and test the generator offline.

GJR-GARCH(1,1) with standardised Student-t innovations (weekly):
    r_t = sigma_t z_t,   z_t ~ t_nu scaled to unit variance
    sigma_t^2 = omega + (alpha + gamma 1{r_{t-1}<0}) r_{t-1}^2 + beta sigma_{t-1}^2
Unconditional variance  v = omega / (1 - alpha - gamma/2 - beta)  (symmetric z), so omega is set from a target v.
Fat tails (nu), volatility clustering (beta, alpha) and leverage (gamma) are all present by construction,
so a good generator must reproduce them.
"""
from __future__ import annotations

import numpy as np


def simulate_gjr(n: int, target_weekly_vol: float, seed: int, alpha: float = 0.06,
                 gamma: float = 0.06, beta: float = 0.88, nu: float = 5.0, burn: int = 500) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = target_weekly_vol ** 2
    persist = alpha + 0.5 * gamma + beta
    assert persist < 1
    omega = v * (1 - persist)
    z = rng.standard_t(nu, size=n + burn) / np.sqrt(nu / (nu - 2.0))
    r = np.empty(n + burn)
    s2 = v
    for t in range(n + burn):
        r[t] = np.sqrt(s2) * z[t]
        s2 = omega + (alpha + (gamma if r[t] < 0 else 0.0)) * r[t] ** 2 + beta * s2
    return r[burn:]


def synthetic_basket(n_weeks: int = 800, vols=(0.075, 0.06, 0.07, 0.045, 0.04, 0.05, 0.055, 0.065),
                     seed: int = 0) -> dict:
    return {f"SYN{i}": simulate_gjr(n_weeks, v, seed + i) for i, v in enumerate(vols)}
