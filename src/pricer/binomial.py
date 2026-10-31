"""CRR binomial tree for the same convertible (cross-check for LSM).

u = exp(sigma sqrt(dt)), d = 1/u, p = (e^{(r-q)dt} - d)/(u - d).
Same Tsiveriotis-Fernandes cash/equity split and same contract logic as lsm.py.
Soft call supported only with n_required = window = 1 (single-step trigger).
"""
from __future__ import annotations

import numpy as np

from src.contracts import Convertible


def price_binomial(S0: float, bond: Convertible, r: float, q: float, sigma: float,
                   spread: float, n_steps: int) -> float:
    sc = bond.soft_call
    if sc is not None and not (sc.n_required == 1 and sc.window == 1):
        raise ValueError("binomial supports only 1-of-1 soft call triggers")
    T = bond.maturity
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    dfr, dfc = np.exp(-r * dt), np.exp(-(r + spread) * dt)
    CR = bond.conversion_ratio
    cpn = bond.coupon_vector(n_steps)
    put = bond.put_vector(n_steps)
    prot = bond.protection_step(n_steps)

    j = np.arange(n_steps + 1)
    S = S0 * u ** (2 * j - n_steps)          # node j at step n: u^j d^(n-j)
    parity = CR * S
    conv = parity > bond.redemption_value
    eq = np.where(conv, parity, 0.0)
    cash = np.where(conv, 0.0, bond.redemption_value)
    cash = cash + cpn[n_steps]

    for k in range(n_steps - 1, -1, -1):
        cash = dfc * (p * cash[1:k + 2] + (1 - p) * cash[0:k + 1])
        eq = dfr * (p * eq[1:k + 2] + (1 - p) * eq[0:k + 1])
        S = S0 * u ** (2 * np.arange(k + 1) - k)
        parity = CR * S
        if k > 0:
            V = cash + eq
            X_put = put[k]
            use_put = X_put > parity
            X = np.where(use_put, X_put, parity)
            ex = X > V
            eq = np.where(ex, np.where(use_put, 0.0, parity), eq)
            cash = np.where(ex, np.where(use_put, X_put, 0.0), cash)
            if sc is not None and k >= max(prot, 1):
                called = S >= sc.trigger
                c_conv = parity > sc.price
                eq = np.where(called, np.where(c_conv, parity, 0.0), eq)
                cash = np.where(called, np.where(c_conv, 0.0, sc.price), cash)
            cash = cash + cpn[k]
    return float((cash + eq)[0])
