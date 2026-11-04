"""Longstaff-Schwartz pricer for a convertible bond (contract-agnostic given a path array).

State carried backward, per path, at time t_k (discounted to t_k):
    cash : bond-like cash flows, discounted at r + s   (Tsiveriotis-Fernandes split)
    eq   : equity-like flows (conversion value), discounted at r
Total value V = cash + eq.

Algorithm
---------
1. Forward: find end[i] = first step where the soft-call N-of-M trigger holds (else n).
2. Backward k = n..0:
     - paths with end == k   : terminal payoff (maturity redemption or call), holder
                               takes max(redemption-or-call price, parity).
     - paths with end  > k   : roll (cash *= e^{-(r+s)dt}, eq *= e^{-r dt}); for 0<k<n
                               regress V on polynomial(S_k/S_0) over alive paths -> C_hat;
                               exercise (convert or put) if max(parity, put) > C_hat.
     - coupon paid at t_k is added to cash for every path that is still alive at t_k.
3. Price = mean over paths of V at k=0.  LSM is low-biased (suboptimal policy), so
   price_lsm is a lower estimate of the true value under the chosen basis.
"""
from __future__ import annotations

import numpy as np

from src.contracts import Convertible


def _call_end_steps(paths: np.ndarray, bond: Convertible):
    """Return (end, called) where end[i] in 1..n and called[i] is True if soft call ended the path."""
    N, n1 = paths.shape
    n = n1 - 1
    sc = bond.soft_call
    end = np.full(N, n, dtype=int)
    called = np.zeros(N, dtype=bool)
    if sc is None:
        return end, called
    M = sc.window
    above = (paths >= sc.trigger).astype(np.int32)
    cs = np.cumsum(above, axis=1)
    cnt = cs.copy()
    cnt[:, M:] = cs[:, M:] - cs[:, :-M]          # number above in the last M steps (inclusive)
    k = np.arange(n1)
    eligible = (k >= max(M - 1, bond.protection_step(n), 1)) & (k < n)
    trig = (cnt >= sc.n_required) & eligible[None, :]
    has = trig.any(axis=1)
    first = trig.argmax(axis=1)
    end = np.where(has, first, n)
    called = has
    return end, called


def _basis(x: np.ndarray, degree: int, hinges: tuple) -> np.ndarray:
    """Polynomial in x = S/S0 plus hinge (call-payoff) features max(x - h, 0).

    The convertible value has a hockey-stick kink near parity = bond floor; hinges let the
    regression represent it, which removes most of LSM's premature-conversion bias.
    """
    cols = [x ** j for j in range(degree, -1, -1)]
    cols += [np.maximum(x - h, 0.0) for h in hinges]
    return np.column_stack(cols)


def price_lsm(paths: np.ndarray, bond: Convertible, r: float, spread: float,
              degree: int = 3, pair_average: bool = False, hinges: str | tuple = "auto",
              hazard: np.ndarray | None = None, recovery: float = 0.0, keep_pv: bool = False) -> dict:
    N, n1 = paths.shape
    n = n1 - 1
    dt = bond.maturity / n
    CR = bond.conversion_ratio
    dfr, dfc = np.exp(-r * dt), np.exp(-(r + spread) * dt)
    cpn = bond.coupon_vector(n)
    put = bond.put_vector(n)
    S0 = paths[0, 0]
    end, called = _call_end_steps(paths, bond)
    if hinges == "auto":
        k0 = bond.redemption_value / CR / S0          # parity-equals-redemption moneyness
        hinges = tuple(k0 * m for m in (0.6, 0.8, 1.0, 1.25, 1.6))
    elif hinges is None:
        hinges = ()

    cash = np.zeros(N)
    eq = np.zeros(N)
    converted_early = np.zeros(N, dtype=bool)
    put_exercised = np.zeros(N, dtype=bool)

    for k in range(n, -1, -1):
        S = paths[:, k]
        parity = CR * S
        # --- terminal paths at step k
        term = end == k
        if term.any():
            if k == n:
                floor = bond.redemption_value
            else:
                floor = bond.soft_call.price
            conv = parity[term] > floor
            tidx = np.where(term)[0]
            eq[tidx] = np.where(conv, parity[term], 0.0)
            cash[tidx] = np.where(conv, 0.0, floor)
        # --- alive paths: roll back and decide
        alive = end > k
        if alive.any():
            if hazard is None:
                cash[alive] *= dfc
                eq[alive] *= dfr
            else:
                # jump to default during step k -> k+1 (hazard[:, k] known at k): survive w.p. q, else recovery*face
                ia0 = np.where(alive)[0]
                qk = np.exp(-hazard[ia0, k] * dt)
                cash[ia0] = dfc * qk * cash[ia0] + dfr * (1.0 - qk) * recovery * bond.face
                eq[ia0] = dfr * qk * eq[ia0]
            if 0 < k:
                a = np.where(alive)[0]
                V = cash[a] + eq[a]
                x = S[a] / S0
                B = _basis(x, degree, hinges)
                coef, *_ = np.linalg.lstsq(B, V, rcond=None)
                Chat = B @ coef
                X_put = put[k]
                use_put = X_put > parity[a]
                X = np.where(use_put, X_put, parity[a])
                ex = X > Chat
                if ex.any():
                    ia = a[ex]
                    up = use_put[ex]
                    eq[ia] = np.where(up, 0.0, parity[ia])
                    cash[ia] = np.where(up, X_put, 0.0)
                    converted_early[ia] = ~up
                    put_exercised[ia] = up
        # --- coupon at t_k for every path still alive at t_k (end >= k)
        if k > 0 and cpn[k] != 0.0:
            cash[end >= k] += cpn[k]

    pv = cash + eq
    if pair_average:
        pv_eff = pv.reshape(2, -1).mean(axis=0)
    else:
        pv_eff = pv
    price = pv_eff.mean()
    se = pv_eff.std(ddof=1) / np.sqrt(len(pv_eff))
    out_pv = {"pv": pv_eff} if keep_pv else {}
    return {
        **out_pv,
        "price": float(price), "se": float(se),
        "ci95": (float(price - 1.96 * se), float(price + 1.96 * se)),
        "frac_called": float(called.mean()),
        "frac_converted_early": float(converted_early.mean()),
        "frac_put": float(put_exercised.mean()),
    }
