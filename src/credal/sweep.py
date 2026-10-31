"""Credal-set price sweep for a convertible bond.

Credal set  Theta = [sigma_lo, sigma_hi] x [rho_lo, rho_hi] x [s_lo, s_hi]
  sigma: annual volatility (applied analytically to unit-variance weekly paths)
  rho  : volatility persistence (the flow's learned condition)
  s    : credit spread (discounts the cash component in the pricer)
Lower / upper prevision:  P_low = inf_{theta in Theta} P(theta),  P_high = sup_{theta in Theta} P(theta).

Estimation details that keep the interval honest
  * Common random numbers: one path bank per rho is reused for every (sigma, s), so the surface is smooth.
  * Winner's curse: inf/sup of a noisy surface is biased outward. Two independent banks A and B are drawn; the extreme
    point is SELECTED on A and RE-EVALUATED on B. We report both; their gap measures the selection bias.
  * The surface is evaluated on a grid and interpolated bilinearly in (sigma, s) at each rho. `interp_error` checks the
    interpolation against direct pricing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from src.contracts import Convertible, SoftCall
from src.flow.sample import paths_from_weekly
from src.pricer.lsm import price_lsm

S0, R, Q, T = 100.0, 0.03, 0.01, 3.0


def default_bond() -> Convertible:
    return Convertible(conversion_ratio=0.8, maturity=T, coupon=0.04, coupon_freq=2,
                       soft_call=SoftCall(price=104.0, trigger=170.0, n_required=1, window=1, protection_years=1.0),
                       puts={1.0: 100.0, 2.0: 100.0})


def price_one(Z: np.ndarray, sigma: float, spread: float, bond: Convertible) -> tuple[float, float]:
    paths = paths_from_weekly(Z, S0, R, Q, T, sigma / np.sqrt(52.0))
    res = price_lsm(paths, bond, R, spread)
    return res["price"], res["se"]


@dataclass
class Surface:
    sigmas: np.ndarray
    rhos: np.ndarray
    spreads: np.ndarray
    P: np.ndarray       # (n_sigma, n_rho, n_spread) prices
    SE: np.ndarray

    def interpolators(self):
        """One bilinear interpolator over (sigma, s) per rho index."""
        return [RegularGridInterpolator((self.sigmas, self.spreads), self.P[:, k, :]) for k in range(len(self.rhos))]


def build_surface(source, bond: Convertible, sigmas, rhos, spreads, n_paths: int, seed: int, verbose: bool = False) -> Surface:
    sigmas, rhos, spreads = map(np.asarray, (sigmas, rhos, spreads))
    P = np.empty((len(sigmas), len(rhos), len(spreads))); SE = np.empty_like(P)
    for k, rho in enumerate(rhos):
        Z = source(float(rho), n_paths, seed + k)
        for i, sg in enumerate(sigmas):
            for j, sp in enumerate(spreads):
                P[i, k, j], SE[i, k, j] = price_one(Z, float(sg), float(sp), bond)
        if verbose:
            print(f"  rho={rho:.3f} done", flush=True)
    return Surface(sigmas, rhos, spreads, P, SE)


def box_bounds(A: Surface, B: Surface, sig_range, rho_range, s_range, n_fine: int = 9) -> dict:
    """Lower / upper prevision over the box. Selection on surface A, re-evaluation on independent surface B."""
    sig_range = (max(sig_range[0], A.sigmas[0]), min(sig_range[1], A.sigmas[-1]))
    s_range = (max(s_range[0], A.spreads[0]), min(s_range[1], A.spreads[-1]))
    ks = [k for k, r in enumerate(A.rhos) if rho_range[0] - 1e-9 <= r <= rho_range[1] + 1e-9]
    if not ks:
        raise ValueError("rho range contains no grid point")
    sg, sp = np.meshgrid(np.linspace(*sig_range, n_fine), np.linspace(*s_range, n_fine), indexing="ij")
    pts = np.column_stack([sg.ravel(), sp.ravel()])
    IA, IB = A.interpolators(), B.interpolators()
    pa = np.stack([IA[k](pts) for k in ks]); pb = np.stack([IB[k](pts) for k in ks])     # (n_rho_in, n_pts)
    ilo, ihi = np.unravel_index(pa.argmin(), pa.shape), np.unravel_index(pa.argmax(), pa.shape)
    return {"low_one_stage": float(pa.min()), "high_one_stage": float(pa.max()),
            "low": float(pb[ilo]), "high": float(pb[ihi]),
            "argmin": (float(pts[ilo[1], 0]), float(A.rhos[ks[ilo[0]]]), float(pts[ilo[1], 1])),
            "argmax": (float(pts[ihi[1], 0]), float(A.rhos[ks[ihi[0]]]), float(pts[ihi[1], 1])),
            "mid_rho_point": None}


def interp_error(A: Surface, source, bond, n_paths: int, seed: int, n_test: int = 6, rng_seed: int = 0):
    """Max |interpolated - directly priced| at random interior (sigma, rho_grid, s) points, using the SAME bank as A."""
    rng = np.random.default_rng(rng_seed); errs = []
    IA = A.interpolators()
    for _ in range(n_test):
        k = int(rng.integers(len(A.rhos)))
        sg = float(rng.uniform(A.sigmas[0], A.sigmas[-1])); sp = float(rng.uniform(A.spreads[0], A.spreads[-1]))
        Z = source(float(A.rhos[k]), n_paths, seed + k)
        direct, _ = price_one(Z, sg, sp, bond)
        errs.append(abs(float(IA[k]([[sg, sp]])[0]) - direct))
    return float(max(errs)), float(np.mean(errs))


def rescale_bond(bond: Convertible, spot: float, ref: float = 100.0) -> Convertible:
    """Express a bond written for a stock at `spot` in the pricer's S0=100 units.

    Payoffs depend on conversion_ratio * S and on absolute trigger levels, so with S' = S * ref / spot
    (conversion_ratio -> CR * spot / ref, trigger -> trigger * ref / spot) every payoff is unchanged.
    """
    from dataclasses import replace
    k = spot / ref
    sc = bond.soft_call
    return replace(bond, conversion_ratio=bond.conversion_ratio * k,
                   soft_call=None if sc is None else replace(sc, trigger=sc.trigger / k))


def quality_report(S: Surface, se_thresh: float = 1.0) -> dict:
    """Flag grid nodes whose Monte Carlo SE is too large to trust (typically high sigma with high rho)."""
    bad = np.argwhere(S.SE > se_thresh)
    return {"max_se": float(S.SE.max()), "mean_se": float(S.SE.mean()), "n_bad": int(len(bad)), "n_nodes": int(S.SE.size),
            "bad_nodes": [(float(S.sigmas[i]), float(S.rhos[k]), float(S.spreads[j])) for i, k, j in bad[:10]]}
