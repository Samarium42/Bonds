"""Controlled backtest of credal price bounds against a reality with KNOWN price.

Reality = two-state regime-switching volatility with Student-t shocks (NOT in the GJR-GARCH family the flow is trained on,
so the test includes model misspecification). The true price P* is computed once per scenario with many paths.
A modeler sees only noisy estimates  sigma_hat = sigma* e^{eta},  s_hat = s* e^{nu}  (eta ~ N(0,e_sig^2), nu ~ N(0,e_s^2)),
builds the credal box  sigma in sigma_hat e^{+-z e_sig},  s in s_hat e^{+-z e_s},  rho in the full calibrated range, and reports
[P_low, P_high]. We count how often P* lies inside, and compare with a GBM point estimate at (sigma_hat, s_hat).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import binomtest, norm
from scipy.interpolate import RegularGridInterpolator

from src.credal.sweep import Surface, box_bounds, price_one


def regime_switch_Z(n: int, N: int, seed: int, ratio: float = 2.0, stay: float = 0.97, nu: float = 5.0) -> np.ndarray:
    """Unit-variance weekly returns: sd_state in {lo, hi=ratio*lo}, symmetric two-state Markov chain with P(stay)=stay."""
    rng = np.random.default_rng(seed)
    lo = np.sqrt(2.0 / (1.0 + ratio ** 2)); hi = ratio * lo           # 0.5 lo^2 + 0.5 hi^2 = 1
    state = rng.random(n) < 0.5
    Z = np.empty((n, N))
    scale = np.sqrt((nu - 2.0) / nu)
    for t in range(N):
        Z[:, t] = np.where(state, hi, lo) * rng.standard_t(nu, n) * scale
        state = state ^ (rng.random(n) > stay)
    return Z


def wilson(k: int, n: int, level: float = 0.95):
    z = norm.ppf(0.5 + level / 2); p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(c - h), float(c + h)


def gbm_surface(Zg: np.ndarray, bond, sigmas, spreads):
    """Price grid (sigma, s) for the GBM point-estimate model, from one GBM bank (common random numbers)."""
    P = np.array([[price_one(Zg, float(sg), float(sp), bond)[0] for sp in spreads] for sg in sigmas])
    return RegularGridInterpolator((np.asarray(sigmas), np.asarray(spreads)), P)


def run_scenario(A: Surface, B: Surface, G, bond, Z_real: np.ndarray, sig_star: float, s_star: float, n_rep: int, seed: int,
                 e_sig: float = 0.15, e_s: float = 0.20, z: float = 1.645, rho_range=(0.90, 0.985)) -> dict:
    p_star, se_star = price_one(Z_real, sig_star, s_star, bond)
    rng = np.random.default_rng(seed)
    cov, clipped, low, high, pt = [], 0, [], [], []
    for _ in range(n_rep):
        sg_hat = sig_star * np.exp(rng.normal(0, e_sig)); s_hat = s_star * np.exp(rng.normal(0, e_s))
        sig_box = (sg_hat * np.exp(-z * e_sig), sg_hat * np.exp(z * e_sig)); s_box = (s_hat * np.exp(-z * e_s), s_hat * np.exp(z * e_s))
        if sig_box[1] > A.sigmas[-1] or sig_box[0] < A.sigmas[0] or s_box[1] > A.spreads[-1] or s_box[0] < A.spreads[0]:
            clipped += 1
        b = box_bounds(A, B, sig_box, rho_range, s_box)
        low.append(b["low"]); high.append(b["high"]); cov.append(b["low"] <= p_star <= b["high"])
        pt.append(float(G([[np.clip(sg_hat, A.sigmas[0], A.sigmas[-1]), np.clip(s_hat, A.spreads[0], A.spreads[-1])]])[0]))
    cov = np.array(cov); low, high, pt = map(np.array, (low, high, pt))
    err = pt - p_star
    k = int(cov.sum())
    return {"p_star": p_star, "se_star": se_star, "n": n_rep, "covered": k, "coverage": k / n_rep, "wilson95": wilson(k, n_rep),
            "mean_width": float((high - low).mean()), "mean_half_width": float((high - low).mean() / 2),
            "gbm_mean_err": float(err.mean()), "gbm_mean_abs_err": float(np.abs(err).mean()),
            "gbm_frac_err_gt_1": float((np.abs(err) > 1.0).mean()), "gbm_frac_inside_interval": float(((pt >= low) & (pt <= high)).mean()),
            "boxes_clipped_by_grid": clipped}


def expected_coverage_if_truth_in_box(z: float = 1.645) -> float:
    """Probability that sigma* and s* both lie in their boxes: (2 Phi(z) - 1)^2 (boxes are +-z error-scales around the
    noisy estimates). If the price surface is monotone in sigma and s and the model is right, coverage of P* is at least this."""
    return float((2 * norm.cdf(z) - 1) ** 2)
