"""Gaussian-free calibration of GJR-GARCH(1,1) with Student-t innovations by maximum likelihood.

Model:  r_t = sigma_t z_t,  z_t ~ t_nu scaled to unit variance,
        sigma_t^2 = omega + (alpha + gamma 1{r_{t-1}<0}) r_{t-1}^2 + beta sigma_{t-1}^2.
Persistence rho = alpha + gamma/2 + beta (<1); unconditional variance v = omega / (1 - rho).

Log-likelihood per observation (unit-variance t):
  l_t = lgamma((nu+1)/2) - lgamma(nu/2) - 0.5 log(pi (nu-2)) - log sigma_t - (nu+1)/2 log(1 + r_t^2 / ((nu-2) sigma_t^2)).
Because the numerator a_t = (alpha + gamma 1{r_{t-1}<0}) r_{t-1}^2 is known from the data, the variance recursion is
the linear filter  s2_t = omega + a_t + beta s2_{t-1}  and is evaluated with scipy.signal.lfilter (no python loop).
Variance targeting: omega = v (1 - rho) with v the sample variance, which removes one parameter and stabilises the fit.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import gammaln


@dataclass
class GJRFit:
    alpha: float
    gamma: float
    beta: float
    nu: float
    var: float            # unconditional variance of the (daily) series
    loglik: float
    n: int

    @property
    def rho(self) -> float:
        return self.alpha + 0.5 * self.gamma + self.beta

    @property
    def omega(self) -> float:
        return self.var * (1 - self.rho)


def _sigma2(r, alpha, gamma, beta, v):
    rho = alpha + 0.5 * gamma + beta
    omega = v * (1 - rho)
    a = np.zeros_like(r)
    a[1:] = (alpha + gamma * (r[:-1] < 0)) * r[:-1] ** 2
    s2 = lfilter([1.0], [1.0, -beta], omega + a, zi=[beta * v])[0]
    return s2


def _nll(p, r, v):
    alpha, gamma, beta, nu = p
    if alpha < 0 or gamma < -alpha or beta < 0 or alpha + 0.5 * gamma + beta >= 0.9995 or nu <= 2.05:
        return 1e10
    s2 = _sigma2(r, alpha, gamma, beta, v)
    if np.any(s2 <= 0):
        return 1e10
    ll = (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
          - 0.5 * np.log(s2) - (nu + 1) / 2 * np.log1p(r ** 2 / ((nu - 2) * s2)))
    return -float(ll.sum())


def fit_gjr_t(r: np.ndarray) -> GJRFit:
    r = np.asarray(r, dtype=float)
    r = r - r.mean()
    v = float(r.var())
    best = None
    for p0 in ([0.05, 0.05, 0.88, 6.0], [0.03, 0.03, 0.93, 5.0], [0.08, 0.08, 0.80, 8.0]):
        res = minimize(_nll, p0, args=(r, v), method="Nelder-Mead",
                       options={"xatol": 1e-5, "fatol": 1e-7, "maxiter": 4000})
        if best is None or res.fun < best.fun:
            best = res
    a, g, b, nu = best.x
    return GJRFit(a, g, b, nu, v, -best.fun, len(r))


def bootstrap_ci(r: np.ndarray, n_boot: int = 30, block: int = 60, seed: int = 0) -> dict:
    """Moving-block bootstrap of the refit: returns 5%/95% percentiles of rho, gamma, nu."""
    rng = np.random.default_rng(seed); r = np.asarray(r); n = len(r); nb = int(np.ceil(n / block))
    out = {"rho": [], "gamma": [], "nu": []}
    for _ in range(n_boot):
        st = rng.integers(0, n - block + 1, nb)
        f = fit_gjr_t(np.concatenate([r[s:s + block] for s in st])[:n])
        out["rho"].append(f.rho); out["gamma"].append(f.gamma); out["nu"].append(f.nu)
    return {k: (float(np.percentile(v, 5)), float(np.percentile(v, 95))) for k, v in out.items()}


def parametric_bootstrap_ci(fit: GJRFit, n_boot: int = 40, seed: int = 0) -> dict:
    """Parametric bootstrap of the MLE: simulate from the fitted model with the same n, refit, and report the
    basic (bias-corrected) 90% interval  [2*theta_hat - q95, 2*theta_hat - q05].

    Preferred over the moving-block bootstrap here: blocks cut the long volatility memory and bias the refitted
    persistence downward, which would understate the credal range.
    """
    from src.data.synthetic import simulate_gjr
    out = {"rho": [], "gamma": [], "nu": []}
    for b in range(n_boot):
        r = simulate_gjr(fit.n, float(np.sqrt(fit.var)), seed + b, fit.alpha, fit.gamma, fit.beta, fit.nu)
        f = fit_gjr_t(r)
        out["rho"].append(f.rho); out["gamma"].append(f.gamma); out["nu"].append(f.nu)
    hat = {"rho": fit.rho, "gamma": fit.gamma, "nu": fit.nu}
    return {k: (float(2 * hat[k] - np.percentile(v, 95)), float(2 * hat[k] - np.percentile(v, 5))) for k, v in out.items()}
