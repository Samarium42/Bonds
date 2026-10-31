"""Parametric path family: daily GJR-GARCH-t aggregated to weekly, indexed by persistence rho.

Because the family is scale-equivariant (multiplying returns by c multiplies omega by c^2 and leaves
(alpha, gamma, beta, nu) unchanged), volatility sigma is applied ANALYTICALLY after sampling. The learned part is the
dynamics: how strongly volatility clusters (rho = alpha + gamma/2 + beta). Weekly sums are scaled to unit variance.
"""
from __future__ import annotations

import numpy as np

RHO_MID, RHO_SCALE = 0.95, 0.03          # conditioning input is c = (rho - RHO_MID) / RHO_SCALE


def cond_from_rho(rho: np.ndarray) -> np.ndarray:
    return ((np.asarray(rho, dtype=np.float32) - RHO_MID) / RHO_SCALE)[:, None]


def simulate_family(rho, n_weeks: int, seed: int, alpha: float = 0.05, gamma: float = 0.06, nu: float = 5.0,
                    days: int = 5, burn: int = 500) -> np.ndarray:
    """rho: array (n_paths,). Returns weekly log-returns (n_paths, n_weeks) with unit unconditional variance."""
    rho = np.asarray(rho, dtype=np.float64)
    n = len(rho)
    beta = rho - alpha - 0.5 * gamma
    assert np.all(beta > 0) and np.all(rho < 1), "rho outside the admissible range for these alpha, gamma"
    rng = np.random.default_rng(seed)
    v = 1.0 / days                                 # daily variance so that the weekly sum has variance 1
    omega = v * (1 - rho)
    total = burn + n_weeks * days
    s2 = np.full(n, v)
    scale = np.sqrt((nu - 2.0) / nu)
    out = np.empty((n, n_weeks * days), dtype=np.float32)
    for t in range(total):
        z = rng.standard_t(nu, size=n) * scale
        r = np.sqrt(s2) * z
        if t >= burn:
            out[:, t - burn] = r
        s2 = omega + (alpha + gamma * (r < 0)) * r * r + beta * s2
    return out.reshape(n, n_weeks, days).sum(axis=2)


def training_bank(n_paths: int, n_weeks: int, rho_range=(0.90, 0.99), seed: int = 0, **kw):
    rng = np.random.default_rng(seed)
    rho = rng.uniform(*rho_range, size=n_paths)
    return simulate_family(rho, n_weeks, seed + 1, **kw), cond_from_rho(rho)
