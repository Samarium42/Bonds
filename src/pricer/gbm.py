"""Risk-neutral GBM path simulator (Phase 0 ground truth).

dS = (r - q) S dt + sigma S dW   =>   S_{k+1} = S_k exp((r - q - sigma^2/2) dt + sigma sqrt(dt) Z)
Exact in distribution on the grid (no discretisation error in the stock law).
"""
from __future__ import annotations

import numpy as np


def simulate_gbm(S0: float, r: float, q: float, sigma: float, T: float,
                 n_steps: int, n_paths: int, seed: int | None = None,
                 antithetic: bool = True) -> np.ndarray:
    """Return array (n_paths, n_steps+1). If antithetic, rows [0:N/2] and [N/2:N] are mirrors."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    if antithetic:
        if n_paths % 2:
            raise ValueError("n_paths must be even with antithetic=True")
        z = rng.standard_normal((n_paths // 2, n_steps))
        z = np.vstack([z, -z])
    else:
        z = rng.standard_normal((n_paths, n_steps))
    log_inc = (r - q - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = S0
    paths[:, 1:] = S0 * np.exp(np.cumsum(log_inc, axis=1))
    return paths
