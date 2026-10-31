"""Sample paths from the trained flow and map them to the pricer's (n_paths, n_steps+1) price format.

Integrates dx/dt = v_theta(x, t) from t=0 (N(0,I)) to t=1 (data law) with `ode_steps` Euler (or Heun) steps.
"""
from __future__ import annotations

import numpy as np
import torch

from src.flow.cfm_model import VelocityNet


@torch.no_grad()
def sample_standardised(model: VelocityNet, n: int, ode_steps: int = 25, seed: int | None = None,
                        method: str = "heun", batch: int = 4096) -> np.ndarray:
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    outs = []
    for s in range(0, n, batch):
        m = min(batch, n - s)
        x = torch.randn(m, model.N, generator=g)
        h = 1.0 / ode_steps
        for i in range(ode_steps):
            t = torch.full((m,), i * h)
            v = model(x, t)
            if method == "heun":
                x_e = x + h * v
                v2 = model(x_e, torch.full((m,), (i + 1) * h))
                x = x + 0.5 * h * (v + v2)
            else:
                x = x + h * v
        outs.append(x.numpy())
    return np.concatenate(outs)


def martingale_correct(S_raw: np.ndarray, S0: float, r: float, q: float, T: float) -> np.ndarray:
    """Enforce E[S_k] = S0 exp((r-q) t_k) on the grid by rescaling each column by its sample mean.

    S_k <- S0 e^{(r-q)t_k} * S_raw_k / mean_i(S_raw_k).  Changes only the drift term (a deterministic
    per-date factor), not the shape of the return law; the sample-mean estimation error is O(sd/sqrt(n)).
    """
    n_steps = S_raw.shape[1] - 1
    t = np.linspace(0, T, n_steps + 1)
    scale = S0 * np.exp((r - q) * t) / S_raw.mean(axis=0)
    return S_raw * scale[None, :]


def simulate_flow(model: VelocityNet, S0: float, r: float, q: float, T: float, n_paths: int,
                  weekly_vol: float, ode_steps: int = 25, seed: int | None = None) -> np.ndarray:
    """Drop-in for pricer.gbm.simulate_gbm: returns (n_paths, N+1), N = model.N weekly steps.

    Standardised returns are rescaled to `weekly_vol` (demeaned first: the risk-neutral drift is put back
    by martingale_correct), exponentiated, and corrected to a martingale after discounting.
    """
    z = sample_standardised(model, n_paths, ode_steps, seed)
    z = z - z.mean()                       # global demean: drift is set risk-neutrally below
    logret = weekly_vol * z
    X = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(logret, axis=1)], axis=1)
    return martingale_correct(S0 * np.exp(X), S0, r, q, T)
