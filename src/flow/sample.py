"""Sample paths from the trained flow and map them to the pricer's (n_paths, n_steps+1) price format.

Integrates dx/dt = v_theta(x, t) from t=0 (N(0,I)) to t=1 (data law) with `ode_steps` Euler (or Heun) steps.
"""
from __future__ import annotations

import numpy as np
import torch

from src.flow.cfm_model import VelocityNet


@torch.no_grad()
def sample_standardised(model: VelocityNet, n: int, ode_steps: int = 25, seed: int | None = None,
                        method: str = "heun", batch: int = 4096, cond: np.ndarray | None = None) -> np.ndarray:
    """cond: (n, cond_dim) array, or None for an unconditional model."""
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    outs = []
    for s in range(0, n, batch):
        m = min(batch, n - s)
        x = torch.randn(m, model.N, generator=g)
        c = None if cond is None else torch.from_numpy(np.asarray(cond[s:s + m], dtype=np.float32))
        h = 1.0 / ode_steps
        for i in range(ode_steps):
            t = torch.full((m,), i * h)
            v = model(x, t, c)
            if method == "heun":
                x_e = x + h * v
                v2 = model(x_e, torch.full((m,), (i + 1) * h), c)
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


def paths_from_weekly(Z: np.ndarray, S0: float, r: float, q: float, T: float, weekly_vol: float) -> np.ndarray:
    """Unit-variance weekly returns Z (n_paths, N) -> risk-neutral price paths (n_paths, N+1).

    Global demean (drift is reinstated risk-neutrally), scale by weekly_vol, exponentiate, then martingale_correct.
    Shared by the flow sampler and by the reference simulator so both go through IDENTICAL post-processing.
    """
    Z = Z - Z.mean()
    X = np.concatenate([np.zeros((Z.shape[0], 1)), np.cumsum(weekly_vol * Z, axis=1)], axis=1)
    return martingale_correct(S0 * np.exp(X), S0, r, q, T)


def simulate_flow(model: VelocityNet, S0: float, r: float, q: float, T: float, n_paths: int,
                  weekly_vol: float, ode_steps: int = 25, seed: int | None = None,
                  rho: float | None = None) -> np.ndarray:
    """Drop-in for pricer.gbm.simulate_gbm: returns (n_paths, N+1), N = model.N weekly steps.

    rho is the volatility-persistence condition (required for a conditional model); sigma enters through weekly_vol.
    """
    cond = None
    if model.cond_dim:
        from src.data.family import cond_from_rho
        cond = cond_from_rho(np.full(n_paths, rho))
    Z = sample_standardised(model, n_paths, ode_steps, seed, cond=cond)
    return paths_from_weekly(Z, S0, r, q, T, weekly_vol)
