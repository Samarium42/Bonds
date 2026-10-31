"""Sources of unit-variance weekly path banks Z(rho): (n_paths, N), reused across sigma and spread (common random numbers).

A source is any callable  source(rho: float, n: int, seed: int) -> np.ndarray.
  family_source : exact simulation of the GJR-GARCH-t family (fast; the reference the flow is validated against)
  flow_source   : the rho-conditioned flow matching sampler (slow on CPU; sampled banks are cached on disk)
"""
from __future__ import annotations

import os

import numpy as np

from src.data.family import cond_from_rho, simulate_family


def family_source(N: int = 156):
    def src(rho: float, n: int, seed: int) -> np.ndarray:
        return simulate_family(np.full(n, rho), N, seed=seed)
    return src


def flow_source(model, cache_dir: str = "results/zbank", ode_steps: int = 25):
    from src.flow.sample import sample_standardised
    os.makedirs(cache_dir, exist_ok=True)

    def src(rho: float, n: int, seed: int) -> np.ndarray:
        path = os.path.join(cache_dir, f"flow_rho{rho:.4f}_n{n}_s{seed}.npy")
        if os.path.exists(path):
            return np.load(path)
        Z = sample_standardised(model, n, ode_steps=ode_steps, seed=seed, cond=cond_from_rho(np.full(n, rho)))
        np.save(path, Z)
        return Z
    return src


def cap_bank(Z: np.ndarray, cap: float | None) -> np.ndarray:
    """Truncate weekly returns at +-cap standard deviations and re-standardise to unit variance.

    Why: the fitted GJR-GARCH-t family has a power-law tail (Hill index about 3.6), so E[exp(sigma * sum Z)] is infinite and
    the Monte Carlo price has theoretically infinite variance; single paths with |Z| of 40 to 90 inflated the SE by up to 5x.
    A 10-sigma cap clips about 7e-5 of weekly returns and moves the mean price by about 0.03 (inside the MC noise) while
    cutting the SE by about a third. Real equities' largest weekly moves are about 5 to 6 sigma.
    """
    if cap is None:
        return Z
    Zc = np.clip(Z, -cap, cap)
    return Zc / Zc.std()


def with_cap(source, cap: float | None = 10.0):
    def src(rho: float, n: int, seed: int) -> np.ndarray:
        return cap_bank(source(rho, n, seed), cap)
    return src
