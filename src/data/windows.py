"""Turn return series into fixed-length training windows.

Per-name standardisation uses the TRAIN period only (no look-ahead):  z = r / sigma_train(name).
Pooling standardised names is what lets the basket add data; the target vol is restored at sampling time.
"""
from __future__ import annotations

import numpy as np


def standardise(series: dict, train_frac: float = 0.75):
    """Return ({name: z_series}, {name: sigma_train}). sigma from first train_frac of each series."""
    z, sig = {}, {}
    for k, x in series.items():
        n_tr = int(len(x) * train_frac)
        s = float(np.std(x[:n_tr], ddof=1))
        sig[k] = s
        z[k] = np.asarray(x) / s
    return z, sig


def sliding_windows(x: np.ndarray, N: int, stride: int = 1) -> np.ndarray:
    idx = np.arange(0, len(x) - N + 1, stride)
    return np.stack([x[i:i + N] for i in idx]) if len(idx) else np.empty((0, N))


def stationary_bootstrap(x: np.ndarray, N: int, n_windows: int, mean_block: float, rng) -> np.ndarray:
    """Politis-Romano: blocks of Geometric(1/mean_block) length, wrap-around. Preserves marginals and
    dependence up to ~mean_block lags; destroys dependence across block joins."""
    n = len(x)
    p = 1.0 / mean_block
    out = np.empty((n_windows, N))
    for w in range(n_windows):
        i = rng.integers(n)
        for k in range(N):
            out[w, k] = x[i]
            i = rng.integers(n) if rng.random() < p else (i + 1) % n
    return out


def build_train_set(z: dict, N: int, train_frac: float = 0.75, stride: int = 4,
                    n_boot_per_name: int = 0, mean_block: float = 26.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    parts = []
    for k, x in z.items():
        tr = x[: int(len(x) * train_frac)]
        parts.append(sliding_windows(tr, N, stride))
        if n_boot_per_name:
            parts.append(stationary_bootstrap(tr, N, n_boot_per_name, mean_block, rng))
    return np.concatenate(parts).astype(np.float32)


def heldout_windows(z_one: np.ndarray, N: int, train_frac: float = 0.75, stride: int = 4) -> np.ndarray:
    te = z_one[int(len(z_one) * train_frac):]
    return sliding_windows(te, N, stride).astype(np.float32)
