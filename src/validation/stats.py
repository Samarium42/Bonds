"""Statistical tools (numpy + math only, so they run anywhere)."""
from __future__ import annotations

import math

import numpy as np


def norm_cdf(x):
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(x) / math.sqrt(2.0)))


def bs_call(S, K, r, q, sigma, T):
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * math.exp(-q * T) * float(norm_cdf(d1)) - K * math.exp(-r * T) * float(norm_cdf(d2))


def ks_normal(x):
    """One-sample KS test of standardised x against N(0,1). Returns (D, asymptotic p-value)."""
    x = np.sort((np.asarray(x) - np.mean(x)) / np.std(x, ddof=1))
    n = len(x)
    F = norm_cdf(x)
    D = max(np.max(np.arange(1, n + 1) / n - F), np.max(F - np.arange(0, n) / n))
    lam = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * D
    p = 2 * sum((-1) ** (j - 1) * math.exp(-2 * j * j * lam * lam) for j in range(1, 101))
    return float(D), float(min(max(p, 0.0), 1.0))


def jarque_bera(x):
    """JB = n/6 (S^2 + (K-3)^2/4) ~ chi2(2) under normality; p = exp(-JB/2)."""
    x = np.asarray(x)
    n = len(x)
    z = (x - x.mean()) / x.std()
    S, K = np.mean(z ** 3), np.mean(z ** 4)
    jb = n / 6.0 * (S ** 2 + (K - 3.0) ** 2 / 4.0)
    return float(jb), float(math.exp(-jb / 2.0)), float(S), float(K)


def acf(x, nlags=20):
    x = np.asarray(x) - np.mean(x)
    denom = np.sum(x * x)
    return np.array([np.sum(x[: len(x) - l] * x[l:]) / denom for l in range(nlags + 1)])


def ljung_box_stat(x, nlags=10):
    """Q = n(n+2) sum_{l=1}^h rho_l^2/(n-l)  ~ chi2(h) under no autocorrelation."""
    n = len(x)
    r = acf(x, nlags)[1:]
    return float(n * (n + 2) * np.sum(r ** 2 / (n - np.arange(1, nlags + 1))))


def replicate(fn, n_reps, seed0=0):
    """Run fn(seed) n_reps times. Returns (mean, between-run SE, all values)."""
    vals = np.array([fn(seed0 + i) for i in range(n_reps)])
    return float(vals.mean()), float(vals.std(ddof=1) / np.sqrt(n_reps)), vals
