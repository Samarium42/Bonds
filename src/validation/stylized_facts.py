"""Statistical comparison of generated vs real (or reference) standardised return windows.

All functions take arrays of shape (n_windows, N). Tests are two-sided unless stated.
"""
from __future__ import annotations

import math

import numpy as np

from src.validation.stats import acf, ljung_box_stat


# ---------- marginal shape ----------
def moments(x: np.ndarray) -> dict:
    x = x.ravel()
    z = (x - x.mean()) / x.std()
    return {"std": float(x.std()), "skew": float(np.mean(z ** 3)), "ex_kurt": float(np.mean(z ** 4) - 3.0)}


def hill_alpha(x: np.ndarray, frac: float = 0.05):
    """Hill estimator of the tail index of |x|: alpha = 1 / mean_{i<=k} log(x_(i) / x_(k+1)).
    Asymptotic SE = alpha / sqrt(k). Smaller alpha = fatter tail (Gaussian: alpha -> infinity)."""
    a = np.sort(np.abs(x.ravel()))[::-1]
    k = max(int(len(a) * frac), 10)
    alpha = 1.0 / np.mean(np.log(a[:k] / a[k]))
    return float(alpha), float(alpha / math.sqrt(k))


def ks_two_sample(a: np.ndarray, b: np.ndarray):
    """Two-sample KS: D = sup |F_a - F_b|; asymptotic p via Kolmogorov distribution. Observations are
    treated as iid, which overstates the evidence when windows overlap: read p as indicative."""
    a, b = np.sort(a.ravel()), np.sort(b.ravel())
    allv = np.concatenate([a, b])
    D = np.max(np.abs(np.searchsorted(a, allv, side="right") / len(a) - np.searchsorted(b, allv, side="right") / len(b)))
    ne = len(a) * len(b) / (len(a) + len(b))
    lam = (math.sqrt(ne) + 0.12 + 0.11 / math.sqrt(ne)) * D
    p = 2 * sum((-1) ** (j - 1) * math.exp(-2 * j * j * lam * lam) for j in range(1, 101))
    return float(D), float(min(max(p, 0.0), 1.0))


# ---------- dependence ----------
def pooled_acf(W: np.ndarray, nlags: int = 20, power: int = 1) -> np.ndarray:
    """Average of per-window sample ACFs of |x|^power (power=2: squared returns => volatility clustering)."""
    y = np.abs(W) ** power if power != 1 else W
    return np.mean([acf(w, nlags) for w in y], axis=0)


def ljung_box_sq(W: np.ndarray, nlags: int = 10):
    """Ljung-Box on squared returns of the concatenated series: Q ~ chi2(nlags) under no ARCH effect."""
    from scipy.stats import chi2
    q = ljung_box_stat(W.ravel() ** 2, nlags)
    return q, float(chi2.sf(q, nlags))


def leverage(W: np.ndarray, nlags: int = 5) -> np.ndarray:
    """corr(r_t, r_{t+l}^2), l=1..nlags; negative for equities (down moves raise future vol)."""
    out = []
    for l in range(1, nlags + 1):
        a = W[:, :-l].ravel(); b = (W[:, l:] ** 2).ravel()
        out.append(np.corrcoef(a, b)[0, 1])
    return np.array(out)


# ---------- joint law ----------
def energy_distance(A: np.ndarray, B: np.ndarray) -> float:
    """E = 2 E|X-Y| - E|X-X'| - E|Y-Y'| >= 0, = 0 iff the two laws coincide (V-statistic)."""
    from scipy.spatial.distance import cdist
    return float(2 * cdist(A, B).mean() - cdist(A, A).mean() - cdist(B, B).mean())


def energy_test(A: np.ndarray, B: np.ndarray, n_perm: int = 200, max_n: int = 500, seed: int = 0):
    """Permutation test of H0: A and B are from the same law, with the energy distance as statistic.
    Returns (E_obs, p) where p = (1 + #{E_perm >= E_obs}) / (1 + n_perm)."""
    rng = np.random.default_rng(seed)
    A = A[rng.permutation(len(A))[:max_n]]; B = B[rng.permutation(len(B))[:max_n]]
    from scipy.spatial.distance import cdist
    Z = np.concatenate([A, B]); D = cdist(Z, Z); n = len(A)
    def e(ix, iy):
        return 2 * D[np.ix_(ix, iy)].mean() - D[np.ix_(ix, ix)].mean() - D[np.ix_(iy, iy)].mean()
    idx = np.arange(len(Z)); obs = e(idx[:n], idx[n:])
    cnt = sum(e(*np.split(rng.permutation(idx), [n])) >= obs for _ in range(n_perm))
    return float(obs), float((1 + cnt) / (1 + n_perm))


def memorisation_ratio(gen: np.ndarray, train: np.ndarray, heldout: np.ndarray, max_n: int = 400, seed: int = 0):
    """median NN-distance(gen -> train) / median NN-distance(heldout -> train). A ratio << 1 means the
    generator reproduces training windows rather than the law (overfitting)."""
    from scipy.spatial.distance import cdist
    rng = np.random.default_rng(seed)
    g = gen[rng.permutation(len(gen))[:max_n]]; h = heldout[rng.permutation(len(heldout))[:max_n]]
    t = train[rng.permutation(len(train))[:2000]]
    return float(np.median(cdist(g, t).min(1)) / np.median(cdist(h, t).min(1)))


# ---------- report ----------
def block_bootstrap_ci(stat, x: np.ndarray, block: int = 26, n_boot: int = 200, seed: int = 0):
    """Moving-block bootstrap 95% CI of stat(1-d series): respects the dependence that an iid bootstrap would ignore."""
    rng = np.random.default_rng(seed); n = len(x); nb = int(np.ceil(n / block))
    vals = []
    for _ in range(n_boot):
        st = rng.integers(0, n - block + 1, nb)
        vals.append(stat(np.concatenate([x[s:s + block] for s in st])[:n]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def report(real: np.ndarray, gen: np.ndarray, train: np.ndarray | None = None, nlags: int = 10) -> dict:
    out = {"real_moments": moments(real), "gen_moments": moments(gen)}
    out["hill_real"], out["hill_gen"] = hill_alpha(real), hill_alpha(gen)
    out["ks_two_sample"] = ks_two_sample(real, gen)
    out["acf_sq_real"] = pooled_acf(real, nlags, 2).tolist()
    out["acf_sq_gen"] = pooled_acf(gen, nlags, 2).tolist()
    out["ljung_box_sq_real"] = ljung_box_sq(real, nlags)
    out["ljung_box_sq_gen"] = ljung_box_sq(gen, nlags)
    out["leverage_real"] = leverage(real).tolist()
    out["leverage_gen"] = leverage(gen).tolist()
    out["energy"] = energy_test(real, gen)
    if train is not None:
        out["memorisation_ratio"] = memorisation_ratio(gen, train, real)
    return out
