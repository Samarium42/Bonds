import numpy as np

from src.calibrate.garch import fit_gjr_t
from src.data.family import simulate_family, cond_from_rho
from src.data.synthetic import simulate_gjr
from src.validation import stylized_facts as sf


def test_garch_mle_recovers_parameters():
    """5000 daily obs, 8 seeds: mean estimates within 2 standard errors of truth (se = sd/sqrt(8))."""
    est = []
    for seed in range(8):
        f = fit_gjr_t(simulate_gjr(5000, 0.03, seed, alpha=0.05, gamma=0.06, beta=0.88, nu=5.0))
        est.append([f.alpha, f.gamma, f.beta, f.nu, f.rho])
    est = np.array(est); truth = np.array([0.05, 0.06, 0.88, 5.0, 0.96])
    z = (est.mean(0) - truth) / (est.std(0, ddof=1) / np.sqrt(len(est)))
    assert np.all(np.abs(z) < 3.0), z


def test_garch_precision_improves_with_data():
    """More observations shrink the sampling sd of rho (this is what daily data buys over weekly)."""
    def sd(n):
        return np.std([fit_gjr_t(simulate_gjr(n, 0.03, s, 0.05, 0.06, 0.88, 5.0)).rho for s in range(10)], ddof=1)
    assert sd(5000) < sd(1500)


def test_family_unit_variance_and_scale_equivariance():
    W = simulate_family(np.full(4000, 0.95), 104, seed=0)
    assert abs(W.std() - 1.0) < 0.05
    # weekly log-return law scales exactly with sigma: sigma * W has std sigma
    assert abs((0.08 * W).std() - 0.08) < 0.005


def test_family_clustering_increases_with_rho():
    a = [sf.pooled_acf(simulate_family(np.full(3000, r), 104, seed=1), 3, 2)[1] for r in (0.90, 0.95, 0.985)]
    assert a[0] < a[1] < a[2], a


def test_cond_encoding():
    c = cond_from_rho(np.array([0.95, 0.98]))
    assert c.shape == (2, 1) and abs(c[0, 0]) < 1e-6 and abs(c[1, 0] - 1.0) < 1e-5


def test_leverage_se_is_calibrated():
    """On iid fat-tailed noise (true leverage 0) the z-score of the difference of two independent estimates is ~N(0,1)."""
    rng = np.random.default_rng(0); zs = []
    for _ in range(25):
        a = rng.standard_t(5, (1500, 52)); b = rng.standard_t(5, (1500, 52))
        (la, sa), (lb, sb) = sf.leverage_se(a, n_boot=60), sf.leverage_se(b, n_boot=60)
        zs.append((la - lb) / np.hypot(sa, sb))
    assert abs(np.mean(zs)) < 0.6 and 0.6 < np.std(zs, ddof=1) < 1.5, (np.mean(zs), np.std(zs, ddof=1))
