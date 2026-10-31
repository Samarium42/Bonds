import math
import numpy as np

from src.contracts import Convertible, SoftCall
from src.pricer.gbm import simulate_gbm
from src.pricer.lsm import price_lsm
from src.pricer.binomial import price_binomial
from src.validation.stats import bs_call, ks_normal, jarque_bera, replicate

S0, R, Q, SIG, T = 100.0, 0.03, 0.0, 0.35, 3.0
NSTEPS = 72  # multiple of T*coupon_freq = 6


def test_gbm_martingale():
    """E[e^{-(r-q)T} S_T] = S0: z-score of the discounted mean must be small."""
    p = simulate_gbm(S0, R, Q, SIG, T, NSTEPS, 200_000, seed=1, antithetic=False)
    x = np.exp(-(R - Q) * T) * p[:, -1]
    z = (x.mean() - S0) / (x.std(ddof=1) / np.sqrt(len(x)))
    assert abs(z) < 3.5, z


def test_gbm_logreturns_are_normal():
    """Step log-returns should be N((r-q-s^2/2)dt, s^2 dt): KS and Jarque-Bera must not reject."""
    p = simulate_gbm(S0, R, Q, SIG, T, NSTEPS, 2_000, seed=2, antithetic=False)
    lr = np.diff(np.log(p), axis=1).ravel()[:20_000]
    D, pv = ks_normal(lr)
    jb, pjb, skew, kurt = jarque_bera(lr)
    assert pv > 0.001 and pjb > 0.001, (D, pv, jb, pjb)
    assert abs(kurt - 3) < 0.1 and abs(skew) < 0.05


def test_lsm_matches_analytic_zero_coupon():
    """q=0, no call/put, s=0, zero coupon: price = F e^{-rT} + CR * BS_call(S0, K=F/CR)."""
    bond = Convertible(conversion_ratio=1.0, maturity=T, coupon=0.0)
    paths = simulate_gbm(S0, R, Q, SIG, T, NSTEPS, 400_000, seed=3)
    res = price_lsm(paths, bond, R, 0.0, pair_average=True)
    exact = 100.0 * math.exp(-R * T) + 1.0 * bs_call(S0, 100.0, R, Q, SIG, T)
    assert abs(res["price"] - exact) < 4 * res["se"] + 0.03, (res, exact)


def test_bond_floor_limit():
    """Stock ~ 0: price -> sum cpn e^{-(r+s)t_j} + F e^{-(r+s)T} (pure bond value)."""
    s = 0.04
    bond = Convertible(conversion_ratio=1.0, maturity=T, coupon=0.03, coupon_freq=2)
    paths = simulate_gbm(0.01, R, Q, 0.2, T, NSTEPS, 20_000, seed=4)
    res = price_lsm(paths, bond, R, s, pair_average=True)
    pbv = sum(bond.coupon_amount * math.exp(-(R + s) * j / 2) for j in range(1, 7)) \
        + 100.0 * math.exp(-(R + s) * T)
    assert abs(res["price"] - pbv) < 1e-6 + 4 * res["se"], (res["price"], pbv)


def _rich_bond():
    return Convertible(
        conversion_ratio=0.8, maturity=T, coupon=0.04, coupon_freq=2,
        soft_call=SoftCall(price=104.0, trigger=170.0, n_required=1, window=1, protection_years=1.0),
        puts={1.0: 100.0, 2.0: 100.0},
    )


def _tree_vs_lsm(bond, steps, n_paths=100_000, seed=5):
    q, s, sig = 0.01, 0.03, 0.30
    tree = price_binomial(S0, bond, R, q, sig, s, n_steps=steps)
    paths = simulate_gbm(S0, R, q, sig, T, steps, n_paths, seed=seed)
    res = price_lsm(paths, bond, R, s, pair_average=True)
    return res, tree


def test_lsm_vs_binomial_same_grid_with_call():
    """Coupons + puts + 1-of-1 soft call + spread + dividends, SAME monitoring grid in both.

    LSM follows a suboptimal exercise policy, so LSM <= tree up to MC noise:
    one-sided check (no more than 4 SE above) and a shortfall bound of 0.5 (~0.5% of price).
    Comparing different grids is invalid: discrete call-monitoring changes the price.
    """
    res, tree = _tree_vs_lsm(_rich_bond(), 72)
    diff = res["price"] - tree
    assert diff < 4 * res["se"], (res, tree)
    assert diff > -0.5, (res, tree)


def test_lsm_vs_binomial_no_call():
    """Same check with the call switched off: isolates put / coupon / spread / decision logic."""
    bond = Convertible(conversion_ratio=0.8, maturity=T, coupon=0.04, coupon_freq=2,
                       puts={1.0: 100.0, 2.0: 100.0})
    res, tree = _tree_vs_lsm(bond, 72)
    diff = res["price"] - tree
    assert diff < 4 * res["se"], (res, tree)
    assert diff > -0.5, (res, tree)


def test_monotone_in_vol_and_spread():
    """Common random numbers: price non-decreasing in sigma, non-increasing in spread."""
    bond = Convertible(conversion_ratio=1.0, maturity=T, coupon=0.02)
    def P(sig, s):
        paths = simulate_gbm(S0, R, Q, sig, T, NSTEPS, 100_000, seed=6)
        return price_lsm(paths, bond, R, s, pair_average=True)["price"]
    assert P(0.45, 0.02) > P(0.25, 0.02)
    assert P(0.35, 0.00) > P(0.35, 0.05)


def test_replication_se_is_honest():
    """The reported within-run SE should match the between-seed std of the estimator (ratio ~ 1)."""
    bond = Convertible(conversion_ratio=1.0, maturity=T, coupon=0.02)
    ses = []
    def one(seed):
        paths = simulate_gbm(S0, R, Q, SIG, T, NSTEPS, 20_000, seed=100 + seed)
        res = price_lsm(paths, bond, R, 0.02, pair_average=True)
        ses.append(res["se"])
        return res["price"]
    _, _, vals = replicate(one, 30)
    ratio = np.mean(ses) / vals.std(ddof=1)
    assert 0.7 < ratio < 1.4, ratio
