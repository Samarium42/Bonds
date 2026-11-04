"""Phase 5 tests: jump-to-default pricer against closed forms and an explicit default-time simulation."""
import numpy as np

from src.contracts import Convertible, SoftCall
from src.credal.sweep import default_bond, S0, R as RF, Q, T
from src.credit.jump import compensated_paths, zcb_value, calibrate_lam0, price_jd, hazard
from src.data.family import simulate_family
from src.pricer.lsm import price_lsm
from src.flow.sample import paths_from_weekly
from src.credal.sources import cap_bank

N = 156
WV = 0.5 / np.sqrt(52.0)


def _Z(n=20000, seed=3):
    return cap_bank(simulate_family(np.full(n, 0.95), N, seed=seed), 10.0)


def test_zero_hazard_reduces_to_phase0():
    Z = _Z(8000)
    bond = default_bond()
    base = price_lsm(paths_from_weekly(Z, S0, RF, Q, T, WV), bond, RF, 0.0)["price"]
    jd = price_jd(Z, S0, RF, Q, T, WV, bond, lam0=0.0, beta=0.0, recovery=0.4)["price"]
    assert abs(base - jd) < 1e-9


def test_zcb_closed_form_constant_hazard():
    lam0, rec, dt = 0.07, 0.35, T / N
    lam = np.full((5, N + 1), lam0)
    a = np.exp(-RF * dt) * np.exp(-lam0 * dt)
    c = np.exp(-RF * dt) * (1 - np.exp(-lam0 * dt)) * rec * 100.0
    exact = a ** N * 100.0 + c * (1 - a ** N) / (1 - a)
    assert abs(zcb_value(lam, RF, T, rec) - exact) < 1e-9
    cont = 100 * (np.exp(-(RF + lam0) * T) + rec * lam0 * (1 - np.exp(-(RF + lam0) * T)) / (RF + lam0))
    assert abs(zcb_value(lam, RF, T, rec) - cont) < 0.1          # O(dt) discretisation (recovery paid at end of step)


def test_risky_coupon_bond_closed_form():
    """Convertible with ~zero conversion value, no call or put, equals the deterministic risky coupon bond."""
    bond = Convertible(conversion_ratio=1e-9, maturity=T, coupon=0.05, coupon_freq=2)
    lam0, rec, dt = 0.06, 0.4, T / N
    Z = _Z(4000)
    got = price_jd(Z, S0, RF, Q, T, WV, bond, lam0, 0.0, rec)["price"]
    q_, dfr = np.exp(-lam0 * dt), np.exp(-RF * dt)
    cpn = bond.coupon_vector(N)
    V = 100.0 + cpn[N]
    for k in range(N - 1, -1, -1):
        V = dfr * (q_ * V + (1 - q_) * rec * 100.0)
        if k > 0:
            V += cpn[k] * (1.0)
    assert abs(got - V) < 1e-6


def test_compensation_is_martingale_and_matches_explicit_default_simulation():
    """Independent check of drift compensation and of survival weighting by simulating actual default times."""
    Z = _Z(60000, seed=11)
    lam0, beta, rec = 0.08, 1.5, 0.3
    S, lam, w, err = compensated_paths(Z, S0, RF, Q, T, WV, lam0, beta)
    assert err < 1e-6
    dt = T / N
    rng = np.random.default_rng(5)
    E = rng.exponential(size=S.shape[0])
    cum = np.concatenate([np.zeros((S.shape[0], 1)), dt * np.cumsum(lam[:, :-1], axis=1)], axis=1)   # cum[:, k] = hazard up to t_k
    surv_T = cum[:, N] < E                                                                            # alive at T
    # E[S_T 1{alive}] e^{-rT} should equal S0 e^{-qT}
    lhs = np.exp(-RF * T) * S[:, N] * surv_T
    target = S0 * np.exp(-Q * T)
    assert abs(lhs.mean() - target) < 4 * lhs.std(ddof=1) / np.sqrt(len(lhs))
    # European payoff max(100, 0.8 S_T) with recovery on default: explicit vs survival-weighted
    dk = np.where(surv_T, N, (cum >= E[:, None]).argmax(axis=1))    # first step index whose cumulative hazard crosses E
    pay_explicit = np.where(surv_T, np.maximum(100.0, 0.8 * S[:, N]) * np.exp(-RF * T),
                            rec * 100.0 * np.exp(-RF * dk * dt))
    wT = w[:, N]
    pay_weighted = np.exp(-RF * T) * wT * np.maximum(100.0, 0.8 * S[:, N])
    dens = (w[:, :-1] - w[:, 1:])                               # P(default in step k | path)
    disc = np.exp(-RF * dt * (np.arange(N) + 1))
    pay_weighted = pay_weighted + rec * 100.0 * (dens * disc).sum(axis=1)
    se = np.sqrt(pay_explicit.var(ddof=1) / len(pay_explicit))
    assert abs(pay_explicit.mean() - pay_weighted.mean()) < 4 * se
    assert pay_weighted.std(ddof=1) < pay_explicit.std(ddof=1)     # survival weighting removes default-time noise


def test_calibration_matches_flat_spread_zcb():
    Z = _Z(10000, seed=2)
    for beta, rec, s in [(0.0, 0.4, 0.04), (1.0, 0.4, 0.04), (2.0, 0.0, 0.07)]:
        l0 = calibrate_lam0(Z, S0, RF, Q, T, WV, beta, rec, s)
        _, lam, _, err = compensated_paths(Z, S0, RF, Q, T, WV, l0, beta)
        assert err < 1e-6
        assert abs(zcb_value(lam, RF, T, rec) - 100 * np.exp(-(RF + s) * T)) < 1e-4
    # beta = 0: lam0 is close to s / (1 - recovery)
    l0 = calibrate_lam0(Z, S0, RF, Q, T, WV, 0.0, 0.4, 0.04)
    assert abs(l0 - 0.04 / 0.6) < 0.01


def test_hazard_shape():
    S = np.array([[50.0, 100.0, 200.0]])
    h = hazard(S, 100.0, 0.1, 2.0)
    assert np.allclose(h, [[0.4, 0.1, 0.025]])
