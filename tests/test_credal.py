import numpy as np

from src.credal.backtest import regime_switch_Z, wilson, expected_coverage_if_truth_in_box
from src.credal.sources import cap_bank
from src.credal.sweep import Surface, box_bounds


def _surface(P, sig=(0.3, 0.6, 0.9), rho=(0.9, 0.95), spr=(0.01, 0.05, 0.09)):
    return Surface(np.array(sig), np.array(rho), np.array(spr), P, np.zeros_like(P))


def test_cap_bank_unit_variance_and_bound():
    Z = np.random.default_rng(0).standard_t(3, (5000, 52))
    C = cap_bank(Z, 10.0)
    assert abs(C.std() - 1.0) < 1e-9 and np.abs(C).max() <= 10.0 / Z.clip(-10, 10).std() + 1e-9


def test_regime_switch_unit_variance_and_clustering():
    Z = regime_switch_Z(30000, 104, 0, ratio=3.0, stay=0.99)
    assert abs(Z.var() - 1.0) < 0.05
    sq = Z ** 2
    # theory: corr(Z_t^2, Z_{t+1}^2) = Var(sd^2) (2 stay - 1) / (E[sd^4] E[eps^4] - 1), with t5 shocks (kurtosis 9)
    lo2 = 2.0 / (1 + 9.0); hi2 = 9.0 * lo2
    var_sd2 = 0.5 * (lo2 - 1) ** 2 + 0.5 * (hi2 - 1) ** 2; e_sd4 = 0.5 * (lo2 ** 2 + hi2 ** 2)
    theory = var_sd2 * (2 * 0.99 - 1) / (e_sd4 * 9.0 - 1.0)
    got = np.corrcoef(sq[:, :-1].ravel(), sq[:, 1:].ravel())[0, 1]
    assert abs(got - theory) < 0.015, (got, theory)


def test_wilson_interval_properties():
    lo, hi = wilson(81, 100)
    assert lo < 0.81 < hi and 0 <= lo and hi <= 1
    lo0, hi0 = wilson(0, 50); assert lo0 == 0.0 or lo0 < 1e-9
    lo1, hi1 = wilson(50, 50); assert hi1 > 1 - 1e-9
    assert abs(expected_coverage_if_truth_in_box() - 0.81) < 0.001


def test_box_bounds_monotone_surface_picks_corners():
    # P increasing in sigma, decreasing in spread, independent of rho -> inf at (sig_lo, s_hi), sup at (sig_hi, s_lo)
    sg, rh, sp = np.meshgrid([0.3, 0.6, 0.9], [0.9, 0.95], [0.01, 0.05, 0.09], indexing="ij")
    P = 100 + 20 * sg - 100 * sp
    A = _surface(P); B = _surface(P)
    b = box_bounds(A, B, (0.4, 0.8), (0.9, 0.95), (0.02, 0.07))
    assert abs(b["low"] - (100 + 20 * 0.4 - 100 * 0.07)) < 1e-9 and abs(b["high"] - (100 + 20 * 0.8 - 100 * 0.02)) < 1e-9
    assert b["argmin"][0] == 0.4 and abs(b["argmin"][2] - 0.07) < 1e-12


def test_two_stage_bounds_remove_winners_curse():
    """Pure noise surface: true P is flat = 100. One-stage inf is biased low; the two-stage re-evaluation is unbiased."""
    rng = np.random.default_rng(0); one, two = [], []
    for _ in range(300):
        A = _surface(100 + rng.normal(0, 1.0, (3, 2, 3))); B = _surface(100 + rng.normal(0, 1.0, (3, 2, 3)))
        b = box_bounds(A, B, (0.3, 0.9), (0.9, 0.95), (0.01, 0.09))
        one.append(b["low_one_stage"]); two.append(b["low"])
    se2 = np.std(two, ddof=1) / np.sqrt(len(two))
    assert np.mean(one) < 99.0                       # clear downward selection bias
    assert abs(np.mean(two) - 100.0) < 4 * se2       # two-stage: unbiased


def test_rescale_bond_preserves_price():
    from src.contracts import Convertible, SoftCall
    from src.credal.sweep import rescale_bond, price_one, default_bond
    from src.pricer.gbm import simulate_gbm
    from src.pricer.lsm import price_lsm
    bond = default_bond()
    spot = 250.0
    paths100 = simulate_gbm(100.0, 0.03, 0.01, 0.4, 3.0, 156, 5000, seed=3, antithetic=False)
    # a stock quoted at 250: same relative paths, scaled by 2.5; the ORIGINAL bond is written in 100-unit terms,
    # so express the 250-quoted bond (CR_q = CR/2.5-style terms) via rescale_bond and compare
    quoted = Convertible(conversion_ratio=bond.conversion_ratio / 2.5, maturity=bond.maturity, coupon=bond.coupon,
                         coupon_freq=bond.coupon_freq, soft_call=SoftCall(bond.soft_call.price, bond.soft_call.trigger * 2.5,
                         bond.soft_call.n_required, bond.soft_call.window, bond.soft_call.protection_years), puts=dict(bond.puts))
    p_ref = price_lsm(paths100, bond, 0.03, 0.03)["price"]
    p_resc = price_lsm(paths100, rescale_bond(quoted, spot), 0.03, 0.03)["price"]
    assert abs(p_ref - p_resc) < 1e-9, (p_ref, p_resc)


def test_quality_report_flags_noisy_nodes():
    from src.credal.sweep import quality_report
    P = np.full((3, 2, 3), 100.0); SE = np.full_like(P, 0.1); SE[2, 1, 0] = 5.0
    S = Surface(np.array([0.3, 0.6, 0.9]), np.array([0.9, 0.95]), np.array([0.01, 0.05, 0.09]), P, SE)
    q = quality_report(S)
    assert q["n_bad"] == 1 and q["bad_nodes"][0] == (0.9, 0.95, 0.01) and q["max_se"] == 5.0
