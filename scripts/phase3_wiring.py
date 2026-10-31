"""Phase 3: price the same convertible with three path sources and compare.

    GBM    : constant-vol log-normal (Phase 0 ground truth for the pricer)
    FAMILY : the TRUE GJR-GARCH-t family the flow was trained on (same post-processing as the flow)
    FLOW   : rho-conditioned flow matching sampler

The pipeline check is FLOW vs FAMILY (same law if the generator is right): z = (P_flow - P_family)/sqrt(se1^2+se2^2).
GBM is context: the gap GBM vs FAMILY is the price of ignoring clustering/fat tails; the spread of FAMILY over rho at
fixed sigma is a first look at the credal interval.
"""
import argparse, json
import numpy as np
from src.contracts import Convertible, SoftCall
from src.data.family import simulate_family
from src.flow.train import load
from src.flow.sample import simulate_flow, paths_from_weekly
from src.pricer.gbm import simulate_gbm
from src.pricer.lsm import price_lsm

ap = argparse.ArgumentParser(); ap.add_argument("--model", default="results/family_model.pt")
ap.add_argument("--paths", type=int, default=30_000)
ap.add_argument("--rhos", type=float, nargs="+", default=[0.90, 0.95, 0.985])
ap.add_argument("--sigmas", type=float, nargs="+", default=[0.35, 0.50, 0.70]); a = ap.parse_args()
import os
m = load(a.model); N = m.N
S0, R, Q, SPREAD, T = 100.0, 0.03, 0.01, 0.03, 3.0
bond = Convertible(conversion_ratio=0.8, maturity=T, coupon=0.04, coupon_freq=2,
                   soft_call=SoftCall(price=104.0, trigger=170.0, n_required=1, window=1, protection_years=1.0),
                   puts={1.0: 100.0, 2.0: 100.0})
from src.flow.sample import sample_standardised
from src.data.family import cond_from_rho
rows = []
# sigma enters analytically, so draw unit-variance Z once per rho and reuse it across sigma (cells are therefore correlated)
Zfam, Zflw = {}, {}
for rho in a.rhos:
    Zfam[rho] = simulate_family(np.full(a.paths, rho), N, seed=11)
    cache = f"results/zflow_rho{rho}_n{a.paths}.npy"          # flow sampling is the slow step: cache it
    if os.path.exists(cache):
        Zflw[rho] = np.load(cache)
    else:
        Zflw[rho] = sample_standardised(m, a.paths, ode_steps=25, seed=3, cond=cond_from_rho(np.full(a.paths, rho)))
        np.save(cache, Zflw[rho])
    print("sampled rho", rho, flush=True)
print(f"{'sig':>5} {'rho':>6} | {'GBM':>16} {'FAMILY':>16} {'FLOW':>16} | z(flow-family)")
for sig in a.sigmas:
    wv = sig / np.sqrt(52.0)
    g = price_lsm(simulate_gbm(S0, R, Q, sig, T, N, a.paths, seed=1, antithetic=False), bond, R, SPREAD)
    for rho in a.rhos:
        fam = price_lsm(paths_from_weekly(Zfam[rho], S0, R, Q, T, wv), bond, R, SPREAD)
        flw = price_lsm(paths_from_weekly(Zflw[rho], S0, R, Q, T, wv), bond, R, SPREAD)
        z = (flw["price"] - fam["price"]) / np.hypot(flw["se"], fam["se"])
        rows.append(dict(sigma=sig, rho=rho, gbm=g["price"], gbm_se=g["se"], family=fam["price"], family_se=fam["se"],
                         flow=flw["price"], flow_se=flw["se"], z=float(z), called=flw["frac_called"], put=flw["frac_put"]))
        print(f"{sig:5.2f} {rho:6.3f} | {g['price']:7.3f}±{g['se']:.3f}  {fam['price']:7.3f}±{fam['se']:.3f}  {flw['price']:7.3f}±{flw['se']:.3f} | {z:+.2f}", flush=True)
json.dump(rows, open("results/phase3_prices.json", "w"), indent=1)
zs = np.array([r["z"] for r in rows])
print(f"\nmax |z| = {np.abs(zs).max():.2f}  (criterion: |z| < 3 in all {len(zs)} cells; mean z = {zs.mean():+.2f})")
