"""Phase 5b: flat-spread (Phase 4) vs jump-to-default credit model, matched on the straight bond.

For each (sigma, s, recovery, beta): calibrate lam0 so the risky ZCB equals face*exp(-(r+s)T), price the convertible under
both models on the SAME path bank (paired difference, with a paired SE), and invert the flat model to get the
equivalent flat spread s_eq of the jump price. basis = s_eq - s is the credit-model basis a flat-spread desk would see.
"""
import argparse, json, time
import numpy as np
from scipy.interpolate import interp1d

from src.credal.sources import family_source, cap_bank
from src.credal.sweep import default_bond, S0, R as RF, Q, T
from src.credit.jump import calibrate_lam0, price_jd
from src.flow.sample import paths_from_weekly
from src.pricer.lsm import price_lsm

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=30000)
ap.add_argument("--rho", type=float, default=0.95)
ap.add_argument("--out", default="results/credit_gap.json")
a = ap.parse_args()

SIG = [0.30, 0.60]; SPR = [0.02, 0.05, 0.08]; REC = [0.0, 0.4]; BETA = [0.0, 1.0, 2.0]
bond = default_bond()
Z = cap_bank(family_source(156)(a.rho, a.n, 101), 10.0)
grid = np.array([0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.13, 0.17])
rows = []; t0 = time.time()
for sg in SIG:
    wv = sg / np.sqrt(52.0)
    paths = paths_from_weekly(Z, S0, RF, Q, T, wv)
    flat_curve = np.array([price_lsm(paths, bond, RF, s)["price"] for s in grid])
    inv = interp1d(flat_curve[::-1], grid[::-1], kind="cubic", bounds_error=False, fill_value=np.nan)
    for s in SPR:
        flat = price_lsm(paths, bond, RF, s, keep_pv=True)
        for rec in REC:
            for beta in BETA:
                l0 = calibrate_lam0(Z, S0, RF, Q, T, wv, beta, rec, s)
                jd = price_jd(Z, S0, RF, Q, T, wv, bond, l0, beta, rec, keep_pv=True)
                d = jd["pv"] - flat["pv"]
                rows.append(dict(sigma=sg, s=s, rec=rec, beta=beta, lam0=l0, flat=flat["price"], jd=jd["price"],
                                 diff=float(d.mean()), diff_se=float(d.std(ddof=1) / np.sqrt(len(d))),
                                 s_eq=float(inv(jd["price"])), comp_err=jd["comp_err"],
                                 frac_put_flat=flat["frac_put"], frac_put_jd=jd["frac_put"],
                                 frac_conv_flat=flat["frac_converted_early"], frac_conv_jd=jd["frac_converted_early"]))
                r = rows[-1]
                print(f"sig={sg:.2f} s={s:.2f} R={rec:.1f} b={beta:.0f} lam0={l0:.3f} flat={r['flat']:.2f} jd={r['jd']:.2f} "
                      f"diff={r['diff']:+.2f}+-{r['diff_se']:.2f} s_eq={r['s_eq']:.4f} ({time.time()-t0:.0f}s)", flush=True)
json.dump(dict(n=a.n, rho=a.rho, rows=rows, grid=grid.tolist()), open(a.out, "w"), indent=1)
