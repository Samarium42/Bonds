"""Phase 5c step 1: price surfaces on one independent path bank (run twice: --bank A and --bank B, in parallel).

Nodes: sigma x s x beta x recovery at fixed persistence rho. For every (sigma, s, beta, rec) lam0 is calibrated so the
straight bond matches the flat spread s. Also stores the flat-spread price on the same (sigma, s) nodes.
"""
import argparse, json, itertools, time
import numpy as np

from src.credal.sources import family_source, cap_bank
from src.credal.sweep import default_bond, S0, R as RF, Q, T
from src.credit.jump import calibrate_lam0, price_jd
from src.flow.sample import paths_from_weekly
from src.pricer.lsm import price_lsm

ap = argparse.ArgumentParser()
ap.add_argument("--bank", required=True, choices=["A", "B"])
ap.add_argument("--n", type=int, default=20000)
ap.add_argument("--rho", type=float, default=0.95)
a = ap.parse_args()
SIG = [0.30, 0.45, 0.60]; SPR = [0.02, 0.035, 0.05]; BETA = [0.0, 1.0, 2.0]; REC = [0.0, 0.2, 0.4]
bond = default_bond()
Z = cap_bank(family_source(156)(a.rho, a.n, 201 if a.bank == "A" else 202), 10.0)
Zc = Z[:10000]
flat = np.zeros((3, 3)); jd = np.zeros((3, 3, 3, 3)); se = np.zeros((3, 3, 3, 3)); lam = np.zeros((3, 3, 3, 3))
t0 = time.time()
for i, sg in enumerate(SIG):
    wv = sg / np.sqrt(52.0)
    paths = paths_from_weekly(Z, S0, RF, Q, T, wv)
    for j, s in enumerate(SPR):
        flat[i, j] = price_lsm(paths, bond, RF, s)["price"]
        for k, b in enumerate(BETA):
            for m, rec in enumerate(REC):
                l0 = calibrate_lam0(Zc, S0, RF, Q, T, wv, b, rec, s)
                r = price_jd(Z, S0, RF, Q, T, wv, bond, l0, b, rec)
                jd[i, j, k, m], se[i, j, k, m], lam[i, j, k, m] = r["price"], r["se"], l0
        print(f"bank {a.bank} sigma={sg} s={s} done ({time.time()-t0:.0f}s)", flush=True)
json.dump(dict(sig=SIG, spr=SPR, beta=BETA, rec=REC, flat=flat.tolist(), jd=jd.tolist(), se=se.tolist(), lam=lam.tolist(),
               n=a.n, rho=a.rho), open(f"results/credit_credal_{a.bank}.json", "w"))
