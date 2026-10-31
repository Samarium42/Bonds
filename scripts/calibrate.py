"""Calibrate the path family and the credal ranges from DAILY returns.

    python -m scripts.calibrate --csv data/raw/daily_returns.csv --name TSLA
    python -m scripts.calibrate --synthetic            # offline check on simulated data

Outputs configs/credal_ranges.json with:
  rho_range   5%-95% band of persistence across bootstrap refits of every basket name (model uncertainty for rho)
  sigma_ann   [min, max] of rolling 1y realised annual vol of the target name (data-driven bounds for sigma)
  alpha, gamma, nu  point values used to build the training family (basket medians)
"""
import argparse, json
import numpy as np
from src.calibrate.garch import fit_gjr_t, parametric_bootstrap_ci

ap = argparse.ArgumentParser(); ap.add_argument("--csv", default="data/raw/daily_returns.csv")
ap.add_argument("--name", default="TSLA"); ap.add_argument("--synthetic", action="store_true")
ap.add_argument("--out", default="configs/credal_ranges.json"); ap.add_argument("--boots", type=int, default=30)
a = ap.parse_args()
if a.synthetic:
    from src.data.synthetic import simulate_gjr
    series = {f"SYN{i}": simulate_gjr(3500, v, 10 + i, 0.05, 0.06, 0.88 + 0.01 * (i % 3 - 1), 5.0) for i, v in enumerate((0.03, 0.025, 0.028))}
    target = "SYN0"
else:
    from src.data.download import load
    series = load(a.csv); target = a.name
fits, rho_lo, rho_hi = {}, [], []
for k, r in series.items():
    f = fit_gjr_t(r); ci = parametric_bootstrap_ci(f, n_boot=a.boots)
    fits[k] = dict(alpha=f.alpha, gamma=f.gamma, beta=f.beta, nu=f.nu, rho=f.rho, rho_ci=ci["rho"], n=f.n)
    rho_lo.append(ci["rho"][0]); rho_hi.append(ci["rho"][1])
    print(f"{k:6s} rho={f.rho:.3f} CI[{ci['rho'][0]:.3f},{ci['rho'][1]:.3f}] gamma={f.gamma:.3f} nu={f.nu:.2f} n={f.n}")
x = np.asarray(series[target]); w = 252
roll = np.array([x[i:i + w].std() * np.sqrt(252) for i in range(0, len(x) - w, 21)])
cfg = dict(rho_range=[float(min(rho_lo)), float(min(max(rho_hi), 0.995))],
           sigma_ann=[float(roll.min()), float(roll.max())],
           alpha=float(np.median([f["alpha"] for f in fits.values()])), gamma=float(np.median([f["gamma"] for f in fits.values()])),
           nu=float(np.median([f["nu"] for f in fits.values()])), target=target, per_name=fits)
json.dump(cfg, open(a.out, "w"), indent=1)
print("rho_range", np.round(cfg["rho_range"], 3), "sigma_ann (rolling 1y)", np.round(cfg["sigma_ann"], 3))
