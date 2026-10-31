"""Robust cheap/rich verdict for ONE convertible at ONE date.

    python -m scripts.run_sweep --spot 250 --market-price 112.4 --sigma 0.45 0.80 --rho 0.93 0.985 --spread 0.02 0.06 \
        --model results/family_model_feat.pt [--bond configs/tesla_2024.yaml]

Prints [P_low, P_high] over the credal box and:  market < P_low  -> robustly CHEAP;  market > P_high -> robustly RICH;
otherwise INCONCLUSIVE (the market price is consistent with some plausible model).
Limit: the learned generator has a fixed path length (N=156 weekly steps = 3.0 years), so the bond must have 3.0 years left.
"""
import argparse, json
import numpy as np
from src.contracts import Convertible
from src.credal.sources import flow_source, family_source, with_cap
from src.credal.sweep import build_surface, box_bounds, default_bond, rescale_bond, interp_error, quality_report

ap = argparse.ArgumentParser()
ap.add_argument("--spot", type=float, required=True); ap.add_argument("--market-price", type=float, required=True)
ap.add_argument("--sigma", type=float, nargs=2, required=True, help="annual vol range"); ap.add_argument("--rho", type=float, nargs=2, default=[0.93, 0.985])
ap.add_argument("--spread", type=float, nargs=2, required=True); ap.add_argument("--bond", default=None)
ap.add_argument("--source", choices=["flow", "family"], default="flow"); ap.add_argument("--model", default="results/family_model_feat.pt")
ap.add_argument("--paths", type=int, default=20000); ap.add_argument("--cap", type=float, default=10.0); ap.add_argument("--out", default="results/sweep.json")
a = ap.parse_args()
bond = Convertible.from_yaml(a.bond) if a.bond else default_bond()
assert abs(bond.maturity - 3.0) < 1e-9, "generator path length is 3.0 years (156 weekly steps)"
bond = rescale_bond(bond, a.spot)
if a.source == "flow":
    from src.flow.train import load
    base = flow_source(load(a.model))
else:
    base = family_source()
src = with_cap(base, a.cap)
sig = np.linspace(a.sigma[0], a.sigma[1], 4); spr = np.linspace(a.spread[0], a.spread[1], 3)
rho = np.unique(np.round(np.linspace(a.rho[0], a.rho[1], 4), 4))
A = build_surface(src, bond, sig, rho, spr, a.paths, seed=1000, verbose=True); B = build_surface(src, bond, sig, rho, spr, a.paths, seed=2000)
b = box_bounds(A, B, tuple(a.sigma), tuple(a.rho), tuple(a.spread))
err = interp_error(A, src, bond, a.paths, 1000, n_test=4)
q = quality_report(A)
if q['n_bad']:
    print(f"WARNING: {q['n_bad']}/{q['n_nodes']} grid nodes have MC SE > 1 (max {q['max_se']:.1f}); bounds touching them are unreliable. Examples (sigma, rho, s): {q['bad_nodes'][:3]}")
m = a.market_price
verdict = "robustly CHEAP (market below the whole credal set)" if m < b["low"] else "robustly RICH (market above the whole credal set)" if m > b["high"] else "INCONCLUSIVE (market inside the credal interval)"
print(f"\ncredal interval  [{b['low']:.2f}, {b['high']:.2f}]   (one-stage {b['low_one_stage']:.2f}, {b['high_one_stage']:.2f}; mean MC SE {A.SE.mean():.2f})")
print(f"argmin (sigma, rho, s) = {b['argmin']}   argmax = {b['argmax']}")
print(f"interpolation error vs direct pricing: max {err[0]:.3f}, mean {err[1]:.3f}")
print(f"market {m:.2f} -> {verdict}")
json.dump({"low": b["low"], "high": b["high"], "market": m, "verdict": verdict, "bounds": b, "interp_err": err}, open(a.out, "w"), indent=1, default=float)
