"""Controlled backtest of the credal bounds.

    python -m scripts.controlled_backtest --source family --reps 150
    python -m scripts.controlled_backtest --source flow --model results/family_model_feat.pt --reps 150   (slow: flow sampling)
"""
import argparse, json, time
import numpy as np
from src.credal.sources import family_source, flow_source, with_cap
from src.credal.sweep import build_surface, default_bond, interp_error, quality_report
from src.credal.backtest import regime_switch_Z, gbm_surface, run_scenario, expected_coverage_if_truth_in_box, wilson

ap = argparse.ArgumentParser()
ap.add_argument("--source", choices=["family", "flow"], default="family"); ap.add_argument("--model", default="results/family_model_feat.pt")
ap.add_argument("--reps", type=int, default=150); ap.add_argument("--paths", type=int, default=30000)
ap.add_argument("--cap", type=float, default=10.0); ap.add_argument("--out", default="results/controlled_backtest.json")
a = ap.parse_args()
if a.source == "flow":
    from src.flow.train import load
    base = flow_source(load(a.model))
else:
    base = family_source()
src = with_cap(base, a.cap)
bond = default_bond()
SIG = [0.20, 0.30, 0.40, 0.50, 0.60, 0.75, 0.90, 1.10]; RHO = [0.90, 0.93, 0.96, 0.985]; SPR = [0.005, 0.02, 0.04, 0.07, 0.12]   # sigma stays where MC noise is small
t0 = time.time()
A = build_surface(src, bond, SIG, RHO, SPR, a.paths, seed=1000); B = build_surface(src, bond, SIG, RHO, SPR, a.paths, seed=2000)
print(f"surfaces built in {time.time()-t0:.0f}s; mean SE {A.SE.mean():.3f}, max SE {A.SE.max():.3f}", flush=True)
print('surface quality:', quality_report(A), flush=True)
# structure checks on the surface
dsig = np.diff(A.P, axis=0); dsp = np.diff(A.P, axis=2)
print(f"monotone: P increasing in sigma in {np.mean(dsig > 0):.1%} of grid steps, decreasing in s in {np.mean(dsp < 0):.1%}", flush=True)
print("interp error (max, mean) vs direct pricing:", interp_error(A, src, bond, a.paths, 1000, n_test=10), flush=True)
G = gbm_surface(np.random.default_rng(4000).standard_normal((a.paths, 156)), bond, SIG, SPR)
exp_cov = expected_coverage_if_truth_in_box()
print(f"nominal: P(sigma*, s* both inside boxes) = {exp_cov:.3f}\n", flush=True)
rows = []
print(f"{'regime':>8} {'sig*':>5} {'s*':>5} | {'P*':>8} | {'cover':>6} {'Wilson95':>14} | {'width':>6} | GBM: {'bias':>6} {'|err|':>6} {'P(|err|>1)':>10} {'inside':>6} | clip")
for name, ratio, stay in (("mild", 2.0, 0.97), ("stress", 3.0, 0.99)):
    for sg in (0.3, 0.45, 0.6):
        for sp in (0.02, 0.05):
            Zr = np.clip(regime_switch_Z(60000, 156, 3000 + int(sg * 100) + int(sp * 1000), ratio, stay), -a.cap, a.cap); Zr = Zr / Zr.std()
            r = run_scenario(A, B, G, bond, Zr, sg, sp, a.reps, seed=5000 + len(rows)); r.update(regime=name, sigma=sg, s=sp)
            rows.append(r)
            print(f"{name:>8} {sg:5.2f} {sp:5.2f} | {r['p_star']:8.3f} | {r['coverage']:6.3f} [{r['wilson95'][0]:.3f},{r['wilson95'][1]:.3f}] | {r['mean_width']:6.2f} |      {r['gbm_mean_err']:+6.2f} {r['gbm_mean_abs_err']:6.2f} {r['gbm_frac_err_gt_1']:10.2f} {r['gbm_frac_inside_interval']:6.2f} | {r['boxes_clipped_by_grid']}", flush=True)
for name in ("mild", "stress"):
    rr = [r for r in rows if r["regime"] == name]; k = sum(r["covered"] for r in rr); n = sum(r["n"] for r in rr)
    print(f"\n{name}: pooled coverage {k}/{n} = {k/n:.3f}, Wilson95 {tuple(round(x,3) for x in wilson(k, n))}; nominal lower reference {exp_cov:.3f}")
json.dump({"rows": rows, "nominal": exp_cov, "cap": a.cap, "source": a.source}, open(a.out, "w"), indent=1, default=float)
