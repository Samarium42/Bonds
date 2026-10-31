"""Leverage of the trained rho-conditioned flow vs the true family, with bootstrap-over-windows standard errors.

    python -m scripts.leverage_check --model results/family_model.pt --rhos 0.93 0.97 0.985 --n 3000
Generated samples are cached in results/ so reruns are free.
"""
import argparse, os, numpy as np
from src.data.family import simulate_family, cond_from_rho
from src.flow.train import load
from src.flow.sample import sample_standardised
from src.validation import stylized_facts as sf

ap = argparse.ArgumentParser(); ap.add_argument("--model", default="results/family_model.pt")
ap.add_argument("--rhos", type=float, nargs="+", default=[0.93, 0.97, 0.985]); ap.add_argument("--n", type=int, default=3000)
a = ap.parse_args()
m = load(a.model); N = m.N; diffs, ses = [], []
for rho in a.rhos:
    ref = simulate_family(np.full(20_000, rho), N, seed=int(rho * 1000))
    cache = f"results/lev_gen_{os.path.basename(a.model)}_{rho}_{a.n}.npy"
    if os.path.exists(cache): g = np.load(cache)
    else:
        g = sample_standardised(m, a.n, ode_steps=25, seed=1, cond=cond_from_rho(np.full(a.n, rho))); np.save(cache, g)
    for lag in (1, 2):
        (lr_, ser), (lg, seg) = sf.leverage_se(ref, lag), sf.leverage_se(g, lag)
        z = (lg - lr_) / np.hypot(ser, seg)
        print(f"rho={rho} lag{lag}: ref {lr_:+.4f}±{ser:.4f}  gen {lg:+.4f}±{seg:.4f}  diff {lg-lr_:+.4f}  z={z:+.1f}", flush=True)
        if lag == 1: diffs.append(lg - lr_); ses.append(np.hypot(ser, seg))
d, s = np.array(diffs), np.array(ses); w = 1 / s ** 2
pm, pse = (w * d).sum() / w.sum(), 1 / np.sqrt(w.sum())
print(f"\npooled lag-1 leverage difference (gen - ref) over {len(d)} rho cells: {pm:+.4f} ± {pse:.4f}  z={pm/pse:+.1f}")
