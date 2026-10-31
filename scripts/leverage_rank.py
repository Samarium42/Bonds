"""Tail-robust leverage: Spearman correlation of r_t with r_{t+1}^2 (equivalently with |r_{t+1}|), on cached samples.
Bootstrap over windows (ranks fixed globally: a slight approximation). Compares baseline vs feature model vs truth."""
import numpy as np
from scipy.stats import rankdata
from src.data.family import simulate_family

def rank_leverage(W, lag=1, n_boot=300, seed=0):
    a = W[:, :-lag]; b = np.abs(W[:, lag:])
    ra = rankdata(a.ravel()).reshape(a.shape); rb = rankdata(b.ravel()).reshape(b.shape)
    ra = (ra - ra.mean()) / ra.std(); rb = (rb - rb.mean()) / rb.std()
    per = (ra * rb).mean(1)                       # per-window mean of rank products
    rng = np.random.default_rng(seed)
    boots = [per[rng.integers(0, len(per), len(per))].mean() for _ in range(n_boot)]
    return float(per.mean()), float(np.std(boots, ddof=1))

N = 156; rows = {"base": [], "feat": []}
print(f"{'rho':>6} | {'truth':>16} | {'baseline':>16} {'z':>5} | {'features':>16} {'z':>5}")
for rho in (0.93, 0.97, 0.985):
    ref = simulate_family(np.full(20_000, rho), N, seed=int(rho * 1000)); tr, ts = rank_leverage(ref)
    out = []
    for key, name in (("base", "family_model.pt"), ("feat", "family_model_feat.pt")):
        g = np.load(f"results/lev_gen_{name}_{rho}_3000.npy"); gl, gs = rank_leverage(g)
        d, s = gl - tr, np.hypot(gs, ts); rows[key].append((d, s)); out.append((gl, gs, d / s))
    print(f"{rho:6.3f} | {tr:+.4f}±{ts:.4f} | {out[0][0]:+.4f}±{out[0][1]:.4f} {out[0][2]:+5.1f} | {out[1][0]:+.4f}±{out[1][1]:.4f} {out[1][2]:+5.1f}")
for key in rows:
    d = np.array([r[0] for r in rows[key]]); s = np.array([r[1] for r in rows[key]]); w = 1 / s ** 2
    pm, pse = (w * d).sum() / w.sum(), 1 / np.sqrt(w.sum())
    print(f"pooled deficit {key}: {pm:+.4f} ± {pse:.4f}  z={pm/pse:+.1f}")
