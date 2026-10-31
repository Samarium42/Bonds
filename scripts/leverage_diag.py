"""Leverage diagnosis at N=52 (fast proxy).   python -m scripts.leverage_diag baseline|feat|wide|feat_wide"""
import sys, numpy as np, torch
torch.set_num_threads(1)
from src.data.family import training_bank, simulate_family, cond_from_rho
from src.flow.train import train
from src.flow.sample import sample_standardised
from src.validation import stylized_facts as sf
variant = sys.argv[1]; N = 52
feat = "feat" in variant; ch = 128 if "wide" in variant else 64
X, C = training_bank(30_000, N, seed=0)
m, hist = train(X, steps=2000, lr=2e-3, log_every=1000, C=C, monitor=False, ch=ch, feat=feat)
print(f"[{variant}] final loss {np.mean(hist[-200:]):.4f}", flush=True)
for rho in (0.93, 0.985):
    ref = simulate_family(np.full(20_000, rho), N, seed=int(rho * 1000))
    g = sample_standardised(m, 2500, ode_steps=25, seed=1, cond=cond_from_rho(np.full(2500, rho)))
    (lr_, ser), (lg, seg) = sf.leverage_se(ref), sf.leverage_se(g)
    z = (lg - lr_) / np.hypot(ser, seg)
    print(f"[{variant}] rho={rho}: leverage lag1 ref {lr_:+.4f}±{ser:.4f}  gen {lg:+.4f}±{seg:.4f}  z={z:+.1f} | "
          f"ACF sq1 ref {sf.pooled_acf(ref,1,2)[1]:.3f} gen {sf.pooled_acf(g,1,2)[1]:.3f} | std gen {g.std():.3f}", flush=True)
