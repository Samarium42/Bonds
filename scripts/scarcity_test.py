"""Separate the effects of path length N and training-set size. Usage: scarcity_test.py N abundant|scarce [steps]"""
import sys, numpy as np, torch
torch.set_num_threads(1)
from src.data.synthetic import simulate_gjr, synthetic_basket
from src.data.windows import standardise, build_train_set, sliding_windows
from src.flow.train import train
from src.flow.sample import sample_standardised
from src.validation import stylized_facts as sf
N = int(sys.argv[1]); mode = sys.argv[2]; steps = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
if mode == "abundant":
    X = sliding_windows(simulate_gjr(1_200_000, 0.075, seed=5) / 0.075, N, stride=N).astype(np.float32)
else:
    z, _ = standardise(synthetic_basket()); X = build_train_set(z, N, stride=4)
print(f"N={N} {mode}: train windows {X.shape}, std {X.std():.3f}", flush=True)
m, hist = train(X, steps=steps, lr=2e-3, log_every=500, ch=64, monitor=False)
ref = sliding_windows(simulate_gjr(300_000, 0.075, seed=77) / 0.075, N, stride=N)
g = sample_standardised(m, 2000, ode_steps=25, seed=1)
iqr = lambda a: np.subtract(*np.percentile(a, [75, 25]))
print(f"RESULT N={N} {mode}: std ref/gen/train {ref.std():.3f}/{g.std():.3f}/{X.std():.3f}  IQR ref/gen/train {iqr(ref):.2f}/{iqr(g):.2f}/{iqr(X):.2f}")
print("   ACF sq ref", np.round(sf.pooled_acf(ref, 4, 2)[1:], 3), "gen", np.round(sf.pooled_acf(g, 4, 2)[1:], 3), "train", np.round(sf.pooled_acf(X, 4, 2)[1:], 3))
print("   energy (E,p)", sf.energy_test(ref[:600], g[:600], n_perm=99), " memorisation", round(sf.memorisation_ratio(g, X, ref[:600]), 3))
