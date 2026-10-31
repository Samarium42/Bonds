"""End-to-end Phase 1 check on a market with KNOWN stylized facts.

    python -m scripts.validate_synthetic --steps 3000 --boot 0 --tag plain
"""
import argparse, json
import numpy as np
from src.data.synthetic import synthetic_basket, simulate_gjr
from src.data.windows import standardise, build_train_set, sliding_windows
from src.flow.train import train, save
from src.flow.sample import sample_standardised
from src.validation import stylized_facts as sf

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--boot", type=int, default=0)
ap.add_argument("--N", type=int, default=156)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--tag", default="plain")
a = ap.parse_args()

series = synthetic_basket()
z, sig = standardise(series)
X = build_train_set(z, a.N, n_boot_per_name=a.boot)
print("train windows", X.shape, flush=True)

model, hist = train(X, steps=a.steps, lr=a.lr, log_every=500)
save(model, f"results/synth_{a.tag}.pt", {"sigma_train": sig})

# reference: independent long sample from the same DGP, standardised by the TRUE unconditional vol
ref_series = simulate_gjr(200_000, 0.075, seed=999)
ref = sliding_windows(ref_series / 0.075, a.N, stride=a.N)       # non-overlapping windows
gen = sample_standardised(model, 2000, ode_steps=25, seed=1)
gen10 = sample_standardised(model, 2000, ode_steps=10, seed=1)

rep = sf.report(ref[:2000], gen, train=X)
rep["ode_convergence"] = {"ks_10_vs_25_steps": sf.ks_two_sample(gen10, gen)}
rep["acf_sq_train"] = sf.pooled_acf(X, 10, 2).tolist()
rep["train_moments"] = sf.moments(X)
rep["final_loss"] = float(np.mean(hist[-200:]))
json.dump(rep, open(f"results/synth_{a.tag}_report.json", "w"), indent=1, default=float)
np.save(f"results/synth_{a.tag}_gen.npy", gen[:500]); np.save("results/synth_ref.npy", ref[:500]); np.save(f"results/synth_{a.tag}_loss.npy", np.array(hist))

f = lambda v: np.round(v, 3)
print("\n=== REPORT:", a.tag, "===")
print("moments real", rep["real_moments"]); print("moments gen ", rep["gen_moments"]); print("moments train", rep["train_moments"])
print("hill real/gen", rep["hill_real"], rep["hill_gen"])
print("KS 2-sample", rep["ks_two_sample"])
print("ACF sq real ", f(rep["acf_sq_real"][1:6])); print("ACF sq gen  ", f(rep["acf_sq_gen"][1:6])); print("ACF sq train", f(rep["acf_sq_train"][1:6]))
print("LB(sq) real/gen", rep["ljung_box_sq_real"], rep["ljung_box_sq_gen"])
print("leverage real", f(rep["leverage_real"])); print("leverage gen ", f(rep["leverage_gen"]))
print("energy (E, p)", rep["energy"], " memorisation ratio", rep["memorisation_ratio"])
print("ODE 10 vs 25 steps KS", rep["ode_convergence"])
