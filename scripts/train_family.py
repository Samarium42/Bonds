"""Train the rho-conditioned path generator on the parametric family.   python -m scripts.train_family --steps 5000"""
import argparse, json, numpy as np
from src.data.family import training_bank
from src.flow.train import train, save

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=5000)
ap.add_argument("--N", type=int, default=156)
ap.add_argument("--bank", type=int, default=60_000)
ap.add_argument("--out", default="results/family_model.pt")
ap.add_argument("--feat", action=argparse.BooleanOptionalAction, default=True, help="add x^2 and tanh(x) input channels (default on; --no-feat for the baseline)")
a = ap.parse_args()
X, C = training_bank(a.bank, a.N, rho_range=(0.90, 0.99), seed=0)
print("bank", X.shape, "std", X.std().round(3), flush=True)
m, hist = train(X, steps=a.steps, lr=2e-3, log_every=500, C=C, monitor=False, feat=a.feat)
save(m, a.out, {"rho_range": [0.90, 0.99], "steps": a.steps, "feat": a.feat})
json.dump({"loss": hist}, open(a.out.replace(".pt", "_loss.json"), "w"))
print("saved", a.out)
