"""Steerability test: does the rho-conditioned flow match the TRUE family at rho values it can be probed on?

    python -m scripts.validate_family --model results/family_model.pt
Reference = fresh paths from the same family (independent seed). Held-in = inside the training range [0.90, 0.99];
the out-of-range value is reported separately and is expected to degrade.
"""
import argparse, json
import numpy as np
from src.data.family import simulate_family, cond_from_rho
from src.flow.train import load
from src.flow.sample import sample_standardised
from src.validation import stylized_facts as sf

ap = argparse.ArgumentParser(); ap.add_argument("--model", default="results/family_model.pt")
ap.add_argument("--n", type=int, default=3000); a = ap.parse_args()
m = load(a.model); N = m.N
rows, out = [], {}
for rho, tag in [(0.91, "in"), (0.93, "in"), (0.95, "in"), (0.97, "in"), (0.985, "in"), (0.85, "OUT of range")]:
    ref = simulate_family(np.full(a.n, rho), N, seed=int(rho * 1000) + 7)
    c = cond_from_rho(np.full(a.n, rho))
    g = sample_standardised(m, a.n, ode_steps=25, seed=1, cond=c)
    g10 = sample_standardised(m, 1000, ode_steps=10, seed=1, cond=c[:1000])
    E, p = sf.energy_test(ref[:600], g[:600], n_perm=199)
    r = dict(rho=rho, tag=tag, std_ref=float(ref.std()), std_gen=float(g.std()),
             acf_ref=sf.pooled_acf(ref, 5, 2)[1:].round(3).tolist(), acf_gen=sf.pooled_acf(g, 5, 2)[1:].round(3).tolist(),
             hill_ref=sf.hill_alpha(ref)[0], hill_gen=sf.hill_alpha(g)[0],
             lev_ref=float(sf.leverage(ref, 1)[0]), lev_gen=float(sf.leverage(g, 1)[0]),
             energy_p=p, ode10_vs_25_std=float(g10.std() - g[:1000].std()))
    rows.append(r); print(r, flush=True)
# monotonicity of clustering in rho (generated)
ins = [r for r in rows if r["tag"] == "in"]
mono = all(ins[i]["acf_gen"][0] < ins[i + 1]["acf_gen"][0] for i in range(len(ins) - 1))
print("\nlag-1 ACF of squares, generated, increasing in rho:", mono)
print("Bonferroni threshold for 5 in-range energy tests: p >", 0.05 / 5)
json.dump(rows, open("results/family_validation.json", "w"), indent=1)
