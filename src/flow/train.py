"""Train the unconditional path generator.

    python -m src.flow.train --source synthetic --steps 3000 --out results/synthetic_model.pt
    python -m src.flow.train --source real --csv data/raw/weekly_returns.csv --steps 8000 --out results/tsla_model.pt
"""
from __future__ import annotations

import argparse
import copy
import json
import time

import numpy as np
import torch

from src.data.windows import build_train_set, standardise
from src.flow.cfm_model import VelocityNet, cfm_loss


def train(X: np.ndarray, steps: int = 3000, batch: int = 128, lr: float = 1e-3, ema_decay: float = 0.998,
          ch: int = 64, seed: int = 0, log_every: int = 250, device: str = "cpu", monitor: bool = True,
          norm: str = "group", glob: bool = False, param: str = "v"):
    torch.manual_seed(seed)
    N = X.shape[1]
    model = VelocityNet(N, ch=ch, norm=norm, glob=glob, param=param).to(device)
    ema = copy.deepcopy(model).eval()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    data = torch.from_numpy(X).to(device)
    hist, t0 = [], time.time()
    for it in range(1, steps + 1):
        idx = torch.randint(0, data.shape[0], (batch,), device=device)
        loss = cfm_loss(model, data[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        d = min(ema_decay, (1 + it) / (10 + it))          # warm-up: early steps are not dominated by the init
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(d).add_(pm.detach(), alpha=1 - d)
        hist.append(loss.item())
        if it % log_every == 0:
            msg = f"step {it:6d}  loss {np.mean(hist[-log_every:]):.4f}  ({time.time()-t0:.0f}s)"
            if monitor:       # cheap health check: generated std should approach the data std (~1)
                from src.flow.sample import sample_standardised
                g = sample_standardised(ema, 256, ode_steps=10, seed=0, method="heun")
                msg += f"  gen_std {g.std():.3f} (data {X.std():.3f})"
            print(msg, flush=True)
    return ema, hist


def save(model: VelocityNet, path: str, meta: dict) -> None:
    torch.save({"state": model.state_dict(), "N": model.N, "meta": meta}, path)


def load(path: str, device: str = "cpu") -> VelocityNet:
    ck = torch.load(path, map_location=device)
    m = VelocityNet(ck["N"]).to(device)
    m.load_state_dict(ck["state"])
    return m.eval()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["synthetic", "real"], default="synthetic")
    ap.add_argument("--csv", default="data/raw/weekly_returns.csv")
    ap.add_argument("--N", type=int, default=156)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--boot", type=int, default=0, help="stationary-bootstrap windows per name")
    ap.add_argument("--out", default="results/model.pt")
    a = ap.parse_args()
    if a.source == "synthetic":
        from src.data.synthetic import synthetic_basket
        series = synthetic_basket()
    else:
        from src.data.download import load as load_csv
        series = load_csv(a.csv)
    z, sig = standardise(series)
    X = build_train_set(z, a.N, n_boot_per_name=a.boot)
    print("train windows:", X.shape)
    m, hist = train(X, steps=a.steps)
    save(m, a.out, {"sigma_train": sig, "args": vars(a)})
    json.dump({"loss": hist}, open(a.out.replace(".pt", "_loss.json"), "w"))
