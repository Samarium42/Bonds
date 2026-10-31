"""Velocity network v_theta(x_t, t [, cond]) for whole-path conditional flow matching.

A path is x in R^N (standardised weekly log-returns). The net is a stack of dilated 1D conv residual blocks
with FiLM conditioning on time (and on an optional condition vector, reserved for Phase 2's theta).
Dilations 1..64 give a receptive field > 250 steps, so every output position can see the whole path:
that is what lets a non-autoregressive model learn volatility clustering across the horizon.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10_000.0) * torch.arange(half, device=t.device) / half)
    a = 1000.0 * t[:, None] * freqs[None]
    return torch.cat([a.sin(), a.cos()], dim=-1)


class ChanNorm(nn.Module):
    """LayerNorm over channels at each position (keeps the amplitude of the path across time)."""
    def __init__(self, ch):
        super().__init__()
        self.ln = nn.LayerNorm(ch)

    def forward(self, x):
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


def make_norm(kind: str, ch: int) -> nn.Module:
    if kind == "group":      # normalises each window over (channels, time): removes window-level amplitude
        return nn.GroupNorm(8, ch)
    if kind == "chan":
        return ChanNorm(ch)
    return nn.Identity()


class Block(nn.Module):
    def __init__(self, ch: int, dil: int, emb: int, norm: str = "group", glob: bool = False):
        super().__init__()
        self.glob = nn.Linear(2 * ch, ch) if glob else None   # window-level summary (mean, rms over time)
        self.n = make_norm(norm, ch)
        self.c1 = nn.Conv1d(ch, ch, 3, padding=dil, dilation=dil)
        self.film = nn.Linear(emb, 2 * ch)
        self.c2 = nn.Conv1d(ch, ch, 1)
        nn.init.zeros_(self.c2.weight)
        nn.init.zeros_(self.c2.bias)

    def forward(self, x, e):
        g, b = self.film(e).chunk(2, dim=-1)
        h = self.n(x) * (1 + g[..., None]) + b[..., None]
        h = self.c1(F.silu(h))
        if self.glob is not None:
            h = h + self.glob(torch.cat([h.mean(-1), h.pow(2).mean(-1).add(1e-6).sqrt()], dim=-1))[..., None]
        return x + self.c2(F.silu(h))


class VelocityNet(nn.Module):
    def __init__(self, N: int, ch: int = 64, emb: int = 128, cond_dim: int = 0,
                 dilations=(1, 2, 4, 8, 16, 32, 64, 1, 4, 16), norm: str = "group",
                 glob: bool = False, param: str = "v", feat: bool = False):
        super().__init__()
        self.N, self.cond_dim, self.emb, self.param, self.feat = N, cond_dim, emb, param, feat
        self.t_mlp = nn.Sequential(nn.Linear(emb, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.c_mlp = nn.Sequential(nn.Linear(cond_dim, emb), nn.SiLU(), nn.Linear(emb, emb)) if cond_dim else None
        self.inp = nn.Conv1d(4 if feat else 2, ch, 1)         # channels: x_t, position k/N (+ x^2, tanh(x) if feat)
        self.blocks = nn.ModuleList(Block(ch, d, emb, norm, glob) for d in dilations)
        self.out = nn.Conv1d(ch, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.register_buffer("pos", torch.linspace(0, 1, N)[None, None, :])

    def forward(self, x, t, cond=None):
        out = self.raw(x, t, cond)
        if self.param == "x":            # network predicts the clean path; v = (xhat - x_t) / (1 - t)
            return (out - x) / (1 - t).clamp(min=0.02)[:, None]
        return out

    def raw(self, x, t, cond=None):
        e = self.t_mlp(sinusoidal(t, self.emb))
        if self.c_mlp is not None:
            e = e + self.c_mlp(cond)
        chans = [x[:, None, :], self.pos.expand(x.shape[0], -1, -1)]
        if self.feat:            # even/odd features make the sign asymmetry (leverage) cheap to represent
            chans += [x[:, None, :] ** 2, torch.tanh(x)[:, None, :]]
        h = self.inp(torch.cat(chans, dim=1))
        for b in self.blocks:
            h = b(h, e)
        return self.out(h)[:, 0, :]


def cfm_loss(model: VelocityNet, x1: torch.Tensor, cond=None) -> torch.Tensor:
    """Conditional flow matching with the straight (optimal-transport) path:
        x0 ~ N(0,I), x_t = (1-t) x0 + t x1, target velocity u = x1 - x0,
        L = E_{t~U[0,1], x0, x1} || v_theta(x_t, t) - (x1 - x0) ||^2 / N.
    """
    x0 = torch.randn_like(x1)
    t = torch.rand(x1.shape[0], device=x1.device)
    xt = (1 - t)[:, None] * x0 + t[:, None] * x1
    if model.param == "x":               # same minimiser (posterior mean E[x1|x_t]), better conditioned in high dim
        return ((model.raw(xt, t, cond) - x1) ** 2).mean()
    v = model(xt, t, cond)
    return ((v - (x1 - x0)) ** 2).mean()
