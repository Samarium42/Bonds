"""Convertible bond term-sheet objects.

Conventions
-----------
* Face value is per bond (default 100). conversion_ratio = shares per bond.
* Time grid: n_steps equal steps on [0, T], dt = T / n_steps, step k is time k*dt.
* Coupons are paid on the grid at times j / coupon_freq (j = 1..T*coupon_freq).
* Soft call is an N-of-M trigger on the *step* grid: the issuer calls the first time
  S >= trigger on at least n_required of the last `window` steps (after protection ends).
  The issuer is assumed to call immediately once eligible (barrier simplification,
  not a second optimal-stopping problem). Holder then receives max(call price, parity).
* Accrued interest at call/put/conversion is ignored (conservative, documented).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import yaml


@dataclass(frozen=True)
class SoftCall:
    price: float           # call redemption price per bond
    trigger: float         # absolute stock level
    n_required: int        # N
    window: int            # M (in steps)
    protection_years: float = 0.0


@dataclass(frozen=True)
class Convertible:
    conversion_ratio: float
    maturity: float = 5.0
    face: float = 100.0
    coupon: float = 0.02            # annual rate on face
    coupon_freq: int = 2
    redemption: Optional[float] = None   # defaults to face
    soft_call: Optional[SoftCall] = None
    puts: Dict[float, float] = field(default_factory=dict)  # time (years) -> put price

    @property
    def redemption_value(self) -> float:
        return self.face if self.redemption is None else self.redemption

    @property
    def coupon_amount(self) -> float:
        return self.face * self.coupon / self.coupon_freq

    def check_grid(self, n_steps: int) -> None:
        per_period = n_steps / (self.maturity * self.coupon_freq)
        if abs(per_period - round(per_period)) > 1e-9 or per_period < 1:
            raise ValueError("n_steps must be a multiple of maturity*coupon_freq")

    def coupon_vector(self, n_steps: int) -> np.ndarray:
        """cpn[k] = coupon paid at step k (k=1..n)."""
        self.check_grid(n_steps)
        out = np.zeros(n_steps + 1)
        step = n_steps // int(round(self.maturity * self.coupon_freq))
        out[step::step] = self.coupon_amount
        return out

    def put_vector(self, n_steps: int) -> np.ndarray:
        """put[k] = put price at step k, or -inf if no put at k."""
        out = np.full(n_steps + 1, -np.inf)
        dt = self.maturity / n_steps
        for t, px in self.puts.items():
            k = int(round(t / dt))
            if 0 < k < n_steps:
                out[k] = px
        return out

    def protection_step(self, n_steps: int) -> int:
        if self.soft_call is None:
            return n_steps + 1
        dt = self.maturity / n_steps
        return int(np.ceil(self.soft_call.protection_years / dt - 1e-12))

    @staticmethod
    def from_yaml(path: str) -> "Convertible":
        with open(path) as f:
            d = yaml.safe_load(f)
        sc = d.pop("soft_call", None)
        puts = {float(k): float(v) for k, v in (d.pop("puts", None) or {}).items()}
        return Convertible(soft_call=SoftCall(**sc) if sc else None, puts=puts, **d)
