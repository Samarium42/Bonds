"""Phase 5c step 2: credal bounds with and without the credit-model dimensions (select on bank A, evaluate on bank B)."""
import json, itertools
import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI

A = json.load(open("results/credit_credal_A.json")); B = json.load(open("results/credit_credal_B.json"))
sig, spr, beta, rec = A["sig"], A["spr"], A["beta"], A["rec"]
SIG = (0.45 * np.exp(-1.645 * 0.15), 0.45 * np.exp(1.645 * 0.15)); SPR = (0.035 * np.exp(-1.645 * 0.20), 0.035 * np.exp(1.645 * 0.20))
BETA, REC = (0.0, 2.0), (0.0, 0.4)
print(f"sigma box {SIG[0]:.3f}..{SIG[1]:.3f}  s box {SPR[0]:.4f}..{SPR[1]:.4f}  beta {BETA}  recovery {REC}  rho={A['rho']}")


def surf(D, which):
    if which == "flat":
        return RGI((sig, spr), np.array(D["flat"]))
    return RGI((sig, spr, beta, rec), np.array(D["jd"]))


def bounds(which, sb, pb, bb, rb, fine=11):
    """Box bounds; point fixed (a degenerate range) when lo == hi."""
    ax = [np.linspace(*sb, fine), np.linspace(*pb, fine)]
    if which == "jd":
        ax += [np.linspace(*bb, fine if bb[0] < bb[1] else 1), np.linspace(*rb, fine if rb[0] < rb[1] else 1)]
    pts = np.array(list(itertools.product(*ax)))
    fA, fB = surf(A, which), surf(B, which)
    vA, vB = fA(pts), fB(pts)
    lo, hi = vA.argmin(), vA.argmax()
    return dict(lo1=float(vA[lo]), hi1=float(vA[hi]), lo=float(vB[lo]), hi=float(vB[hi]),
                arg_lo=pts[lo].round(4).tolist(), arg_hi=pts[hi].round(4).tolist())


cases = {
    "flat spread (Phase 4 style), sigma x s": ("flat", SIG, SPR, None, None),
    "jump, beta=1 R=0.2 fixed, sigma x s": ("jd", SIG, SPR, (1.0, 1.0), (0.2, 0.2)),
    "jump, credit model only (sigma, s at centre)": ("jd", (0.45, 0.45), (0.035, 0.035), BETA, REC),
    "jump, full credal: sigma x s x beta x R": ("jd", SIG, SPR, BETA, REC),
}
out = {}
for name, args in cases.items():
    b = bounds(*args)
    b["width"] = b["hi"] - b["lo"]
    out[name] = b
    print(f"{name:48s} A:[{b['lo1']:.2f},{b['hi1']:.2f}]  two-stage B:[{b['lo']:.2f},{b['hi']:.2f}]  width {b['width']:.2f}   argmin {b['arg_lo']} argmax {b['arg_hi']}")
w0 = out["flat spread (Phase 4 style), sigma x s"]["width"]; w1 = out["jump, full credal: sigma x s x beta x R"]["width"]
print(f"\nwidth added by credit-model dimensions: {w1 - w0:+.2f} ({100*(w1-w0)/w0:+.1f}% of the flat-spread width)")
# bank-to-bank consistency of the surface (sampling noise gauge)
d = np.array(A["jd"]) - np.array(B["jd"]); se = np.hypot(np.array(A["se"]), np.array(B["se"]))
print(f"bank A - bank B over {d.size} jd nodes: mean {d.mean():+.3f}, sd {d.std():.3f}, mean SE of difference {se.mean():.3f}")
json.dump(out, open("results/credit_credal.json", "w"), indent=1)
