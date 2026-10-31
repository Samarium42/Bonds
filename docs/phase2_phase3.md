
## Phase 2: conditional generator on a calibrated family (option c: daily calibration + parametric training paths)

**Why this design.** Phase 1 showed the generator needs many independent windows, which real weekly equities cannot
supply. So the flow is trained on paths from a daily GJR-GARCH(1,1)-t family (summed to weekly), with parameters
calibrated on DAILY data (about 5 times more observations, which tightens the persistence estimate: sd of rho fell from
0.023 at n=2000 to 0.010 at n=5000 in `tests/test_family.py`). The flow is then an amortised simulator over the credal
set, not a model that discovers the empirical law. That is the claim to make in the write-up.

**What is learned vs analytic.** The family is scale-equivariant, so volatility sigma is applied analytically after
sampling (exact). The flow is conditioned on the persistence rho = alpha + gamma/2 + beta, which controls volatility
clustering and is not analytic. The credal set is therefore over (sigma, rho, s); the spread s enters the pricer only.

**Calibration** (`scripts/calibrate.py`, `src/calibrate/garch.py`). Maximum likelihood with Student-t innovations; the
variance recursion is a linear filter so each fit takes about 0.1s. Credal ranges for rho use a PARAMETRIC bootstrap with a
bias-corrected interval: a moving-block bootstrap cuts the long volatility memory, biased refitted rho low, and produced
intervals that did not contain the point estimate. On simulated data the corrected intervals contain the true rho.

**Steerability test** (`scripts/validate_family.py`; flow vs fresh paths from the true family; 3000 paths each; 25 Heun steps):

| rho | std ref / gen | ACF of squares lag 1 (ref / gen) | Hill ref / gen | Energy test p |
|---|---|---|---|---|
| 0.91 | 1.001 / 1.004 | 0.083 / 0.084 | 3.83 / 3.78 | 0.030 |
| 0.93 | 0.999 / 1.000 | 0.088 / 0.090 | 3.72 / 3.73 | 0.535 |
| 0.95 | 0.998 / 1.003 | 0.102 / 0.103 | 3.57 / 3.58 | 0.255 |
| 0.97 | 1.011 / 1.000 | 0.119 / 0.122 | 3.33 / 3.33 | 0.740 |
| 0.985 | 0.987 / 0.985 | 0.136 / 0.141 | 2.92 / 3.02 | 0.175 |
| 0.85 (outside training range) | 0.998 / 1.071 | 0.057 / 0.088 | 3.96 / 3.62 | 0.005 (fails) |

All in-range energy tests exceed the Bonferroni threshold 0.05/5 = 0.01, and clustering increases monotonically in rho.
Outside the training range the generator degrades, so the training range must cover the credal set with margin.

**Known weakness: leverage.** The reference lag-1 correlation corr(r_t, r_{t+1}^2) is about -0.04 to -0.06 and the flow
reproduces only part of it (-0.03 at rho=0.91, about 0 at rho=0.985). With roughly 4.6e5 observations per cell the sampling
error is about 0.0015, so this shortfall is real. It matters for equity convertibles (skew), and is open work.

## Phase 3: wiring into the pricer

`simulate_flow` and the reference family simulator share one post-processing function (`paths_from_weekly`: demean, scale
by weekly vol, exponentiate, martingale-correct), so the pricer sees identical treatment of both. `scripts/phase3_wiring.py`
prices one convertible three ways (GBM, true family, flow).

Result of the small run that fits the sandbox (rho=0.95, 5000 paths per source, bond: 3y, 4% coupon, 1-of-1 soft call at 170,
puts at years 1 and 2, r=3%, q=1%, spread 3%):

| sigma | GBM | Family | Flow | z (flow - family) |
|---|---|---|---|---|
| 0.35 | 109.33 +- 0.30 | 109.47 +- 0.30 | 109.28 +- 0.30 | -0.45 |
| 0.50 | 115.01 +- 0.41 | 115.17 +- 0.43 | 115.08 +- 0.44 | -0.14 |
| 0.70 | 121.29 +- 0.62 | 121.37 +- 0.73 | 121.96 +- 0.67 | +0.59 |

This rules out gross wiring errors. It does NOT establish that the flow price matches the family price to better than about
one point, and it cannot separate GBM from the family (all gaps inside one standard error). Cells share the same draws
across sigma, so the three z-values are not independent. Flow sampling on 2 CPU cores took about 10 minutes per 5000 paths;
the full grid (`--paths 30000`, three rho values) should be run on the Mac with common random numbers for a sharper comparison.
