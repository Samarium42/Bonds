
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

**Leverage (corrected).** My first reading of this was wrong. I compared Pearson corr(r_t, r_{t+1}^2) and quoted a
sampling error of about 0.0015 by treating all observations as independent. With fat tails and windows as the independent
unit the real standard error is 0.004 to 0.015, and the apparent "flow loses the leverage effect" gap was mostly a
statistic dominated by extreme tails. A rank-based version (Spearman corr of r_t with |r_{t+1}|, about 3 to 8 times less
noisy) gives, on cached samples (3000 flow paths and 20000 reference paths per rho, N=156):

| rho | truth | baseline flow | flow with sign/square input features |
|---|---|---|---|
| 0.93 | -0.0305 +- 0.0006 | -0.0290 +- 0.0015 (z +0.9) | -0.0305 +- 0.0015 (z 0.0) |
| 0.97 | -0.0337 +- 0.0006 | -0.0318 +- 0.0015 (z +1.2) | -0.0335 +- 0.0015 (z +0.1) |
| 0.985 | -0.0339 +- 0.0005 | -0.0325 +- 0.0015 (z +0.9) | -0.0344 +- 0.0015 (z -0.3) |
| pooled deficit | | +0.0016 +- 0.0009 (z +1.8) | -0.0001 +- 0.0009 (z -0.1) |

Adding x^2 and tanh(x) as input channels (`--feat`, now the default; `--no-feat` gives the baseline) is the only change
between the two models (same data, steps, seed). It does not hurt anything else: std, ACF of squares at lags 1 to 4, Hill
index and energy tests are equal to or slightly better than baseline. The paired difference between the two models is
not itself significant.

What remains: the Pearson-on-squares statistic, which is driven by the extreme tail, still shows a deficit for the feature
model (pooled +0.0086 +- 0.0047, z +1.8), consistent with slightly thinner extreme tails at high rho (Hill 2.99 vs 2.92 and
std 0.978 vs 0.991 at rho=0.985). Reading: bulk asymmetry is captured, extreme-tail asymmetry is slightly under-represented
at the top of the rho range. A fast N=52 proxy did not reproduce the original gap (z -0.7 and -1.4 for the baseline), so
fixes should be judged at N=156.

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
