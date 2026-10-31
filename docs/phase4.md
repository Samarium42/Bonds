# Phase 4: credal sweep and controlled backtest

## What is computed

Credal set Theta = [sigma_lo, sigma_hi] x [rho_lo, rho_hi] x [s_lo, s_hi], where sigma is annual volatility (applied
analytically), rho is volatility persistence (the flow's learned condition) and s is the credit spread (discounts the cash
component). Lower and upper prevision: P_low = inf over Theta of P(theta), P_high = sup over Theta of P(theta).

Estimation choices that keep the interval honest (`src/credal/sweep.py`):

* **Common random numbers.** One unit-variance path bank per rho is reused across every (sigma, s). The surface is smooth and
  its monotonicity is testable (price rose with sigma in 93.6% of grid steps and fell with s in 100%; the sigma exceptions sit
  at noisy high-sigma nodes).
* **Winner's curse.** The inf/sup of a noisy surface is biased outward. The extreme point is selected on bank A and
  re-evaluated on an independent bank B. `tests/test_credal.py` shows on a flat noise surface that the one-stage inf is
  biased low while the two-stage value is unbiased.
* **Tail cap.** The fitted family has a power-law tail (Hill index about 3.6), so exponential moments are infinite and MC price
  variance is theoretically infinite; single paths with |Z| of 40 to 90 inflated the SE up to 5 times. Weekly returns are
  truncated at 10 sigma and re-standardised. Measured at sigma 0.5, s 0.03, rho 0.95, 15k paths, 8 seeds:

| cap (sigma) | mean price | mean MC SE | sd of price across seeds |
|---|---|---|---|
| none | 114.771 | 0.376 | 0.187 |
| 20 | 114.730 | 0.264 | 0.087 |
| 10 (used) | 114.743 | 0.246 | 0.121 |
| 6 | 114.772 | 0.240 | 0.106 |

At 10 sigma the cap clips 7e-5 of weekly returns and removes 0.6% of variance. The price moves by 0.03, inside the noise.
This is a modelling choice, not a free lunch: real weekly equity moves rarely exceed 5 to 6 sigma.

## Where the interval width comes from (40k paths, ρ grid 0.90 to 0.985)

| sigma | s | P across rho (0.90 / 0.93 / 0.96 / 0.985) | rho range | GBM |
|---|---|---|---|---|
| 0.40 | 0.02 | 112.62 / 112.65 / 112.30 / 111.53 | 1.12 | 113.00 |
| 0.60 | 0.02 | 119.69 / 119.64 / 119.28 / 117.57 | 2.12 | 120.26 |
| 0.80 | 0.02 | 126.00 / 126.05 / 125.55 / 122.14 | 3.91 | 126.57 |

A +-28% volatility box alone moves the price by 7.4 to 9.8; the spread range 0.02 to 0.05 by about 3.7; the model-form
dimension rho by 1 to 4. For this contract the interval is dominated by volatility and spread uncertainty. GBM and the
clustered family agree within 0.7 for rho up to 0.96 and diverge only at rho 0.985 (family 1.5 to 4.4 lower).

## Controlled backtest (known truth)

Reality is a two-state regime-switching volatility process with Student-t shocks, outside the GARCH family, so the true price
P* is computable. The modeler sees sigma_hat = sigma* e^eta (eta ~ N(0, 0.15^2)) and s_hat = s* e^nu (nu ~ N(0, 0.2^2)),
builds the 90% box around them with rho in [0.90, 0.985], and reports [P_low, P_high]. 150 replications per scenario.
Paths come from the exact family simulator here (flow sampling is too slow on 2 CPU cores; see "Still to do").

Second run (σ grid capped at 1.1, 30k paths per node; `results/controlled_backtest_v2.json`):

| regime | sigma* | s* | P* | coverage | mean width | GBM mean abs err |
|---|---|---|---|---|---|---|
| mild | 0.30 | 0.02 | 108.76 | 0.973 | 8.8 | 1.59 |
| mild | 0.30 | 0.05 | 104.62 | 0.967 | 11.0 | 1.79 |
| mild | 0.45 | 0.02 | 114.88 | 0.987 | 11.4 | 2.01 |
| mild | 0.45 | 0.05 | 110.82 | 0.973 | 13.9 | 2.36 |
| mild | 0.60 | 0.02 | 120.31 | 0.813 | 12.6 | 2.39 |
| mild | 0.60 | 0.05 | 116.49 | 0.947 | 15.8 | 2.68 |
| stress | 0.30 | 0.02 | 108.21 | 0.960 | 9.0 | 2.09 |
| stress | 0.30 | 0.05 | 103.57 | 0.987 | 11.0 | 2.13 |
| stress | 0.45 | 0.02 | 114.12 | 0.980 | 11.4 | 1.96 |
| stress | 0.45 | 0.05 | 109.56 | 0.980 | 13.8 | 2.58 |
| stress | 0.60 | 0.02 | 119.29 | 0.767 | 12.5 | 2.92 |
| stress | 0.60 | 0.05 | 115.13 | 0.947 | 15.5 | 2.89 |

Pooled coverage 0.943 (mild) and 0.937 (stress). If the model were right and price monotone in sigma and s, coverage would be
at least P(both sigma* and s* in their boxes) = (2 Phi(1.645) - 1)^2 = 0.81.

## How to read this (caveats that matter)

1. **Two cells fall below the 0.81 reference** (0.813 and 0.767, both at sigma* 0.6, s* 0.02). The cause is misspecification:
   the reality prices at 120.31, above every family member at the TRUE parameters (119.69 at rho 0.90, the highest), because
   the family has fat-tailed daily shocks and contains no near-Gaussian member. A credal set only covers a truth that some
   member can reach; include low-persistence / near-Gaussian members or widen rho downward.
2. **The Wilson intervals understate the uncertainty.** The 150 replications in a scenario share one realised pair of price
   surfaces, so they are not independent. The same scenario (sigma* 0.6, s* 0.02, mild) gave 0.973 in the first run
   (20k paths, wider grid) and 0.813 in the second. Treat per-scenario coverage as uncertain by about +-15 points. Proper
   uncertainty needs independent surface replications (cluster bootstrap over surface seeds); not done.
3. **Intervals are wide relative to point error.** Mean widths of 9 to 16 against GBM point errors of 1.6 to 2.9 (the GBM point
   estimate falls inside the interval in essentially every replication). GBM's error here is mostly parameter-estimation
   noise (bias at most 1.7), so this test does NOT show the credal interval being more accurate than GBM; it shows it is a
   valid, conservative statement of uncertainty. The value is the robust cheap/rich verdict, not a tighter estimate.
4. **Numerical quality limits the usable region.** MC SE is below 0.5 up to sigma 1.0, but some nodes at sigma 1.1 and
   rho 0.985 still have SE up to 28 (interpolation error vs direct pricing up to 11.6). `quality_report` flags such nodes
   and `run_sweep.py` warns when a box touches them; do not trust bounds that reach them. Scenarios with sigma* 0.3 had 23 to 28
   of 150 boxes clipped by the grid minimum (sigma 0.2), which narrows those intervals (conservative for the coverage claim).
5. **Not yet established:** performance with the flow as the generator (only the exact family was used for this backtest),
   and anything on real market prices.

## Still to do

* Rerun `scripts.controlled_backtest --source flow --model results/family_model_feat.pt` on a machine with more cores
  (flow sampling is about 0.1 s per path on 2 CPU cores; the 8 x 4 bank sizes are cached under `results/zbank`).
* Add replicated surfaces for a valid coverage uncertainty, and extend rho downward to include a near-Gaussian member.
* Real data: `python -m src.data.download`, `python -m scripts.calibrate`, then `python -m scripts.run_sweep` with the
  observed spot, market price and the calibrated ranges. Historical convertible prices are not freely available, so a
  walk-forward backtest against market prices needs a data source.
