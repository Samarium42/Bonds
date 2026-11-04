# Phase 5: credit as a jump to default

## Model

Default is a jump with stock-dependent intensity (Carr-Linetsky jump-to-default CEV):

    lambda_t = min(5, lam0 * (S_t / S0)^(-beta)),   beta >= 0   (beta = 0: constant hazard)

At default the stock is worthless and the bond pays `recovery * face`. Given the pre-default path, survival to step k is
w_k = exp(-dt * sum_{j<k} lambda_j), so default times are never simulated (Cox process, no extra Monte Carlo variance):

    cash_k = e^{-r dt} [ q_k cash_{k+1} + (1 - q_k) * recovery * face ],   eq_k = e^{-r dt} q_k eq_{k+1},   q_k = e^{-lambda_k dt}.

Risk-neutral drift: E[S_k 1{tau > k}] = E[S_k w_k] must equal S0 e^{(r-q) t_k}. Pre-default paths are rescaled per time step,
S_k <- c_k S_k, with the fixed point solved by damped iteration (residual below 1e-10 in every experiment).

Calibration: lam0 is chosen so the risky zero-coupon bond (same recovery, same paths) equals `face * exp(-(r + s) T)`. The
flat-spread Phase 4 model and the jump model therefore agree on the straight bond; any convertible price gap is credit-model risk.

Code: `src/credit/jump.py`; `price_lsm` gained optional `hazard`, `recovery`, `keep_pv` arguments (default behaviour unchanged).

## Validation (`tests/test_credit.py`, 6 tests, all pass; needs scipy)

* hazard 0 reproduces the Phase 0 price to 1e-9;
* zero-coupon value equals the exact discrete closed form to 1e-9, and the continuous-time formula to within 0.1;
* a convertible with ~0 conversion value equals the deterministic risky coupon bond to 1e-6;
* explicit default-time simulation (Exp(1) thresholds, beta = 1.5, 60k paths): E[S_T 1{alive}] e^{-rT} = S0 e^{-qT} within 4 SE, and
  the explicit and survival-weighted European payoffs agree within 4 SE, with the weighted one having lower variance;
* calibration matches the flat-spread zero-coupon price to 1e-4 for (beta, R, s) in {(0, .4, .04), (1, .4, .04), (2, 0, .07)}.

## Flat spread versus jump (`scripts/credit_gap.py`, `results/credit_gap.json`)

3-year convertible from Phase 4, rho = 0.95, 30k paths shared by both models (paired SEs). 36 cases: sigma in {0.3, 0.6},
s in {0.02, 0.05, 0.08}, recovery in {0, 0.4}, beta in {0, 1, 2}.

* jump minus flat price ranges from -0.98 to +2.69 (mean absolute 0.75); 27 of 36 differences exceed 3 paired SE;
* the equivalent flat spread (the spread at which the flat model reproduces the jump price) differs from the straight spread by
  -0.0236 to +0.0102;
* mean difference by beta: 0.15 (beta 0), 0.44 (1), 1.11 (2); by recovery: 0.21 (R 0), 0.92 (R 0.4); by s: 0.14 (0.02), 0.58 (0.05),
  0.97 (0.08); by sigma: 0.88 (0.3), 0.25 (0.6);
* with beta = 0 and R = 0 the jump model is slightly BELOW the flat model (-0.12 to -0.98), growing with sigma and s.

The sign pattern is measured. Explanations (stock-dependent hazard moving losses into low-stock states; recovery value) are
plausible but were not isolated by an experiment here.

## Credal set with credit-model dimensions (`scripts/credit_credal_*.py`, `results/credit_credal*.json`)

Box at rho = 0.95: sigma 0.352..0.576 and s 0.0252..0.0486 (the Phase 4 90% boxes around sigma-hat 0.45 and s-hat 0.035),
beta in [0, 2], recovery in [0, 0.4], lam0 recalibrated at every node so the straight bond always matches s. Select on bank A,
evaluate on independent bank B (20k paths each; 3 x 3 x 3 x 3 nodes, multilinear interpolation, 11 points per axis).

| credal set | bounds (bank B) | width |
|---|---|---|
| flat spread, sigma x s | 106.81 .. 117.97 | 11.16 |
| jump, beta 1, R 0.2 fixed, sigma x s | 107.23 .. 118.04 | 10.81 |
| jump, credit model only (sigma, s at centre) | 111.80 .. 113.50 | 1.69 |
| jump, full: sigma x s x beta x R | 106.50 .. 118.85 | 12.35 |

The credit-model dimensions widen the interval by 1.19 (10.7%). Volatility and spread uncertainty remain the dominant
sources. The bound ends are attained at box corners (low: smallest sigma, largest s, beta 0, R 0; high: largest sigma, smallest s,
beta 2, R 0.4).

## Caveats

* One path bank per credal evaluation, rho fixed at 0.95, one bond, paths from the exact family simulator (not the flow);
  rho is not in this box.
* beta and the recovery range are assumptions, not estimated. There is no CDS or bond-spread data here, so the
  "model-implied versus market-implied spread" check of the project plan is only done as the model-versus-model basis above.
  With real data: calibrate s from the straight bond or CDS, then compare.
* The 4D grid has 3 nodes per axis; interpolation error was not measured (Phase 4 found it sizeable at some nodes). Bank-to-bank
  node differences have sd 0.21 against a mean difference SE of 0.27, consistent with pure sampling noise.
* The straight bond is matched through the zero-coupon bond only; coupon-bond and CDS term structure are not matched.
* Hazard is capped at 5 per year; deeply out-of-the-money paths hit the cap.
