# convertible-flow

Robust price bounds for convertible bonds: conditional flow matching as a fast path generator,
an imprecise-probability (credal set) wrapper over volatility and credit spread, and
Longstaff-Schwartz backward induction with real contract features.

Idea in one line: instead of pricing under one volatility/spread assumption, define a set of plausible
models, sample cheaply from each, and report the interval of prices rather than a point estimate.

## Status

| Phase | What | State |
|---|---|---|
| 0 | GBM + LSM pricer, binomial cross-check, statistical tests | done |
| 1 | Path-vector flow matching, stylized-fact validation | next |
| 2 | Conditioning on theta = (sigma, s) | todo |
| 3 | Wire generator into pricer | todo |
| 4 | Credal sweep and bounds, backtests | todo |
| 5 | Credit as a jump process | stretch |

## Layout

```
src/contracts.py          term sheet objects (coupons, N-of-M soft call, put schedule)
src/pricer/gbm.py         risk-neutral GBM paths, shape (n_paths, n_steps+1)
src/pricer/lsm.py         Longstaff-Schwartz pricer (takes ANY path array)
src/pricer/binomial.py    CRR tree cross-check
src/validation/stats.py   SE/CI, KS, Jarque-Bera, ACF, Ljung-Box, replication
tests/                    analytic, tree, limit, monotonicity, SE-honesty tests
```

The seam: every path source returns `(n_paths, n_steps+1)`. Swapping GBM for the flow-matching
sampler changes one line and nothing in the pricer.

## Phase 0 methodology

**Paths.** Under the risk-neutral measure, `S_{k+1} = S_k exp((r - q - sigma^2/2) dt + sigma sqrt(dt) Z_k)`,
`Z_k ~ N(0,1)` iid, exact in law on the grid. Antithetic pairs (Z, -Z) are used; the standard error is
computed on pair averages so the variance reduction is not mis-stated.

**Credit via cash/equity split (Tsiveriotis-Fernandes).** Value `V = C + E`. The cash component
(coupons, redemption, put/call proceeds) is discounted at `r + s`; the equity component (conversion value)
at `r`. The spread `s` is therefore one half of the credal parameter theta = (sigma, s).

**Soft call as a barrier.** The issuer calls at the first step `k` such that `S_j >= B` for at least `N` of
the last `M` steps (after call protection). The path ends there and the holder receives
`max(call price, CR * S_k)`. This avoids a second optimal-stopping problem for the issuer.

**Holder decision (LSM).** Going backward, on alive paths regress the realised discounted continuation
value `V_{k}` on basis functions of `x = S_k / S_0`:
`Chat_k(x) = sum_j beta_j phi_j(x)`, with `phi` = cubic polynomial plus hinge features `(x - h)^+`.
Exercise (convert at parity `CR * S_k`, or put at the put price) when `max(parity, put) > Chat_k`.

**Why hinges.** The value has a hockey-stick kink. With a plain cubic basis, LSM converted early on 38.6% of
paths and priced a zero-coupon test case 0.77 below the exact answer (0.65%), even though for q = 0 early
conversion is provably never optimal: `V >= F e^{-r(T-t)} + CR C_BS(S, F/CR) >= CR S` since
`C_BS >= S - K e^{-r(T-t)}`. Adding hinge features cut the bias to 0.04 to 0.07 (about 1 standard error).

**Estimator and error.** `P_hat = (1/N) sum_i V_0^{(i)}`, `SE = s / sqrt(N)`, 95% CI `P_hat +/- 1.96 SE`.
LSM follows a suboptimal exercise policy, so `E[P_hat] <= P_true` (low bias). That is why tests are
one-sided against the tree.

## Phase 0 validation results

| Test | Criterion | Result |
|---|---|---|
| GBM martingale | `|z| < 3.5` for `e^{-(r-q)T} S_T` vs `S_0` | pass |
| GBM log-returns normal | KS and Jarque-Bera p > 0.001; kurtosis within 0.1 of 3 | pass |
| Zero-coupon, q=0, s=0 | LSM = `F e^{-rT} + CR C_BS(S_0, F/CR)` within 4 SE + 0.03 | pass (bias about -0.04 to -0.07) |
| Bond-floor limit | S ~ 0 gives PV of cash flows at `r + s` | pass |
| Tree, same grid, with call | LSM no more than 4 SE above tree; shortfall under 0.5 | pass |
| Tree, same grid, no call | same | pass |
| Monotonicity | price up in sigma, down in s (common random numbers) | pass |
| SE honesty | mean reported SE / between-seed std in (0.7, 1.4) over 30 seeds | pass |

Observed on identical grids: LSM is 0.1 to 0.2% below the tree, shrinking with grid refinement.
Discrete call monitoring matters: a 72-step LSM against a 1440-step tree differs by about 0.2 for a
reason unrelated to the exercise policy, so cross-checks always share the monitoring grid.

## Backtest and statistical validation plan (Phases 1 to 4)

1. **Estimator validation (done above).** Analytic limits, independent tree, replication across seeds.
2. **Generator validation (Phase 1/2), on a held-out time period.**
   - Marginals: KS and Jarque-Bera; Hill tail index; skewness and kurtosis with bootstrap CIs.
   - Dependence: ACF of squared returns, Ljung-Box `Q = n(n+2) sum_{l<=h} rho_l^2/(n-l) ~ chi2(h)`.
   - Distributional match: two-sample energy distance or MMD with a permutation p-value.
3. **Controlled backtest of the bounds (no market data needed).** Generate "reality" from a known process
   outside the GBM family (Heston, jump-diffusion), compute its high-precision price, and check whether
   `[P_low, P_high]` from the flow-matching sweep contains it. Report containment rate, interval width,
   and how width shrinks as the ambiguity set is tightened.
4. **Walk-forward backtest on real data.** At each date `t`, build `Theta_t` from a trailing window only
   (no look-ahead), price the interval, compare with the observed market price `M_t`.
   Metrics: containment rate `(1/T) sum 1{M_t in [P_low, P_high]}`, mean signed distance outside the band,
   and a trading rule (long when `M_t < P_low`, delta-hedged) evaluated by mean P&L with Newey-West HAC
   t-statistics and a block bootstrap, against the point-estimate strategy.
   Caveat: historical convertible prices are not freely available; this step depends on a data source.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q          # or: python run_tests.py
```
