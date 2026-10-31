# FARE method details

This note describes the computations implemented in `src/fare/` and used by the experiment runners.

## Overview

Each query uses 56 days of history to forecast seven days. Shape retrieval selects eight windows by normalized history distance. Factor retrieval ranks the same library separately for each of two known future factors, alternates between the rankings, and skips duplicate windows.

For backbone `B`, shape reference `S`, and factor reference `F`:

```text
alpha = sigmoid(X @ theta_alpha)
beta  = sigmoid(X @ theta_beta)
R     = S + alpha * (F - S)
P     = max(0, B + beta * (R - B))
```

The effective weights are `(1-beta, beta*(1-alpha), beta*alpha)`. The gates use origin features, current candidate disagreements, and shrunk historical errors.

## Historical references

For a 56-day window, the scale is `max(mean(history), 1)`. Shape distance is the mean squared distance between scale-normalized histories. For each factor, mismatch is the fraction of the seven future factor values that differ. The factor ranking score is:

```text
0.5 * (shape_distance / (1 + shape_distance) + factor_mismatch)
```

The published configuration uses eight unique references: four turns from factor 1 and four from factor 2. Stable sorting resolves ties in input-bank order. Construct each group's bank in chronological origin order and then series-ID order to match the experiment drivers. Each reference forecast averages scale-normalized observed continuations and rescales by the query's historical mean floor.

The reusable retrieval function processes queries in batches of eight by default (`query_batch_size=8`). This limits the size of temporary history-distance arrays without truncating the eligible bank. Changing the query batch size preserves the per-query arithmetic and selection order.

A reference continuation must be fully observed before the query origin: `reference_origin + 6 days < query_origin`. Callers must also restrict the bank to the correct dataset group. The reusable retrieval API accepts reference and query dates to check this boundary. Without date metadata, eligibility is the caller's responsibility. A bank with fewer than eight windows falls back to the query's preceding seven days; the controlled Favorita experiment instead requires a sufficient bank.

## Historical error state

Only development labels calibrate reliability. On each development origin:

1. Construct features from the state of previously completed origins.
2. Record the current origin's candidate forecasts and labels.
3. Update the error state for subsequent origins after the entire target week has completed.

Per-day error is `clip(abs(actual - candidate) / max(history_mean, 1), 0, 10)`. A series' accumulated errors are shrunk toward the global mean with pseudo-count 14. The fourth historical input is `log1p(number_of_completed_rows)`. Empty global history gives zero errors and support; unseen series use the global prior.

Freeze this state before test inference. Completed test weeks may supply later retrieval windows under the time boundary, while test labels never update calibration or gate parameters.

## Features and fitting

The core exposes the same dataset-specific base feature definitions as the experiment code. Add the four historical inputs to obtain 15 inputs for Favorita, 12 for M5, and 14 for Rossmann or Rohlik. Standardize continuous columns using development-row means and population standard deviations with a floor of `1e-8`; leave binary factors unchanged and add an intercept.

Bank-size features use `log1p(bank_size)` on all four datasets. This is the implementation of the quantity written as `log(bank_size)` for Favorita and M5 in the paper.

Both gates start at zero parameters. The loss is:

```text
sum(sqrt((prediction - actual)**2 + 1)) / sum(actual)
    + 0.001 * sum(non_intercept_parameters**2)
```

The optimizer is SciPy L-BFGS-B with analytical gradients, `maxiter=250`, `ftol=1e-9`, and `gtol=1e-6`. Optimization failure raises an error. The low-level `fit_gate` / `predict_gate` functions accept an already standardized design matrix including an intercept. The `fit_fusion` wrapper fits normalization on the supplied development features and saves it with the gate parameters. Neither wrapper chooses temporal splits for the caller.

## Scoring

WAPE is `sum(abs(actual - prediction)) / sum(actual)`, reported as a ratio unless multiplied by 100. MAE is the mean absolute error on the same rows. Aggregate errors across all evaluation rows before dividing; do not average weekly WAPE values. Test labels are joined only for scoring.
