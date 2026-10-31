"""Fit a past-error selector on early development origins and score later origins."""

from __future__ import annotations

import numpy as np

CANDIDATES = ("backbone", "global_static", "shape_half", "factor_half")

def select_global_weights(actual: np.ndarray, parts: np.ndarray) -> tuple[tuple[int, int, int], float]:
    choices = [(i, j, 20 - i - j) for i in range(21) for j in range(21 - i)]
    assert len(choices) == 231
    errors = []
    for weights in choices:
        pred = np.maximum(parts @ np.asarray(weights, dtype=np.float64) / 20, 0)
        errors.append(float(np.abs(actual - pred).sum()))
    best_error = min(errors)
    ties = [index for index, error in enumerate(errors)
            if np.isclose(error, best_error, rtol=0, atol=1e-9)]
    index = min(ties, key=lambda idx: (-choices[idx][0], -choices[idx][1]))
    return choices[index], float(best_error / actual.sum())

def select_by_history(
    actual: np.ndarray,
    candidates: np.ndarray,
    cohort: np.ndarray,
    series_code: np.ndarray,
    train: np.ndarray,
    allowed: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    n_series = int(series_code.max()) + 1
    train_rows = np.flatnonzero(train)
    train_cohort = cohort[train]
    support = np.bincount(train_cohort, minlength=6)
    errors = np.abs(actual[:, None] - candidates)

    cohort_choice = np.full(6, 1, dtype=np.int8)  # Low-support fallback: global static.
    for group in range(6):
        if support[group] < 200:
            continue
        group_rows = train_rows[train_cohort == group]
        losses = errors[np.ix_(group_rows, np.asarray(allowed))].sum(axis=0)
        cohort_choice[group] = allowed[int(np.argmin(losses))]

    selected_by_cohort = cohort_choice[cohort]
    train_series = series_code[train]
    train_counts = np.bincount(train_series, minlength=n_series)
    baseline_error = errors[train_rows, selected_by_cohort[train]]
    best_bound = np.full(n_series, -np.inf, dtype=np.float64)
    override = np.full(n_series, -1, dtype=np.int8)
    enough = train_counts >= 14

    for candidate in allowed:
        gains = baseline_error - errors[train_rows, candidate]
        gain_sum = np.bincount(train_series, weights=gains, minlength=n_series)
        gain_squares = np.bincount(train_series, weights=gains * gains,
                                   minlength=n_series)
        denominator = np.maximum(train_counts, 1)
        gain_mean = gain_sum / denominator
        variance = np.maximum(
            (gain_squares - gain_sum * gain_sum / denominator)
            / np.maximum(train_counts - 1, 1), 0,
        )
        stderr = np.sqrt(variance / denominator)
        lower_bound = gain_mean - 1.64 * stderr
        improve = enough & (lower_bound > 0) & (lower_bound > best_bound)
        best_bound[improve] = lower_bound[improve]
        override[improve] = candidate

    selected = np.where(override[series_code] >= 0,
                        override[series_code], selected_by_cohort)
    rows = np.arange(len(actual))
    cohort_prediction = candidates[rows, selected_by_cohort]
    selective_prediction = candidates[rows, selected]
    metadata = {
        "cohort_support_train": support.astype(int).tolist(),
        "cohort_choice": [CANDIDATES[int(index)] for index in cohort_choice],
        "series_with_train_history": int((train_counts > 0).sum()),
        "series_overridden": int((override >= 0).sum()),
        "late_rows_overridden": int(((override[series_code] >= 0) & ~train).sum()),
        "late_choice_rows": {
            CANDIDATES[index]: int(((selected == index) & ~train).sum())
            for index in allowed
        },
    }
    return cohort_prediction, selective_prediction, selected, override[series_code] >= 0, metadata


def history_hard_control(frame, parts, actual, fit, weights, test):
    """Historical hard selection among B, global static and the two 50:50 blends.

    ``fit`` marks labelled development rows that fit the selector; ``test`` marks
    the rows returned. Cohort/support thresholds and tie handling are those of
    select_by_history, as in the Favorita/M5 control.
    """
    import pandas as pd

    static = np.maximum(parts @ (np.asarray(weights) / 20), 0)
    candidates = np.column_stack([
        parts[:, 0], static, parts[:, [0, 1]].mean(1), parts[:, [0, 2]].mean(1),
    ])
    scale = frame.history_mean.to_numpy()
    cohort = 2 * np.where(scale < 1, 0, np.where(scale < 5, 1, 2))
    cohort += (frame.history_zero_fraction.to_numpy() >= .5).astype(int)
    codes, _ = pd.factorize(frame.series_id, sort=True)
    _, prediction, _, _, record = select_by_history(
        actual, candidates, cohort, codes, fit, (0, 1, 2, 3),
    )
    return np.maximum(prediction[test], 0), record
