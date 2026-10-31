"""Forecast-origin anchored features shared by Ridge and XGBoost.

Nothing in this module reads training_labels() or truth().  In particular,
rolling statistics are anchored at the origin, not at each forecast day.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Iterator

import numpy as np

FEATURE_VERSION = "origin56-direct7-v1"
CONTEXT = 56
HORIZON = 7
ROLL_WINDOWS = (7, 14, 28, 56)
SEASONAL_LAGS = (7, 14, 28, 56)


def feature_names(covariate_names) -> list[str]:
    names = [f"history_lag_{lag}" for lag in range(1, CONTEXT + 1)]
    names += [f"origin_roll_{window}_{stat}" for window in ROLL_WINDOWS
              for stat in ("mean", "std", "min", "max", "zero_fraction", "slope")]
    names += [f"forecast_day_lag_{lag}" for lag in SEASONAL_LAGS]
    names += [f"horizon_{h}" for h in range(1, HORIZON + 1)]
    names += ["target_weekday_sin", "target_weekday_cos", "target_year_sin", "target_year_cos"]
    names += [f"known_future_{name}" for name in covariate_names]
    return names


def rows_for(panel, origin, rows=None) -> np.ndarray:
    """Return absolute panel rows in precisely the requested prediction order."""
    out = panel.rows(origin) if rows is None else rows
    out = np.asarray(out, dtype=np.int64)
    if out.ndim != 1:
        raise ValueError("Panel rows must be a one-dimensional integer index array")
    return out


def row_chunks(panel, origin, batch_rows: int, rows=None) -> Iterator[np.ndarray]:
    if batch_rows < 1:
        raise ValueError("batch_rows must be positive")
    indices = rows_for(panel, origin, rows)
    for start in range(0, len(indices), batch_rows):
        yield indices[start:start + batch_rows]


def design_matrix(panel, origin, rows=None, dtype=np.float32) -> np.ndarray:
    """Return (N * 7, F) ordered by entity then horizon.

    Historical lag 1 is the day before the forecast origin.  The four
    forecast-day seasonal lags have values in the supplied 56-day context even
    at horizon 7.  Calendar values and future covariates are known in advance.
    """
    history = np.asarray(panel.context(origin, rows=rows), dtype=np.float64)
    if history.ndim != 2 or history.shape[1] != CONTEXT:
        raise ValueError(f"Expected N x {CONTEXT} history, got {history.shape}")
    if not np.isfinite(history).all():
        raise ValueError("Non-finite history; inputs may not be imputed silently")
    n = history.shape[0]
    cov = np.asarray(panel.future_covariates(origin, rows=rows), dtype=np.float64)
    if cov.shape != (n, HORIZON, len(panel.covariate_names)):
        raise ValueError(f"Unexpected future covariate shape: {cov.shape}")
    if not np.isfinite(cov).all():
        raise ValueError("Non-finite known future covariates")
    base = [history[:, ::-1]]
    for window in ROLL_WINDOWS:
        values = history[:, -window:]
        time = np.arange(window, dtype=np.float64) - (window - 1) / 2
        slope = values @ time / np.dot(time, time)
        base.append(np.column_stack((values.mean(axis=1), values.std(axis=1),
                                     values.min(axis=1), values.max(axis=1),
                                     (values == 0).mean(axis=1), slope)))
    historical = np.concatenate(base, axis=1)
    names = feature_names(panel.covariate_names)
    result = np.empty((n, HORIZON, len(names)), dtype=dtype)
    offset = historical.shape[1]
    result[:, :, :offset] = historical[:, None, :]
    for lag in SEASONAL_LAGS:
        # For day h=0: history[-lag]; h=6: history[6-lag].
        result[:, :, offset] = history[:, CONTEXT + np.arange(HORIZON) - lag]
        offset += 1
    result[:, :, offset:offset + HORIZON] = np.eye(HORIZON)[None, :, :]
    offset += HORIZON
    first = date.fromisoformat(str(origin)[:10])
    days = [first + timedelta(days=h) for h in range(HORIZON)]
    weekday = np.array([day.weekday() for day in days], dtype=np.float64)
    year_day = np.array([day.timetuple().tm_yday - 1 for day in days], dtype=np.float64)
    calendar = np.column_stack((np.sin(2 * np.pi * weekday / 7),
                                np.cos(2 * np.pi * weekday / 7),
                                np.sin(2 * np.pi * year_day / 365.25),
                                np.cos(2 * np.pi * year_day / 365.25)))
    result[:, :, offset:offset + 4] = calendar[None, :, :]
    result[:, :, offset + 4:] = cov
    result = result.reshape(n * HORIZON, len(names))
    if not np.isfinite(result).all():
        raise ValueError("Feature conversion produced non-finite values")
    return np.ascontiguousarray(result)


def training_batches(panel, origins, batch_rows: int, dtype=np.float32):
    """Yield every legal training row, without sampling entities or origins."""
    for origin in origins:
        for rows in row_chunks(panel, origin, batch_rows):
            features = design_matrix(panel, origin, rows, dtype=dtype)
            labels = np.asarray(panel.training_labels(origin, rows=rows), dtype=dtype)
            if labels.shape != (len(rows), HORIZON) or not np.isfinite(labels).all():
                raise ValueError(f"Invalid training labels at {origin}")
            yield features, np.ascontiguousarray(labels.reshape(-1))
