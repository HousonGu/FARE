"""Error histories and FARE's two bounded gates.

Includes the smoothed fitting objective, analytical gradients, feature
normalization, chronological error accumulation, and frozen inference state.
"""
from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from threadpoolctl import threadpool_limits

SHRINK_ROWS = 14.0
PENALTY = 0.002
HISTORY_NAMES = ["hist_backbone_error", "hist_shape_error", "hist_factor_error", "hist_log_support"]


def _vector(value, name, n=None, nonnegative=True):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 1 or (n is not None and len(value) != n):
        raise ValueError(f"{name} must be a row-aligned one-dimensional array")
    if not np.isfinite(value).all() or (nonnegative and (value < 0).any()):
        raise ValueError(f"Invalid {name}")
    return value


def _features(value, n=None):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or not len(value) or not value.shape[1] or (n is not None and len(value) != n):
        raise ValueError("Features must be a nonempty row-aligned two-dimensional array")
    if not np.isfinite(value).all():
        raise ValueError("Features must be finite")
    return value


def fit_gate(x, y, b, s, f, single=False):
    """Fit the original gate on an already standardized design with intercept.

    ``x[:, 0]`` is an all-one intercept; all other columns are regularized.
    Returns ``(parameters, optimization_record)``. Candidate values must be
    nonnegative before fitting, as in the benchmark's candidate preparation.
    """
    x = _features(x)
    y = _vector(y, "actual", len(x))
    b, s, f = (_vector(value, name, len(x)) for value, name in
               ((b, "backbone"), (s, "shape"), (f, "factor")))
    if not np.array_equal(x[:, 0], np.ones(len(x))):
        raise ValueError("First design column must be the all-one intercept")
    k = x.shape[1]
    denominator = float(y.sum())
    if denominator <= 0:
        raise ValueError("Zero training actual sum")

    def objective(theta):
        if single:
            beta = expit(x @ theta)
            pred = b + beta * (s - b)
        else:
            beta = expit(x @ theta[:k])
            alpha = expit(x @ theta[k:])
            reference = s + alpha * (f - s)
            pred = b + beta * (reference - b)
        residual = pred - y
        smooth = np.sqrt(residual * residual + 1.0)
        penalty = PENALTY * (np.dot(theta[1:k], theta[1:k])
                             + (0 if single else np.dot(theta[k + 1:], theta[k + 1:])))
        gradient_scale = residual / smooth / denominator
        if single:
            gradient = x.T @ (gradient_scale * beta * (1 - beta) * (s - b))
            gradient[1:] += 2 * PENALTY * theta[1:]
        else:
            gb = x.T @ (gradient_scale * beta * (1 - beta) * (reference - b))
            ga = x.T @ (gradient_scale * beta * alpha * (1 - alpha) * (f - s))
            gradient = np.concatenate([gb, ga])
            gradient[1:k] += 2 * PENALTY * theta[1:k]
            gradient[k + 1:] += 2 * PENALTY * theta[k + 1:]
        return float(smooth.sum() / denominator + penalty), gradient

    with threadpool_limits(limits=4):
        result = minimize(objective, np.zeros(k if single else 2 * k), jac=True,
                          method="L-BFGS-B",
                          options={"maxiter": 250, "ftol": 3e-9, "gtol": 3e-6})
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"Gate fit failed: {result.message}")
    return result.x, {"iterations": int(result.nit), "objective": float(result.fun),
                      "parameters": result.x.tolist(), "converged": True}


def predict_gate(x, theta, b, s, f=None):
    """Return ``(prediction, beta, alpha)``; alpha is None for a B+S gate."""
    x = _features(x)
    k = x.shape[1]
    theta = _vector(theta, "parameters", k if f is None else 2 * k, nonnegative=False)
    b, s = _vector(b, "backbone", len(x)), _vector(s, "shape", len(x))
    beta = expit(x @ theta[:k])
    if f is None:
        return np.maximum(b + beta * (s - b), 0), beta, None
    f = _vector(f, "factor", len(x))
    alpha = expit(x @ theta[k:])
    reference = s + alpha * (f - s)
    return np.maximum(b + beta * (reference - b), 0), beta, alpha


def _origin(value):
    result = np.asarray(value, dtype="datetime64[ns]")
    if result.ndim != 0 or np.isnat(result):
        raise ValueError("Origin must be a nonmissing scalar date")
    return result[()]


def _ids(series_ids):
    values = np.asarray(series_ids, dtype=object)
    if values.ndim != 1 or not len(values):
        raise ValueError("series_ids must be a nonempty vector")
    if any(v is None or str(v) in ("", "nan", "<NA>") for v in values):
        raise ValueError("Missing series ID")
    return [str(v) for v in values]


class HistoricalErrorState:
    """Forward-only development errors; freeze this state for test inference.

    Statistics count forecast-day rows. ``features`` must precede ``update`` at
    each development origin. Only explicitly supplied completed development
    labels belong in ``update``. Unknown series use the global error mean.
    """

    def __init__(self):
        self.sums = {}
        self.counts = {}
        self.global_sums = np.zeros(3, dtype=np.float64)
        self.global_count = 0
        self.latest_completed = None
        self.last_origin = None
        self.frozen = False

    def _check_origin(self, origin):
        origin = _origin(origin)
        if self.latest_completed is not None and self.latest_completed >= origin:
            raise ValueError("Historical target overlaps current forecast origin")
        return origin

    def features(self, series_ids, origin):
        """Return shrunk B/S/F errors and log1p(support), shape ``(n_rows, 4)``."""
        self._check_origin(origin)
        ids = _ids(series_ids)
        prior = self.global_sums / self.global_count if self.global_count else np.zeros(3)
        sums = np.asarray([self.sums.get(key, np.zeros(3)) for key in ids])
        counts = np.asarray([self.counts.get(key, 0) for key in ids], dtype=np.float64)
        errors = (sums + SHRINK_ROWS * prior) / (counts[:, None] + SHRINK_ROWS)
        return np.column_stack([np.clip(errors, 0, 10), np.log1p(counts)])

    def update(self, series_ids, origin, target_dates, actual, parts, history_mean):
        """Accumulate one origin's completed development forecast-day rows."""
        if self.frozen:
            raise RuntimeError("Historical error state is frozen")
        origin = self._check_origin(origin)
        if self.last_origin is not None and origin <= self.last_origin:
            raise ValueError("Development origins must advance strictly")
        ids = _ids(series_ids)
        n = len(ids)
        dates = np.asarray(target_dates, dtype="datetime64[ns]")
        if dates.shape != (n,) or np.isnat(dates).any():
            raise ValueError("Invalid target dates")
        if ((dates < origin) | (dates >= origin + np.timedelta64(7, "D"))).any():
            raise ValueError("Targets must be within the seven-day forecast horizon")
        if len(set(zip(ids, dates.astype(np.int64).tolist()))) != n:
            raise ValueError("Duplicate series and target-date rows")
        actual = _vector(actual, "actual", n)
        history_mean = _vector(history_mean, "history_mean", n)
        parts = _features(parts, n)
        if parts.shape[1] != 3 or (parts < 0).any():
            raise ValueError("parts must contain three nonnegative B/S/F columns")
        errors = np.clip(np.abs(actual[:, None] - parts) /
                         np.maximum(history_mean[:, None], 1), 0, 10)
        # Origin-local bincount matches the original vectorized accumulation.
        keys = list(dict.fromkeys(ids))
        mapping = {key: j for j, key in enumerate(keys)}
        codes = np.asarray([mapping[key] for key in ids])
        increments = np.column_stack([np.bincount(codes, weights=errors[:, k], minlength=len(keys))
                                      for k in range(3)])
        counts = np.bincount(codes, minlength=len(keys))
        for j, key in enumerate(keys):
            self.sums[key] = self.sums.get(key, np.zeros(3)) + increments[j]
            self.counts[key] = self.counts.get(key, 0) + int(counts[j])
        self.global_sums += errors.sum(axis=0)
        self.global_count += n
        self.latest_completed, self.last_origin = dates.max(), origin

    def freeze(self):
        """Forbid subsequent updates and return this state."""
        self.frozen = True
        return self

    def to_dict(self):
        return {"version": 1, "shrink_rows": SHRINK_ROWS, "frozen": self.frozen,
                "series_ids": list(self.sums), "sums": [self.sums[k].tolist() for k in self.sums],
                "counts": [self.counts[k] for k in self.sums],
                "global_sums": self.global_sums.tolist(), "global_count": self.global_count,
                "latest_completed": None if self.latest_completed is None else str(self.latest_completed),
                "last_origin": None if self.last_origin is None else str(self.last_origin)}

    @classmethod
    def from_dict(cls, record):
        if record.get("version") != 1 or record.get("shrink_rows") != SHRINK_ROWS:
            raise ValueError("Unsupported historical-state format")
        state = cls()
        keys = [str(v) for v in record["series_ids"]]
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate saved series IDs")
        sums = np.asarray(record["sums"], dtype=np.float64).reshape(-1, 3)
        counts = _vector(record["counts"], "counts", len(keys))
        if len(sums) != len(keys) or not np.isfinite(sums).all() or (sums < 0).any():
            raise ValueError("Invalid saved error sums")
        if not np.equal(counts, np.floor(counts)).all():
            raise ValueError("Counts must be integer forecast-day counts")
        state.sums = dict(zip(keys, sums))
        state.counts = dict(zip(keys, counts.astype(np.int64).tolist()))
        state.global_sums = _vector(record["global_sums"], "global_sums", 3)
        count = record["global_count"]
        if not isinstance(count, int) or count < 0 or count != int(counts.sum()):
            raise ValueError("Saved global count differs from series support")
        state.global_count = count
        if not np.allclose(sums.sum(axis=0), state.global_sums, rtol=1e-10, atol=1e-10):
            raise ValueError("Saved global error sums differ from per-series sums")
        state.latest_completed = None if record["latest_completed"] is None else _origin(record["latest_completed"])
        state.last_origin = None if record["last_origin"] is None else _origin(record["last_origin"])
        if count == 0 and (state.latest_completed is not None or state.last_origin is not None):
            raise ValueError("Saved error support and dates are inconsistent")
        if count > 0 and (state.latest_completed is None or state.last_origin is None):
            raise ValueError("Saved error support and dates are inconsistent")
        state.frozen = bool(record["frozen"])
        return state


def historical_features(frame, key, parts, train):
    """Compatibility adapter for the original DataFrame forward-history API."""
    ids = _ids(frame[key])
    parts = _features(parts, len(ids))
    if parts.shape[1] != 3 or (parts < 0).any():
        raise ValueError("parts must contain three nonnegative candidates")
    train = np.asarray(train, dtype=bool)
    if train.shape != (len(ids),):
        raise ValueError("Invalid training mask")
    origins = np.asarray(frame["origin"], dtype="datetime64[ns]")
    if np.isnat(origins).any():
        raise ValueError("Missing origins")
    state = HistoricalErrorState()
    output = np.empty((len(ids), 4), dtype=np.float64)
    series = list(dict.fromkeys(ids))
    mapping = {value: j for j, value in enumerate(series)}
    codes = np.asarray([mapping[v] for v in ids])
    for origin in np.unique(origins):
        rows = np.flatnonzero(origins == origin)
        row_ids = np.asarray(ids)[rows]
        output[rows] = state.features(row_ids, origin)
        if train[rows].all():
            state.update(row_ids, origin, np.asarray(frame["target_date"])[rows],
                         np.asarray(frame["actual"])[rows], parts[rows],
                         np.asarray(frame["history_mean"])[rows])
        elif train[rows].any():
            raise ValueError("Mixed development and evaluation split in one origin")
    supports = np.asarray([state.counts.get(ids[i], 0) for i in np.flatnonzero(~train)])
    info = {"series_count": len(series), "completed_train_rows": state.global_count,
            "late_support_quantiles": np.quantile(supports, [0, .1, .5, .9, 1]).tolist()
            if len(supports) else [], "state": state.freeze().to_dict()}
    return output, codes, info


def base_features(frame, dataset):
    """Build benchmark features, before historical errors and standardization.

    ``frame`` may be a DataFrame or mapping of row-aligned arrays. Returns
    ``(matrix, names, factor_feature_names)``. Bank sizes use log1p, matching
    the released experimental implementations on all four datasets.
    """
    dataset = dataset.lower()
    col = lambda name: np.asarray(frame[name], dtype=np.float64)
    mean, std = col("history_mean"), col("history_std")
    scale = np.maximum(mean, 1)
    names = ["log_history_mean", "history_cv", "history_zero_fraction"]
    values = [np.log1p(mean), np.clip(std / scale, 0, 10), col("history_zero_fraction")]
    if dataset == "favorita":
        extra = ["log_promo_distance", "log_holiday_distance", "log_promo_bank",
                 "log_holiday_bank", "promo_bin", "holiday"]
        values += [np.log1p(col("promo_distance")), np.log1p(col("holiday_distance")),
                   np.log1p(col("promo_bank_size")), np.log1p(col("holiday_bank_size")),
                   np.minimum(col("onpromotion"), 6) / 6, col("holiday_day")]
        factors = set(extra)
    elif dataset == "m5":
        extra = ["log_bank_size", "snap", "event"]
        values += [np.log1p(col("bank_size")), col("snap"), col("event")]
        factors = {"snap", "event"}
    elif dataset in {"rossmann", "rohlik"}:
        one, two = ("promo", "school_holiday") if dataset == "rossmann" else ("discount", "holiday")
        extra = ["log_bank_size", "log_" + one + "_distance", "log_" + two + "_distance", one, two]
        values += [np.log1p(col("bank_size")), np.log1p(col(one + "_distance")),
                   np.log1p(col(two + "_distance")), col(one), col(two)]
        factors = set(extra[1:])
    else:
        raise ValueError("dataset must be favorita, m5, rossmann, or rohlik")
    names += extra
    names += ["backbone_shape_disagreement", "factor_shape_disagreement"]
    values += [np.clip(np.abs(col("backbone_forecast") - col("shape_retrieval")) / scale, 0, 10),
               np.clip(np.abs(col("factor_retrieval") - col("shape_retrieval")) / scale, 0, 10)]
    factors.add("factor_shape_disagreement")
    return _features(np.column_stack(values)), names, factors


@dataclass
class FusionModel:
    """Fitted gate parameters, development scaler, and optional frozen history."""

    parameters: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    continuous: tuple
    feature_names: tuple
    fit_record: dict = field(default_factory=dict)
    history_state: dict | None = None

    def design(self, features):
        raw = _features(features).copy()
        if raw.shape[1] != len(self.mean):
            raise ValueError("Inference feature count differs from fitted model")
        columns = list(self.continuous)
        raw[:, columns] = (raw[:, columns] - self.mean[columns]) / self.std[columns]
        return np.column_stack([np.ones(len(raw)), raw])

    def predict(self, features, backbone, shape, factor):
        """Return ``(prediction, beta, alpha)`` with the development scaler."""
        return predict_gate(self.design(features), self.parameters, backbone, shape, factor)

    def save(self, path):
        """Save JSON without pickle; history, when supplied, is frozen."""
        record = {"version": 1, "parameters": self.parameters.tolist(),
                  "mean": self.mean.tolist(), "std": self.std.tolist(),
                  "continuous": list(self.continuous), "feature_names": list(self.feature_names),
                  "fit_record": self.fit_record, "history_state": self.history_state}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        if record.get("version") != 1:
            raise ValueError("Unsupported fusion-model format")
        mean = _vector(record["mean"], "mean", nonnegative=False)
        std = _vector(record["std"], "std", len(mean))
        if (std < 1e-8).any():
            raise ValueError("Invalid fitted standard deviations")
        theta = _vector(record["parameters"], "parameters", 2 * (len(mean) + 1), nonnegative=False)
        continuous = tuple(record["continuous"])
        if (any(not isinstance(i, int) or not 0 <= i < len(mean) for i in continuous)
                or len(set(continuous)) != len(continuous)):
            raise ValueError("Invalid continuous feature indices")
        names = tuple(record["feature_names"])
        if len(names) != len(mean) or len(set(names)) != len(names):
            raise ValueError("Invalid feature names")
        history = record.get("history_state")
        if history is not None:
            state = HistoricalErrorState.from_dict(history)
            if not state.frozen:
                raise ValueError("Fitted model history must be frozen")
        return cls(theta, mean, std, continuous, names, record.get("fit_record", {}), history)


def fit_fusion(features, actual, backbone, shape, factor, *, binary_features=(),
               feature_names=None, history_state=None):
    """Standardize development features and fit a portable full FARE gate.

    ``binary_features`` contains column indices or names; those columns retain
    their 0/1 values. Raw inputs exclude the intercept. Historical error columns
    should already have been computed forward in time before this call.
    """
    raw = _features(features)
    names = tuple(feature_names) if feature_names is not None else tuple(f"x{i}" for i in range(raw.shape[1]))
    if len(names) != raw.shape[1] or len(set(names)) != len(names):
        raise ValueError("Feature names must be unique and match feature columns")
    binary = set()
    for value in binary_features:
        if isinstance(value, str):
            if value not in names:
                raise ValueError("Unknown binary feature " + value)
            value = names.index(value)
        if not isinstance(value, (int, np.integer)) or not 0 <= value < raw.shape[1]:
            raise ValueError("Invalid binary feature column")
        binary.add(int(value))
    if binary and not np.isin(raw[:, sorted(binary)], [0, 1]).all():
        raise ValueError("Binary feature columns must contain only 0 and 1")
    continuous = tuple(i for i in range(raw.shape[1]) if i not in binary)
    mean, std = raw.mean(axis=0), np.maximum(raw.std(axis=0), 1e-8)
    columns = list(continuous)
    normalized = raw.copy()
    normalized[:, columns] = (normalized[:, columns] - mean[columns]) / std[columns]
    design = np.column_stack([np.ones(len(raw)), normalized])
    theta, record = fit_gate(design, actual, backbone, shape, factor)
    history = None
    if history_state is not None:
        # Freeze an independent serialization, leaving caller-owned state alone.
        state = HistoricalErrorState.from_dict(history_state.to_dict() if
                                               isinstance(history_state, HistoricalErrorState) else history_state)
        history = state.freeze().to_dict()
    return FusionModel(theta, mean, std, continuous, names, record, history)
