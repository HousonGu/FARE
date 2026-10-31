"""Five standalone conventional baselines; never call an existing experiment.

StatsForecast supplies ETS and fixed-order weekly SARIMA.  Statistical models
are independently fitted to each origin's 56 observations.  The two supervised
models share the forecast-origin anchored features in features.py, and train on
all supplied fit origins.  No prediction path can read the forecast labels.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import gc
import json
import multiprocessing
import os
from pathlib import Path
import sys
import uuid
import warnings

import numpy as np

try:
    from .features import (FEATURE_VERSION, HORIZON, design_matrix, feature_names,
                           row_chunks, rows_for, training_batches)
except ImportError:
    from features import (FEATURE_VERSION, HORIZON, design_matrix, feature_names,
                          row_chunks, rows_for, training_batches)


def _get(config, key, default):
    return config.get(key, default) if isinstance(config, dict) else getattr(config, key, default)


def canonical_method(method: str) -> str:
    token = str(method).lower().replace("_", "").replace("-", "").replace(" ", "")
    aliases = {"seasonalnaive": "seasonal_naive", "snaive": "seasonal_naive",
               "ets": "ets", "sarima": "sarima", "ridge": "ridge",
               "xgboost": "xgboost", "xgb": "xgboost"}
    if token not in aliases:
        raise ValueError(f"Unknown conventional method: {method}")
    return aliases[token]


def candidates(method: str, config) -> list[dict]:
    """Small preregistered grids; the runner selects using fit-tail weeks only."""
    method = canonical_method(method)
    if method == "seasonal_naive":
        return [{"season_length": 7}]
    if method == "ets":
        return [{"model": model, "season_length": 7, "damped": False}
                for model in ("ANN", "AAN", "AAA")]
    if method == "sarima":
        return [{"order": list(order), "seasonal_order": list(seasonal), "season_length": 7,
                 "include_mean": order[1] == seasonal[1] == 0,
                 "include_drift": False, "method": "CSS-ML"}
                for order, seasonal in (((1, 0, 0), (0, 1, 1)),
                                         ((0, 1, 1), (0, 1, 1)),
                                         ((1, 0, 1), (1, 0, 0)))]
    if method == "ridge":
        return [{"alpha": alpha} for alpha in (1.0, 100.0, 10000.0)]
    return [{"max_depth": depth, "eta": 0.05,
             "rounds": int(_get(config, "xgb_rounds", 300)),
             "min_child_weight": 10, "subsample": 0.8, "colsample_bytree": 1.0,
             "lambda": 1.0, "alpha": 0.0}
            for depth in (3, 5, 7)]


def _json_value(value):
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def _atomic_json(path: Path, value):
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temp, path)


def _atomic_npz(path: Path, **arrays):
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temp, path)


def _atomic_booster(path: Path, booster):
    # Keep a recognized .ubj extension while replacing atomically.
    temp = path.with_name(path.stem + ".tmp.ubj")
    booster.save_model(str(temp))
    os.replace(temp, path)


@dataclass
class Predictor:
    method: str
    params: dict
    folder: Path
    config: object
    manifest: dict
    state: object = None
    last_diagnostics: dict = field(default_factory=dict)


class StatisticalFitError(RuntimeError):
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def _model_manifest(panel, method, params, origins, config, seed):
    spec = {"method": method, "params": _json_value(params),
            "dataset": str(panel.dataset), "origins": list(origins),
            "features": FEATURE_VERSION if method in ("ridge", "xgboost") else None,
            "feature_names": feature_names(panel.covariate_names) if method in ("ridge", "xgboost") else None,
            "seed": int(seed) if method == "xgboost" else None}
    if method == "ridge":
        spec["effective_params"] = {**_json_value(params), "fit_intercept": True,
            "standardize_features": "training population mean/std, ddof=0",
            "objective": "sum_squared_error_plus_alpha_l2", "sample_weight": None,
            "target_transform": None}
    elif method in ("ets", "sarima"):
        spec["effective_params"] = {**_json_value(params), "implementation": "statsforecast==2.0.3",
            "context_length": 56, "horizon": 7, "target_transform": None,
            "fit_failure_policy": "raise_no_fallback"}
    if method == "xgboost":
        spec["max_bin"] = int(_get(config, "xgb_max_bin", 128))
        spec["xgb_batch_rows"] = int(_get(config, "xgb_batch_rows",
            max(32768, int(_get(config, "batch_rows", 2048)))))
        spec["xgb_nthread"] = _threads(config)
        spec["xgb_checkpoint_rounds"] = max(1, int(_get(config, "xgb_checkpoint_rounds", 25)))
        spec["xgb_version_required"] = "3.1.3"
        spec["effective_params"] = {**_json_value(params), "objective": "reg:squarederror",
            "tree_method": "hist", "device": "cpu", "seed": int(seed),
            "seed_per_iteration": True, "base_score": 0.0, "max_bin": spec["max_bin"],
            "nthread": spec["xgb_nthread"], "sample_weight": None, "target_transform": None,
            "cache_reset_every_rounds": spec["xgb_checkpoint_rounds"],
            "resume_policy": "identical_scheduled_updater_resets_in_all_runs"}
    return spec


def fit_model(panel, method, params, origins, folder, config, seed=20261005):
    """Create/load a predictor in a new, isolated result folder.

    For statistical methods this saves the specification: their actual fitting
    occurs per entity and origin in predict_model.  For supervised methods the
    supplied origin list must contain fit-only origins (Panel enforces labels).
    Existing checkpoints are reused only when the training specification is unchanged.
    """
    method = canonical_method(method)
    origins = [str(origin) for origin in origins]
    if len(set(origins)) != len(origins):
        raise ValueError("Duplicate training origins are not allowed")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    manifest = _model_manifest(panel, method, params, origins, config, seed)
    path = folder / "predictor.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("complete"):
            return load_model(method, folder, config)
        if {key: existing.get(key) for key in manifest} != manifest:
            _atomic_json(path, manifest)
    else:
        _atomic_json(path, manifest)
    predictor = Predictor(method, dict(params), folder, config, manifest)
    if method in ("ridge", "xgboost") and not origins:
        raise ValueError("A supervised model requires at least one fit origin")
    if method == "ridge":
        _fit_ridge(predictor, panel, origins)
    elif method == "xgboost":
        _fit_xgboost(predictor, panel, origins, int(seed))
    manifest["complete"] = True
    _atomic_json(path, manifest)
    return predictor


def load_model(method, folder, config):
    """Load a completed predictor. Incomplete training resumes via fit_model."""
    method = canonical_method(method)
    folder = Path(folder)
    manifest = json.loads((folder / "predictor.json").read_text(encoding="utf-8"))
    if manifest["method"] != method or not manifest.get("complete"):
        raise ValueError("Missing, incomplete, or wrong-method predictor")
    if method in ("ridge", "xgboost") and manifest["features"] != FEATURE_VERSION:
        raise ValueError("Loaded predictor uses a different feature specification")
    predictor = Predictor(method, manifest["params"], folder, config, manifest)
    if method == "ridge":
        with np.load(folder / "ridge.npz", allow_pickle=False) as saved:
            predictor.state = {name: saved[name].copy() for name in ("coef", "intercept")}
    elif method == "xgboost":
        import xgboost as xgb
        _require_xgb_version(xgb)
        predictor.state = xgb.Booster(model_file=str(folder / "xgboost.ubj"))
        predictor.state.set_param({"nthread": _threads(config), "device": "cpu"})
    return predictor


def _fit_ridge(predictor, panel, origins):
    """Exact pooled Ridge via stable merged centered sufficient statistics.

    Minimizes sum((y - intercept - z @ beta)^2) + alpha * ||beta||^2,
    with z standardized using training features only, population variance ddof=0.
    The unpenalized intercept and zero-variance columns match ordinary Ridge.
    Memory is O(batch_rows * 7 * F + F^2), independent of training-set size.
    """
    n_features = len(feature_names(panel.covariate_names))
    count, mean_y = 0, 0.0
    mean_x = np.zeros(n_features, np.float64)
    scatter_xx = np.zeros((n_features, n_features), np.float64)
    scatter_xy = np.zeros(n_features, np.float64)
    completed = 0
    progress = predictor.folder / "ridge_progress.npz"
    if progress.exists():
        with np.load(progress, allow_pickle=False) as saved:
            count = int(saved["count"])
            mean_y = float(saved["mean_y"])
            completed = int(saved["completed"])
            mean_x = saved["mean_x"].copy()
            scatter_xx = saved["scatter_xx"].copy()
            scatter_xy = saved["scatter_xy"].copy()
    batch_rows = int(_get(predictor.config, "batch_rows", 2048))
    for origin_idx in range(completed, len(origins)):
        origin = origins[origin_idx]
        for x, y in training_batches(panel, [origin], batch_rows, dtype=np.float64):
            batch_count = len(y)
            bx, by = x.mean(axis=0), float(y.mean())
            centered_x, centered_y = x - bx, y - by
            dx, dy = bx - mean_x, by - mean_y
            new_count = count + batch_count
            merge_factor = count * batch_count / new_count
            scatter_xx += centered_x.T @ centered_x + merge_factor * np.outer(dx, dx)
            scatter_xy += centered_x.T @ centered_y + merge_factor * dx * dy
            mean_x += dx * (batch_count / new_count)
            mean_y += dy * (batch_count / new_count)
            count = new_count
        _atomic_npz(progress, count=np.int64(count), mean_y=np.float64(mean_y),
                    mean_x=mean_x, scatter_xx=scatter_xx, scatter_xy=scatter_xy,
                    completed=np.int64(origin_idx + 1))
    if count == 0:
        raise ValueError("No legal training rows for Ridge")
    scale = np.sqrt(np.maximum(np.diag(scatter_xx) / count, 0))
    # Constant calendar / covariate columns cannot contribute after centering.
    constant = scale <= np.finfo(np.float64).eps * np.maximum(1.0, np.abs(mean_x))
    scale[constant] = 1.0
    standardized_xx = scatter_xx / scale[:, None] / scale[None, :]
    standardized_xy = scatter_xy / scale
    standardized_xx[constant, :] = 0
    standardized_xx[:, constant] = 0
    standardized_xy[constant] = 0
    alpha = float(predictor.params["alpha"])
    if alpha <= 0:
        raise ValueError("Ridge alpha must be positive")
    standardized_xx.flat[::n_features + 1] += alpha
    beta = np.linalg.solve(standardized_xx, standardized_xy)
    coef = beta / scale
    intercept = mean_y - mean_x @ coef
    if not np.isfinite(coef).all() or not np.isfinite(intercept):
        raise FloatingPointError("Non-finite Ridge solution")
    _atomic_npz(predictor.folder / "ridge.npz", coef=coef,
                intercept=np.float64(intercept), mean_x=mean_x, scale=scale,
                n_training_samples=np.int64(count))
    predictor.state = {"coef": coef, "intercept": np.float64(intercept)}
    predictor.manifest["n_training_samples"] = count


def _threads(config):
    value = int(_get(config, "xgb_nthread", _get(config, "n_jobs", 1)))
    return (os.cpu_count() or 1) if value == -1 else max(value, 1)


def _require_xgb_version(xgb):
    if xgb.__version__ != "3.1.3":
        raise RuntimeError(f"This reproducible package requires xgboost==3.1.3, got {xgb.__version__}")


_BUILD = {}


def _limit_worker_threads():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMBA_NUM_THREADS"] = "1"
    try:
        import ctypes
        ctypes.CDLL("libgomp.so.1").omp_set_num_threads(1)
    except OSError:
        pass


def _fill_training_slice(task):
    """One forked worker writes its entity slice into the shared feature files."""
    _limit_worker_threads()
    origin, rows, out = task
    rows = np.asarray(rows, dtype=np.int64)
    panel = _BUILD["panel"]
    features = design_matrix(panel, origin, rows=rows, dtype=np.float32)
    labels = np.asarray(panel.training_labels(origin, rows=rows), dtype=np.float32).reshape(-1)
    count = int(labels.shape[0])
    if features.shape != (count, _BUILD["n_features"]) or not np.isfinite(features).all():
        raise ValueError(f"Bad training block at {origin}")
    _BUILD["X"][out:out + count] = features
    _BUILD["y"][out:out + count] = labels
    return count


def _matrix_cache(folder):
    root = folder
    for parent in (folder, *folder.parents):
        if parent.name == "xgboost":
            root = parent
            break
    path = root / "_xy_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _training_matrix(panel, origins, folder, threads):
    """Build the fit matrix once, in parallel, and reuse it for every trial."""
    names = feature_names(panel.covariate_names)
    n_features = len(names)
    tasks = []
    cursor = 0
    for origin in origins:
        indices = np.asarray(panel.rows(origin), dtype=np.int64)
        for start in range(0, len(indices), 8192):
            block = indices[start:start + 8192]
            tasks.append((str(origin), block, cursor + start * HORIZON))
        cursor += len(indices) * HORIZON
    n_rows = cursor
    if n_rows == 0:
        raise ValueError("No legal training rows for XGBoost")
    cache = _matrix_cache(folder)
    key = f"{FEATURE_VERSION}_{origins[0]}_{origins[-1]}_{len(origins)}".replace("/", "_")
    meta_path = cache / f"{key}.json"
    x_path = cache / f"{key}.X.f32"
    y_path = cache / f"{key}.y.f32"
    meta = {"rows": n_rows, "features": n_features, "feature_version": FEATURE_VERSION,
            "origins": list(map(str, origins))}
    if (meta_path.exists() and x_path.exists() and y_path.exists()
            and json.loads(meta_path.read_text(encoding="utf-8")) == meta):
        print(f"[xgb] reuse cached matrix {n_rows} x {n_features}", flush=True)
        X = np.memmap(x_path, dtype=np.float32, mode="r", shape=(n_rows, n_features))
        y = np.memmap(y_path, dtype=np.float32, mode="r", shape=(n_rows,))
        return X, y
    print(f"[xgb] building {n_rows} x {n_features} on {threads} cores "
          f"({len(tasks)} blocks)", flush=True)
    X = np.memmap(x_path, dtype=np.float32, mode="w+", shape=(n_rows, n_features))
    y = np.memmap(y_path, dtype=np.float32, mode="w+", shape=(n_rows,))
    _BUILD.clear()
    _BUILD.update(panel=panel, X=X, y=y, n_features=n_features)
    workers = max(1, min(int(threads), len(tasks)))
    done = 0
    written = 0
    if sys.platform.startswith("linux") and workers > 1:
        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(processes=workers) as pool:
            for count in pool.imap_unordered(_fill_training_slice, tasks, chunksize=1):
                done += 1
                written += int(count)
                if done == len(tasks) or done % 25 == 0:
                    print(f"[xgb] feature blocks {done}/{len(tasks)}", flush=True)
    else:
        for task in tasks:
            written += int(_fill_training_slice(task))
            done += 1
            if done == len(tasks) or done % 25 == 0:
                print(f"[xgb] feature blocks {done}/{len(tasks)}", flush=True)
    if written != n_rows:
        raise ValueError(f"Training matrix row count mismatch: {written} != {n_rows}")
    X.flush()
    y.flush()
    _atomic_json(meta_path, meta)
    print("[xgb] feature matrix ready", flush=True)
    return X, y


def _fit_xgboost(predictor, panel, origins, seed):
    """CPU histogram booster. Features are built once on every core, then kept in memory.

    The public low-level update API receives the global boosting iteration even
    after checkpoint loading. seed_per_iteration plus scheduled updater resets
    (also in uninterrupted runs) preserve the subsampling random sequence across
    restarts.
    """
    import xgboost as xgb
    _require_xgb_version(xgb)
    config = predictor.config
    threads = _threads(config)
    max_bin = predictor.manifest["max_bin"]
    try:
        import ctypes
        ctypes.CDLL("libgomp.so.1").omp_set_num_threads(threads)
    except OSError:
        pass
    os.environ["OMP_NUM_THREADS"] = str(threads)
    X, y = _training_matrix(panel, origins, predictor.folder, threads)
    matrix, booster = None, None
    try:
        matrix = xgb.QuantileDMatrix(X, label=y, max_bin=max_bin, nthread=threads)
        predictor.manifest["n_training_samples"] = int(matrix.num_row())
        if matrix.num_row() != len(y):
            raise ValueError("Quantile matrix row count differs from the training matrix")
        params = {key: value for key, value in predictor.params.items() if key != "rounds"}
        params.update({"objective": "reg:squarederror", "tree_method": "hist", "device": "cpu",
                       "nthread": threads, "max_bin": max_bin, "seed": seed,
                       "seed_per_iteration": True, "base_score": 0.0, "verbosity": 1})
        checkpoint = predictor.folder / "xgboost_checkpoint.ubj"
        progress = predictor.folder / "xgboost_progress.json"
        booster = xgb.Booster(params=params, cache=[matrix],
                              model_file=str(checkpoint) if checkpoint.exists() and progress.exists() else None)
        booster.set_param(params)
        rounds = int(predictor.params["rounds"])
        if rounds < 1:
            raise ValueError("XGBoost requires a positive number of boosting rounds")
        interval = predictor.manifest["xgb_checkpoint_rounds"]
        done = booster.num_boosted_rounds()
        if done > rounds:
            raise ValueError("XGBoost checkpoint has more rounds than the specification")
        print(f"[xgb] boosting {done}/{rounds} on {threads} cores", flush=True)
        for iteration in range(done, rounds):
            booster.update(matrix, iteration=iteration)
            if (iteration + 1) % interval == 0 or iteration + 1 == rounds:
                _atomic_booster(checkpoint, booster)
                _atomic_json(predictor.folder / "xgboost_progress.json",
                             {"completed_rounds": iteration + 1, "total_rounds": rounds,
                              "n_training_samples": int(matrix.num_row())})
                print(f"[xgb] checkpoint {iteration + 1}/{rounds}", flush=True)
                # Reset updater/prediction caches at every scheduled boundary,
                # including uninterrupted runs. Fresh CPU hist updaters consume
                # a random value for their ColumnSampler even when all columns
                # are used; enforcing the same reset boundaries makes row
                # subsampling identical after loading a saved checkpoint.
                booster.reset()
        _atomic_booster(predictor.folder / "xgboost.ubj", booster)
        predictor.state = booster.reset()
    finally:
        if booster is not None:
            booster.reset()
        del matrix, X, y
        gc.collect()


def _statistical_chunk(method, params, indices, context):
    """Worker receives a small history array, never the full panel."""
    from statsforecast.models import ARIMA, AutoETS
    output = np.full((len(indices), HORIZON), np.nan, dtype=np.float64)
    diagnostics = {"analytical_constant_rows": 0, "fit_failures": 0,
                   "warning_count": 0, "warning_examples": [], "failure_examples": []}
    for position, (absolute_row, history) in enumerate(zip(indices, context)):
        try:
            if not np.isfinite(history).all():
                raise ValueError("Non-finite 56-day context")
            if np.all(history == history[0]):
                output[position] = history[0]
                diagnostics["analytical_constant_rows"] += 1
                continue
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                if method == "ets":
                    estimator = AutoETS(season_length=int(params["season_length"]),
                                        model=params["model"], damped=bool(params["damped"]))
                else:
                    estimator = ARIMA(order=tuple(params["order"]),
                                      seasonal_order=tuple(params["seasonal_order"]),
                                      season_length=int(params["season_length"]),
                                      include_mean=bool(params["include_mean"]),
                                      include_drift=bool(params["include_drift"]),
                                      method=params["method"])
                prediction = np.asarray(estimator.forecast(y=history, h=HORIZON)["mean"],
                                        dtype=np.float64)
            diagnostics["warning_count"] += len(caught)
            for warning in caught:
                if len(diagnostics["warning_examples"]) < 20:
                    diagnostics["warning_examples"].append({"row": int(absolute_row),
                        "category": warning.category.__name__, "message": str(warning.message)})
            if prediction.shape != (HORIZON,) or not np.isfinite(prediction).all():
                raise ValueError("Fitted model returned invalid predictions")
            output[position] = prediction
        except Exception as exc:
            diagnostics["fit_failures"] += 1
            if len(diagnostics["failure_examples"]) < 50:
                diagnostics["failure_examples"].append({"row": int(absolute_row),
                    "exception": type(exc).__name__, "message": str(exc)})
    return output, diagnostics


def _predict_statistical(predictor, panel, origin, rows):
    from joblib import Parallel, delayed, parallel_config
    n_jobs = int(_get(predictor.config, "n_jobs", 1))
    chunk_size = max(1, int(_get(predictor.config, "stats_chunk_rows", 64)))
    chunks = row_chunks(panel, origin, chunk_size, rows)
    tasks = (delayed(_statistical_chunk)(predictor.method, predictor.params, indices,
              np.ascontiguousarray(panel.context(origin, rows=indices), dtype=np.float64))
             for indices in chunks)
    diagnostics = {"n_series": len(rows), "analytical_constant_rows": 0,
                   "fit_failures": 0, "fallbacks": 0, "warning_count": 0,
                   "warning_examples": [], "failure_examples": []}
    result = np.empty((len(rows), HORIZON), dtype=np.float64)
    position = 0
    # Process workers avoid Python/GIL bottlenecks. They receive only each chunk
    # and each process limits BLAS to one thread to avoid nested oversubscription.
    with parallel_config(backend="loky", inner_max_num_threads=1):
        completed = Parallel(n_jobs=n_jobs, pre_dispatch="2*n_jobs", return_as="generator")(tasks)
        for output, diag in completed:
            result[position:position + len(output)] = output
            position += len(output)
            for key in ("analytical_constant_rows", "fit_failures", "warning_count"):
                diagnostics[key] += diag[key]
            for key, limit in (("warning_examples", 50), ("failure_examples", 100)):
                diagnostics[key].extend(diag[key][:max(0, limit - len(diagnostics[key]))])
    predictor.last_diagnostics = diagnostics
    _atomic_json(predictor.folder / f"diagnostics_{origin}.json", diagnostics)
    if diagnostics["fit_failures"]:
        raise StatisticalFitError(f"{predictor.method} failed for {diagnostics['fit_failures']} "
                                  f"series at {origin}; no fallback predictions were used", diagnostics)
    return result


def predict_model(predictor, panel, origin, rows=None):
    """Predict nonnegative N x 7 values; invalid fits stop instead of dropping rows."""
    indices = rows_for(panel, origin, rows)
    predictor.last_diagnostics = {"n_series": len(indices), "fit_failures": 0, "fallbacks": 0}
    if predictor.method == "seasonal_naive":
        context = np.asarray(panel.context(origin, rows=indices), dtype=np.float64)
        output = context[:, -HORIZON:].copy()
    elif predictor.method in ("ets", "sarima"):
        output = _predict_statistical(predictor, panel, origin, indices)
    else:
        if feature_names(panel.covariate_names) != predictor.manifest["feature_names"]:
            raise ValueError("Prediction covariate schema differs from fitted model")
        output = np.empty((len(indices), HORIZON), dtype=np.float64)
        batch_rows = int(_get(predictor.config, "batch_rows", 2048))
        position = 0
        for chunk in row_chunks(panel, origin, batch_rows, indices):
            dtype = np.float64 if predictor.method == "ridge" else np.float32
            features = design_matrix(panel, origin, chunk, dtype=dtype)
            if predictor.method == "ridge":
                values = features @ predictor.state["coef"] + predictor.state["intercept"]
            else:
                values = predictor.state.inplace_predict(features)
            output[position:position + len(chunk)] = values.reshape(len(chunk), HORIZON)
            position += len(chunk)
    if output.shape != (len(indices), HORIZON) or not np.isfinite(output).all():
        raise FloatingPointError(f"Invalid {predictor.method} prediction at {origin}")
    predictor.last_diagnostics["negative_predictions_clipped"] = int((output < 0).sum())
    return np.maximum(output, 0)
