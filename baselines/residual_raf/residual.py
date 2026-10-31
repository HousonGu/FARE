"""Frozen-backbone LightGBM residual correction; test labels are never accessed.

The provider must return aligned nonnegative N x 7 Chronos-2 median forecasts
and expose the input folder as ``root``.
Only ``panel.residual_labels`` can supply labels; it must accept dev origins only.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import gc
import json
import math
import os
from pathlib import Path

import numpy as np

try:
    from .features import (FEATURE_VERSION, HORIZON, design_matrix, feature_names,
                           row_chunks, rows_for)
except ImportError:
    from features import (FEATURE_VERSION, HORIZON, design_matrix, feature_names,
                          row_chunks, rows_for)


METHOD_VERSION = "frozen-chronos2-lightgbm-residual-dev-l1-v1"
LIGHTGBM_VERSION = "4.6.0"
FORMAL_SEEDS = (20261005, 20261006, 20261007)
LEAF_GRID = (15, 31, 63)


def _canonical(value):
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    sort_keys=True, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def split_development(panel, tune_origins=None):
    """Chronological inner split, with no overlapping target week at boundary."""
    origins = list(panel.schedule["dev"])
    if len(origins) < 3 or origins != sorted(set(origins)):
        raise ValueError("Residual fitting requires at least three ordered dev origins")
    for left, right in zip(origins, origins[1:]):
        if date.fromisoformat(left) + timedelta(days=HORIZON - 1) >= date.fromisoformat(right):
            raise ValueError("Development target windows overlap")
    test = list(panel.schedule.get("test", []))
    if test and date.fromisoformat(origins[-1]) + timedelta(days=HORIZON - 1) >= date.fromisoformat(test[0]):
        raise ValueError("Development targets overlap the formal test period")
    if tune_origins is None:
        tune_origins = origins[-max(1, math.ceil(len(origins) * 0.2)):]
    tune_origins = list(tune_origins)
    if not tune_origins or len(tune_origins) >= len(origins) or origins[-len(tune_origins):] != tune_origins:
        raise ValueError("Residual tuning origins must be a nonempty strict dev suffix")
    return origins[:-len(tune_origins)], tune_origins


def residual_targets(actual, backbone):
    actual, backbone = np.asarray(actual, np.float64), np.asarray(backbone, np.float64)
    if actual.ndim != 2 or actual.shape[1] != HORIZON or actual.shape != backbone.shape:
        raise ValueError("Expected aligned N x 7 targets and backbone predictions")
    if not np.isfinite(actual).all() or not np.isfinite(backbone).all() or (actual < 0).any() or (backbone < 0).any():
        raise ValueError("Targets and frozen backbone predictions must be finite and nonnegative")
    return actual - backbone


def add_correction(backbone, residual):
    backbone, residual = np.asarray(backbone, np.float64), np.asarray(residual, np.float64)
    if backbone.ndim != 2 or backbone.shape[1] != HORIZON or backbone.shape != residual.shape:
        raise ValueError("Correction and frozen backbone must be aligned N x 7 arrays")
    if not np.isfinite(backbone).all() or not np.isfinite(residual).all() or (backbone < 0).any():
        raise ValueError("Invalid frozen backbone or residual correction")
    output = np.maximum(backbone + residual, 0)
    if not np.isfinite(output).all():
        raise FloatingPointError("Non-finite corrected predictions")
    return output


def _source(panel, provider):
    root = getattr(provider, "root", None)
    if root is None:
        raise ValueError("Frozen backbone provider requires its input folder")
    record = {"dataset": str(panel.dataset), "backbone": str(root), "feature_version": FEATURE_VERSION,
              "method_version": METHOD_VERSION,
              "feature_names": feature_names(panel.covariate_names) + ["frozen_backbone_prediction"]}
    folder = getattr(panel, "folder", None)
    if folder is not None and (Path(folder) / "complete.json").is_file():
        record["prepared_panel"] = str(folder)
    return record


def _base(provider, origin, rows):
    values = np.asarray(provider(origin, rows), dtype=np.float64)
    if values.shape != (len(rows), HORIZON) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"Invalid or unaligned frozen backbone predictions at {origin}")
    return values


def _features(panel, provider, origin, rows):
    original = design_matrix(panel, origin, rows=rows, dtype=np.float32)
    base = _base(provider, origin, rows)
    output = np.empty((len(original), original.shape[1] + 1), np.float32)
    output[:, :-1] = original
    output[:, -1] = base.reshape(-1)
    if not np.isfinite(output).all():
        raise FloatingPointError("Float32 residual features overflowed")
    return output, base


def _verify_artifacts(folder, record):
    for relative in record["artifacts"]:
        path = Path(folder) / relative
        if not path.is_file():
            raise ValueError("Missing residual artifact: " + str(path))


def _training_matrix(panel, provider, origins, source, cache_dir, batch_rows):
    """One disk-backed dev matrix shared by all seeds and candidate fits.

    Constructing a LightGBM Dataset creates its binned representation in RAM.
    mmap bounds feature construction memory; it does not make boosting external
    memory. No row/origin subsampling or float64 full matrix is used.
    """
    spec = _canonical({"source": source, "origins": list(origins), "dtype": "float32"})
    folder = Path(cache_dir) / f"{source['dataset']}_{origins[0]}_{origins[-1]}_{len(origins)}"
    folder.mkdir(parents=True, exist_ok=True)
    marker = folder / "complete.json"
    names = source["feature_names"]
    counts = [len(panel.rows(origin)) * HORIZON for origin in origins]
    n = sum(counts)
    if n <= 0:
        raise ValueError("Empty residual training matrix")
    if marker.is_file():
        complete = json.loads(marker.read_text(encoding="utf-8"))
        if complete["spec"] != spec or complete["shape"] != [n, len(names)]:
            raise ValueError("Residual feature cache specification mismatch")
        _verify_artifacts(folder, complete)
    else:
        x = np.memmap(folder / "X.f32", mode="w+", dtype=np.float32, shape=(n, len(names)))
        y = np.memmap(folder / "residual.f32", mode="w+", dtype=np.float32, shape=(n,))
        cursor = 0
        for origin in origins:
            if origin not in panel.schedule["dev"]:
                raise ValueError("Residual fitting labels can only come from development origins")
            for rows in row_chunks(panel, origin, batch_rows):
                features, base = _features(panel, provider, origin, rows)
                actual = np.asarray(panel.residual_labels(origin, rows=rows), np.float64)
                residual = residual_targets(actual, base).reshape(-1)
                stop = cursor + len(residual)
                x[cursor:stop] = features
                y[cursor:stop] = residual
                if not np.isfinite(y[cursor:stop]).all():
                    raise FloatingPointError("Float32 residual targets overflowed")
                cursor = stop
            print(f"[residual] cached {origin}: {cursor}/{n} rows", flush=True)
        if cursor != n:
            raise ValueError("Residual feature matrix row count mismatch")
        x.flush()
        y.flush()
        del x, y
        complete = {"shape": [n, len(names)], "spec": spec,
                    "rows_per_origin": dict(zip(origins, counts)),
                    "artifacts": ["X.f32", "residual.f32"]}
        _write_json(marker, complete)
    x = np.memmap(folder / "X.f32", mode="r", dtype=np.float32, shape=(n, len(names)))
    y = np.memmap(folder / "residual.f32", mode="r", dtype=np.float32, shape=(n,))
    return x, y, complete


def _effective_params(seed, threads, leaves):
    return {"objective": "regression_l1", "metric": "l1", "boosting_type": "gbdt",
            "learning_rate": 0.05, "num_leaves": int(leaves), "max_depth": -1,
            "min_data_in_leaf": 100, "lambda_l1": 0.0, "lambda_l2": 1.0,
            "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1,
            "max_bin": 63, "feature_pre_filter": False, "verbosity": -1,
            "seed": int(seed), "data_random_seed": int(seed), "bagging_seed": int(seed),
            "feature_fraction_seed": int(seed), "deterministic": True,
            "force_col_wise": True, "device_type": "cpu", "num_threads": int(threads)}


def _lgb():
    import lightgbm as lgb
    if lgb.__version__ != LIGHTGBM_VERSION:
        raise RuntimeError(f"Require lightgbm=={LIGHTGBM_VERSION}; installed {lgb.__version__}")
    return lgb


def _fit_cached(lgb, dataset, folder, params, rounds, training_spec):
    """Checkpoint each completed trial/refit; interrupted trial restarts cleanly."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    spec = _canonical({"params": params, "rounds": int(rounds), "training": training_spec,
                       "lightgbm_version": LIGHTGBM_VERSION})
    marker = folder / "complete.json"
    if marker.is_file():
        record = json.loads(marker.read_text(encoding="utf-8"))
        if record["spec"] != spec:
            raise ValueError("Residual model task changed; use a new output directory")
        _verify_artifacts(folder, record)
        return lgb.Booster(model_file=str(folder / "model.txt")), record
    booster = lgb.train(params, dataset, num_boost_round=rounds,
                        callbacks=[lgb.log_evaluation(period=0)])
    temporary = folder / "model.txt.tmp"
    booster.save_model(str(temporary))
    os.replace(temporary, folder / "model.txt")
    record = {"spec": spec, "complete": True,
              "actual_tree_count": int(booster.num_trees()),
              "artifacts": ["model.txt"]}
    _write_json(marker, record)
    booster.free_dataset()
    return booster, record


def _evaluate(booster, panel, provider, origins, batch_rows, threads):
    total, error, count = 0.0, 0.0, 0
    for origin in origins:
        if origin not in panel.schedule["dev"]:
            raise ValueError("Residual selection may read only development labels")
        for rows in row_chunks(panel, origin, batch_rows):
            features, base = _features(panel, provider, origin, rows)
            actual = np.asarray(panel.residual_labels(origin, rows=rows), np.float64)
            residual_targets(actual, base)
            delta = np.asarray(booster.predict(features, num_threads=threads), np.float64).reshape(base.shape)
            prediction = add_correction(base, delta)
            total += float(actual.sum(dtype=np.float64))
            error += float(np.abs(actual - prediction).sum(dtype=np.float64))
            count += int(actual.size)
    if count == 0 or total <= 0:
        raise ValueError("Selection requires a nonempty, positive-sum development suffix")
    return {"rows": count, "actual_sum": total, "absolute_error_sum": error,
            "wape": error / total, "mae": error / count}


@dataclass
class ResidualPredictor:
    booster: object
    manifest: dict
    folder: Path
    threads: int
    batch_rows: int


def train_residual(panel, base_provider, out_dir, seed, threads=8, batch_rows=2048,
                   rounds=300, tune_origins=None, cache_dir=None):
    """Select leaves on a chronological dev suffix and refit on all dev weeks.

    Formal protocol uses 300 rounds and 3 leaf candidates, then another 300-round
    refit, for each of three seeds. ``rounds`` is recorded in every task
    specification.
    """
    if int(threads) < 1 or int(batch_rows) < 1 or int(rounds) < 1:
        raise ValueError("threads, batch_rows and rounds must be positive")
    threads, batch_rows, rounds = int(threads), int(batch_rows), int(rounds)
    train_origins, tune = split_development(panel, tune_origins)
    origins = train_origins + tune
    source = _source(panel, base_provider)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = {"source": source, "seed": int(seed), "threads": threads, "rounds": rounds,
            "leaf_grid": list(LEAF_GRID), "inner_train_origins": train_origins,
            "inner_tune_origins": tune, "refit_origins": origins,
            "selection_metric": "pooled WAPE of clip(base + residual, 0)",
            "training_target": "actual - frozen nonnegative median backbone forecast",
            "target_transform": None, "test_label_access": False}
    spec = _canonical(spec)
    manifest_path = out_dir / "manifest.json"
    lgb = _lgb()
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["spec"] != spec:
            raise ValueError("Residual experiment specification changed; use a new output directory")
        if manifest.get("complete"):
            _verify_artifacts(out_dir, manifest)
            booster = lgb.Booster(model_file=str(out_dir / "refit" / "model.txt"))
            return ResidualPredictor(booster, manifest, out_dir, threads, batch_rows)
    _write_json(manifest_path, {"spec": spec, "complete": False})
    cache_dir = Path(cache_dir) if cache_dir is not None else out_dir.parent / "_residual_matrix"
    x, y, matrix = _training_matrix(panel, base_provider, origins, source, cache_dir, batch_rows)
    n_train = sum(matrix["rows_per_origin"][origin] for origin in train_origins)
    if n_train <= 0 or n_train >= len(y):
        raise ValueError("Empty chronological inner train or tune subset")
    dataset_params = {"max_bin": 63, "feature_pre_filter": False,
                      "data_random_seed": int(seed), "num_threads": threads}
    inner = lgb.Dataset(x[:n_train], label=y[:n_train],
                        feature_name=source["feature_names"], params=dataset_params,
                        free_raw_data=True)
    trials = []
    try:
        for leaves in LEAF_GRID:
            params = _effective_params(seed, threads, leaves)
            booster, record = _fit_cached(lgb, inner, out_dir / f"trial_{leaves}", params,
                rounds, {"matrix_shape": matrix["shape"], "origins": train_origins,
                         "n_training_samples": n_train, "matrix_prefix_rows": n_train})
            score = _evaluate(booster, panel, base_provider, tune, batch_rows, threads)
            trials.append({"num_leaves": leaves, "params": params, "metric": score})
            print(f"[residual] leaves={leaves} tune WAPE={score['wape']:.8f}", flush=True)
            del booster
    finally:
        del inner
        gc.collect()
    best = min(trials, key=lambda entry: (entry["metric"]["wape"], entry["num_leaves"]))
    selection = {"criterion": "minimum pooled clipped-output WAPE; ties choose fewer leaves",
                 "training_origins": train_origins, "tuning_origins": tune,
                 "trials": trials, "selected_params": best["params"]}
    _write_json(out_dir / "selection.json", selection)
    full = lgb.Dataset(x, label=y, feature_name=source["feature_names"],
                       params=dataset_params, free_raw_data=True)
    try:
        booster, record = _fit_cached(lgb, full, out_dir / "refit", best["params"], rounds,
            {"matrix_shape": matrix["shape"], "origins": origins,
             "n_training_samples": len(y), "selected_params": best["params"]})
    finally:
        del full, x, y
        gc.collect()
    manifest = {"spec": spec, "complete": True,
                "lightgbm_version": LIGHTGBM_VERSION, "selected_params": best["params"],
                "matrix": matrix, "selected_tune_metric": best["metric"],
                "artifacts": ["refit/model.txt", "refit/complete.json", "selection.json"]}
    _write_json(manifest_path, manifest)
    return ResidualPredictor(booster, manifest, out_dir, threads, batch_rows)


def predict_residual(predictor, panel, base_provider, origin, rows=None):
    """Return final nonnegative forecasts, preserving requested entity order."""
    if _source(panel, base_provider) != predictor.manifest["spec"]["source"]:
        raise ValueError("Residual prediction source differs from the fitted experiment")
    indices = rows_for(panel, origin, rows)
    output = np.empty((len(indices), HORIZON), dtype=np.float64)
    cursor = 0
    for chunk in row_chunks(panel, origin, predictor.batch_rows, rows=indices):
        features, base = _features(panel, base_provider, origin, chunk)
        correction = np.asarray(predictor.booster.predict(features,
            num_threads=predictor.threads), dtype=np.float64).reshape(base.shape)
        output[cursor:cursor + len(chunk)] = add_correction(base, correction)
        cursor += len(chunk)
    if cursor != len(indices) or not np.isfinite(output).all():
        raise FloatingPointError("Incomplete or invalid residual predictions")
    return output
