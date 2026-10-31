"""Shared I/O for runs that read the stored Rossmann/Rohlik Chronos-2 candidates.

The source is a completed ``exp.rossmann``/``exp.rohlik`` run with ``references/``
and ``chronos/`` for every development and test origin. Predictions stay
non-negative; overall WAPE is total absolute error over total sales.
"""
from __future__ import annotations

import importlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["origin", "series_id", "target_date"]
EXPECTED = {"rossmann": 196630, "rohlik": 283185}
PENALTY = 0.001
MAXITER = 250
RAW = SOURCE = OUT = None
DEVICE = "cpu"


def configure(dataset, raw, source, out, device):
    """Resolve the input/output folders shared by every function below."""
    global RAW, SOURCE, OUT, DEVICE
    RAW = (raw or ROOT / "data/raw" / dataset).expanduser().resolve()
    SOURCE = (source or ROOT / "results" / dataset / "chronos").expanduser().resolve()
    OUT = Path(out).expanduser().resolve()
    DEVICE = device
    return dict(dataset=dataset, raw=str(RAW), source=str(SOURCE), out=str(OUT), device=DEVICE)


def check_inputs(parser):
    if not RAW.is_dir():
        parser.error(f"Raw dataset directory does not exist: {RAW}")
    for folder in ("references", "chronos"):
        if not (SOURCE / folder).is_dir():
            parser.error(f"Missing {SOURCE / folder}; complete the dataset Chronos-2 run first")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def verify_completed(dest):
    """A completion marker is valid only while its recorded artifacts exist."""
    marker = dest / "complete.json"
    if not marker.exists():
        return False
    saved = json.loads(marker.read_text(encoding="utf-8"))
    if not saved.get("artifacts"):
        raise ValueError(f"Completion marker without artifact list at {dest}; choose a new --out")
    for relative in saved["artifacts"]:
        path = dest / relative
        if not path.is_file():
            raise ValueError(f"Completed artifact missing: {path}")
    return True


def artifact_list(dest):
    return [p.relative_to(dest).as_posix() for p in sorted(dest.rglob("*"))
            if p.is_file() and p.name not in ("complete.json", "RUNNING.lock")]


def validate_identity(family, dataset, extractor=None):
    """Reject cached results after changing the inputs, settings or environment."""
    import importlib.metadata
    import platform
    files = sorted(RAW.glob("*.csv"))
    if not files:
        raise ValueError(f"No raw CSV inputs in {RAW}")
    origins = sorted((SOURCE / "references").glob("*.parquet")) + sorted((SOURCE / "chronos").glob("*.parquet"))
    if not origins:
        raise ValueError(f"No stored candidates in {SOURCE}")
    packages = {}
    for name in ("numpy", "pandas", "scipy", "pyarrow", "torch", "statsmodels"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    record = dict(raw=[str(p) for p in files], source=[str(p) for p in origins],
                  python=platform.python_version(), packages=packages,
                  device=DEVICE, penalty=PENALTY, maxiter=MAXITER)
    if extractor is not None:
        record["extractor"] = str(extractor)
    # Variant/seed results and rebuilt references share a dataset identity.
    marker = OUT / "manifests" / f"{dataset}_{family}.json"
    if marker.exists() and json.loads(marker.read_text(encoding="utf-8")) != record:
        raise ValueError(f"Inputs/settings/environment changed: {marker}; use a new --out")
    if not marker.exists():
        folders = [OUT / dataset]
        if any(p.is_dir() and any(p.iterdir()) for p in folders):
            raise ValueError("Existing results lack an input manifest; choose a new --out")
    write_json(marker, record)


def metric(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    if y.shape != p.shape or not y.size or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Nonfinite score")
    if (y < 0).any() or (p < 0).any() or y.sum() <= 0:
        raise ValueError("Invalid score inputs")
    ae = np.abs(y - p)
    return dict(rows=int(len(y)), actual_sum=float(y.sum()), absolute_error_sum=float(ae.sum()),
                wape=float(ae.sum() / y.sum()), mae=float(ae.mean()))


def weekly(frame, col):
    rows = []
    for origin, part in frame.groupby("origin", sort=True):
        item = metric(part.actual, part[col])
        item["origin"] = str(origin)
        rows.append(item)
    return rows


def finish(dest: Path, scored: pd.DataFrame, col: str, extra: dict) -> None:
    if scored.origin.nunique() != 26:
        raise ValueError(f"Expected 26 test origins, got {scored.origin.nunique()}")
    overall = metric(scored.actual, scored[col])
    dataset = extra["dataset"]
    if overall["rows"] != EXPECTED[dataset]:
        raise ValueError(f"{dataset} rows {overall['rows']} != {EXPECTED[dataset]}")
    pred = scored[KEYS + [col]].rename(columns={col: "forecast"})
    dest.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(dest / "predictions.parquet", index=False)
    payload = dict(overall=overall, weekly=weekly(scored, col), **extra, finished=time.strftime("%Y-%m-%dT%H:%M:%S"))
    write_json(dest / "metrics.json", payload)
    write_json(dest / "complete.json", dict(status="complete", rows=overall["rows"], wape=overall["wape"], mae=overall["mae"], artifacts=artifact_list(dest)))
    print("DONE", extra.get("task"), dataset, overall, flush=True)


def dataset_modules(name: str):
    data_mod = importlib.import_module("data_provider." + name)
    gate_mod = importlib.import_module("exp." + name + "_gates")
    ref_mod = importlib.import_module("exp." + name + "_references")
    raw = RAW
    if name == "rossmann":
        data = data_mod.RossmannData(raw)
        factors = {"log_promo_distance", "log_school_holiday_distance", "promo", "school_holiday", "hist_factor_error"}
    else:
        data = data_mod.RohlikData(raw)
        factors = {"log_discount_distance", "log_holiday_distance", "discount", "holiday", "hist_factor_error"}
    return data, gate_mod, ref_mod, factors


def merge_keys(left, right):
    out = left.merge(right, on=KEYS, how="outer", validate="one_to_one", indicator=True)
    if not out._merge.eq("both").all():
        raise ValueError("Key mismatch")
    return out.drop(columns="_merge")


def load_frame(name, data, ref_dir: Path, backbone_dir: Path, backbone_name="backbone_forecast"):
    chunks = []
    dev, test = data.schedule["dev"], data.schedule["test"]
    for origin in dev + test:
        ref = pd.read_parquet(ref_dir / f"{origin}.parquet")
        back = pd.read_parquet(backbone_dir / f"{origin}.parquet")
        if backbone_name != "backbone_forecast" and "forecast" in back.columns:
            back = back.rename(columns={"forecast": "backbone_forecast"})
        elif "backbone_forecast" not in back.columns and "forecast" in back.columns:
            back = back.rename(columns={"forecast": "backbone_forecast"})
        frame = merge_keys(ref, back[KEYS + ["backbone_forecast"]])
        if origin in dev:
            truth = data.truth(origin)
            frame = merge_keys(frame, truth[KEYS + ["actual"]])
        else:
            frame["actual"] = np.nan
        chunks.append(frame)
    frame = pd.concat(chunks, ignore_index=True).sort_values(KEYS).reset_index(drop=True)
    train = frame.origin.isin(dev).to_numpy()
    return frame, train, test


def score_column(frame, train, data, test, col):
    pred = frame.loc[~train, KEYS + [col]].copy()
    truth = pd.concat([data.truth(o)[KEYS + ["actual"]] for o in test], ignore_index=True)
    scored = merge_keys(truth, pred)
    return scored
