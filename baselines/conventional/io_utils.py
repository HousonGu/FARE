"""Atomic artifacts and strict evaluation for the baseline runs."""
from pathlib import Path
import importlib.metadata
import json
import os
import platform
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
KEYS = ['origin', 'series_id', 'target_date']


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def verify_file(path):
    if not Path(path).is_file():
        raise ValueError('Missing artifact: ' + str(path))


def save_parquet(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    frame.to_parquet(temporary, index=False, compression='zstd')
    os.replace(temporary, path)


def save_csv(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def seal(folder, files, extra=None, name='complete.json'):
    folder = Path(folder)
    write_json(folder / name, dict(artifacts=sorted(Path(p).relative_to(folder).as_posix() for p in files), **(extra or {})))


def verify(folder, name='complete.json'):
    folder = Path(folder)
    marker = folder / name
    if not marker.is_file():
        return None
    record = json.loads(marker.read_text(encoding='utf-8'))
    for relative in record['artifacts']:
        path = folder / relative
        if not path.is_file():
            raise ValueError('Missing artifact: ' + str(path))
    return record


def canonical(frame):
    frame = frame.copy()
    for key in KEYS:
        if frame[key].isna().any():
            raise ValueError('Missing key ' + key)
        frame[key] = frame[key].astype(str)
    if frame.duplicated(KEYS).any():
        raise ValueError('Duplicate prediction keys')
    return frame


def merge(left, right):
    frame = canonical(left).merge(canonical(right), on=KEYS, how='outer', validate='one_to_one', indicator=True, sort=False)
    if not frame['_merge'].eq('both').all():
        raise ValueError('Prediction keys differ; cannot drop rows')
    return frame.drop(columns='_merge')


def metric(actual, prediction):
    y, p = np.asarray(actual, np.float64), np.asarray(prediction, np.float64)
    if (y.ndim != 1 or y.shape != p.shape or not y.size or not np.isfinite(y).all()
            or not np.isfinite(p).all() or (y < 0).any() or (p < 0).any() or y.sum() <= 0):
        raise ValueError('Invalid evaluation values')
    error = float(np.abs(y-p).sum())
    total = float(y.sum())
    return dict(rows=len(y), actual_sum=total, absolute_error_sum=error, wape=error/total, mae=error/len(y))


def pool(records):
    rows = sum(r['rows'] for r in records)
    total = sum(r['actual_sum'] for r in records)
    error = sum(r['absolute_error_sum'] for r in records)
    if rows <= 0 or total <= 0:
        raise ValueError('Empty or zero-sum evaluation')
    return dict(rows=rows, actual_sum=total, absolute_error_sum=error, wape=error/total, mae=error/rows)


def environment():
    versions = {}
    for name in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'pyarrow', 'statsforecast', 'xgboost', 'numba', 'joblib']:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = importlib.metadata.version('xgboost-cpu') if name == 'xgboost' else None
    return dict(python=platform.python_version(), platform=platform.platform(), versions=versions)
