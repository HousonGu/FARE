"""Atomic artifacts and strict evaluation-key checks for the standalone bundle."""
from pathlib import Path
import json
import os
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


def save_parquet(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    frame.to_parquet(temporary, index=False, compression='zstd')
    os.replace(temporary, path)


def verify_file(path):
    if not Path(path).is_file():
        raise ValueError('Missing artifact: ' + str(path))


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
    left, right = canonical(left), canonical(right)
    frame = left.merge(right, on=KEYS, how='outer', validate='one_to_one', indicator=True, sort=False)
    if not frame['_merge'].eq('both').all():
        raise ValueError('Prediction keys differ; no row dropping allowed')
    return frame.drop(columns='_merge')


def metric(actual, prediction):
    actual, prediction = np.asarray(actual, np.float64), np.asarray(prediction, np.float64)
    if (actual.ndim != 1 or actual.shape != prediction.shape or not actual.size
            or not np.isfinite(actual).all() or not np.isfinite(prediction).all()
            or (actual < 0).any() or (prediction < 0).any() or actual.sum() <= 0):
        raise ValueError('Invalid evaluation values')
    error = np.abs(actual - prediction)
    return dict(rows=len(actual), actual_sum=float(actual.sum()), absolute_error_sum=float(error.sum()),
                wape=float(error.sum() / actual.sum()), mae=float(error.mean()))


def pool(records):
    rows = sum(v['rows'] for v in records)
    actual = sum(v['actual_sum'] for v in records)
    error = sum(v['absolute_error_sum'] for v in records)
    return dict(rows=rows, actual_sum=actual, absolute_error_sum=error, wape=error / actual, mae=error / rows)

