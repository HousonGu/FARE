"""Portable, atomic I/O; existing experiment inputs are always read-only."""
from pathlib import Path
import json
import os

KEYS = ['origin', 'series_id', 'target_date']
VARIANTS = ('bs_path', 'bf_path', 'both_constant')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def seal(root, identity):
    root = Path(root)
    artifacts = sorted(str(p.relative_to(root)).replace('\\', '/')
                       for p in root.rglob('*') if p.is_file() and p.name not in ('complete.json', 'RUNNING.lock'))
    write_json(root / 'complete.json', dict(identity=identity, artifacts=artifacts))


def verify(root):
    root = Path(root).resolve()
    record = json.loads((root / 'complete.json').read_text(encoding='utf-8'))
    for name in record['artifacts']:
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('Missing or unsafe completed artifact: ' + name)
    return record['identity']


def canonical(frame):
    import pandas as pd
    frame = frame.copy()
    for name in ('origin', 'target_date'):
        frame[name] = pd.to_datetime(frame[name]).dt.strftime('%Y-%m-%d')
    frame['series_id'] = frame.series_id.astype(str)
    if frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
        raise ValueError('Missing or duplicate prediction keys')
    return frame


def exact_merge(left, right):
    out = left.merge(right, on=KEYS, how='outer', validate='one_to_one', indicator=True)
    if not out._merge.eq('both').all():
        raise ValueError('Prediction keys differ; row dropping is forbidden')
    return out.drop(columns='_merge')


def metric(y, pred):
    import numpy as np
    y, pred = np.asarray(y), np.asarray(pred)
    if not y.size or y.shape != pred.shape or not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError('Invalid scoring arrays')
    if (y < 0).any() or (pred < 0).any():
        raise ValueError('Negative target or prediction')
    ae = float(np.abs(y - pred).sum()); total = float(y.sum())
    return dict(rows=len(y), actual_sum=total, absolute_error_sum=ae,
                wape=ae / total if total else None, mae=ae / len(y))
