import hashlib,json,os
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
CONFIG=json.loads((ROOT/"configs/rohlik_supervised.json").read_text())
KEYS=["origin","series_id","target_date"]
def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def json_write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)
    seal(path)

def parquet_write(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    seal(path)


def seal(path):
    path=Path(path)
    marker=path.with_name(path.name+'.sha256')
    tmp=marker.with_name(marker.name+'.tmp')
    tmp.write_text(digest(path),encoding='ascii')
    os.replace(tmp,marker)


def verify(path):
    path=Path(path)
    if path.with_name(path.name+'.sha256').read_text(encoding='ascii') != digest(path):
        raise ValueError('Artifact changed: '+str(path))

def checked_merge(left, right):
    merged = left.merge(right, on=KEYS, how="outer", validate="one_to_one", indicator=True)
    if not merged._merge.eq("both").all():
        raise ValueError("Prediction keys differ; no inner-join row dropping allowed")
    return merged.drop(columns="_merge")

def metric(y, pred):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    if (y.shape != pred.shape or not y.size or not np.isfinite(y).all()
            or not np.isfinite(pred).all() or (y < 0).any() or (pred < 0).any() or y.sum() <= 0):
        raise ValueError("Invalid scores")
    ae = np.abs(y - pred)
    return dict(rows=len(y), actual_sum=float(y.sum()), absolute_error_sum=float(ae.sum()),
                wape=float(ae.sum() / y.sum()), mae=float(ae.mean()))

def check_prediction(frame):
    if frame.duplicated(KEYS).any() or frame[KEYS].isna().any().any():
        raise ValueError("Duplicate/missing keys")
    delta = (pd.to_datetime(frame.target_date)-pd.to_datetime(frame.origin)).dt.days
    if not delta.between(0, 6).all() or not frame.groupby(["origin", "series_id"]).size().eq(7).all():
        raise ValueError("Incomplete forecast week")
    for col in frame.columns.difference(KEYS):
        if not np.isfinite(frame[col].to_numpy(float)).all():
            raise ValueError(f"Nonfinite {col}")

class RetailData:

    def tree(self, origin, labels=False):
        rows, h, scale, cov, y, keys = self.window(origin, labels=labels)
        n = h / scale[:, None]
        repeated = lambda a: np.repeat(a, 7)
        values = {'log_scale': repeated(np.log1p(scale)), 'zero_fraction': repeated((h == 0).mean(1))}
        for period in (7, 14, 28, 56):
            values[f'mean_{period}'] = repeated(n[:, -period:].mean(1))
            values[f'std_{period}'] = repeated(n[:, -period:].std(1))
        for lag in (7, 14, 28, 56):
            values[f'lag_{lag}'] = n[:, 56 - lag:63 - lag].ravel()
        values['weekday_mean'] = n.reshape(-1, 8, 7).mean(1).ravel()
        values['last'] = repeated(n[:, -1])
        values['horizon'] = np.tile(np.arange(7), len(rows))
        for i, name in enumerate(self.static_names):
            values[name] = repeated(self.static[rows, i])
        values['factor1'] = cov[:, 56:, 7].ravel()
        values['factor2'] = cov[:, 56:, 8].ravel()
        for factor in (7, 8):
            for day in range(7):
                values[f'future_{factor}_{day}'] = repeated(cov[:, 56 + day, factor])
        x = pd.DataFrame(values, dtype=np.float32)
        return (x, None if y is None else (y / scale[:, None]).ravel(), repeated(scale), keys)
