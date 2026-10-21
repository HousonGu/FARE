"""Shared I/O, frozen schedules, and origin-local raw data access."""
from __future__ import annotations
import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.dont_write_bytecode = True
from data_provider.paths import RAW_ROOT, RESULT_ROOT
CONFIG = json.loads((ROOT / "configs/retail.json").read_text(encoding="utf-8"))
KEYS = ["origin", "series_id", "target_date"]


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


def parquet_write(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def weeks(start, n):
    return pd.date_range(start, periods=n, freq="7D").strftime("%Y-%m-%d").tolist()


def schedules():
    s = json.loads((ROOT / "configs/favorita_origins.json").read_text(encoding="utf-8"))
    origins = pd.date_range(CONFIG["m5_first_origin"], CONFIG["m5_last_origin"], freq="7D").strftime("%Y-%m-%d").tolist()
    ndev, ntest = CONFIG["m5_development_weeks"], CONFIG["m5_test_weeks"]
    out = {"m5_full": dict(dataset="m5", fit=origins[:-(ndev+ntest)],
        dev=origins[-(ndev+ntest):-ntest], test=origins[-ntest:],
        kind="unified 52-week retrospective evaluation with one pre-test fit/development protocol"),
        "favorita_original": dict(dataset="favorita", fit=s["fit"], dev=s["dev"],
            test=s["test"], kind="used original evaluation period")}
    for s in out.values():
        s["tune"] = s["fit"][-CONFIG["tuning_weeks"]:]
        s["train"] = s["fit"][:-CONFIG["tuning_weeks"]]
        for a, b in (("train", "tune"), ("fit", "dev"), ("dev", "test")):
            if pd.Timestamp(s[a][-1]) + pd.Timedelta(days=6) >= pd.Timestamp(s[b][0]):
                raise ValueError(f"Overlapping {a}/{b}")
    return out


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


class RetailData:
    def __init__(self, dataset):
        from data_provider.retail_matrix import load_matrix
        self.dataset = dataset
        matrix, groups, first = load_matrix(dataset)
        self.ids = matrix.index.to_numpy()
        self.days = list(matrix.columns)
        self.columns = {day: i for i, day in enumerate(self.days)}
        self.values = matrix.to_numpy(dtype=np.float64)
        self.groups = groups.reindex(matrix.index).to_numpy()
        self.first = first.reindex(matrix.index).to_numpy(dtype="datetime64[ns]")
        raw = RAW_ROOT / dataset
        if dataset == "m5":
            meta = pd.read_csv(raw / "sales_train_evaluation.csv", usecols=["id", "store_id", "item_id", "dept_id", "cat_id", "state_id"]).set_index("id").loc[self.ids]
            static_cols = ["store_id", "item_id", "dept_id", "cat_id", "state_id"]
            cal = pd.read_csv(raw / "calendar.csv").set_index("date").loc[self.days]
            states = pd.Categorical(meta.state_id, categories=["CA", "TX", "WI"]).codes
            snap = np.stack([cal[f"snap_{s}"].to_numpy(np.float32) for s in ("CA", "TX", "WI")])
            self.factor1 = snap[states]
            event = (cal.event_name_1.notna() | cal.event_name_2.notna()).to_numpy(np.float32)
            self.factor2 = np.broadcast_to(event, self.values.shape)
            self.promo = None
        else:
            from data_provider.retail_factors import factors
            self.factor1, self.factor2 = factors(matrix.index, matrix.columns)
            meta = pd.read_csv(raw / "train.csv", usecols=["store_nbr", "family"]).drop_duplicates().sort_values(["store_nbr", "family"]).reset_index(drop=True)
            meta.index = [f"f{i:04d}" for i in range(len(meta))]
            meta = meta.loc[self.ids]
            static_cols = ["store_nbr", "family"]
            p = pd.read_csv(raw / "train.csv", usecols=["date", "store_nbr", "family", "onpromotion"])
            p = p.merge(meta.reset_index(names="series_id"), on=["store_nbr", "family"], validate="many_to_one")
            self.promo = p.pivot(index="series_id", columns="date", values="onpromotion").loc[self.ids, self.days].to_numpy(np.float32)
        self.static_names = static_cols
        self.static = np.column_stack([pd.Categorical(meta[c]).codes for c in static_cols]).astype(np.int64)
        self.cardinalities = {c: int(self.static[:, i].max()) + 1 for i, c in enumerate(static_cols)}
        self.group_rows = [np.flatnonzero(self.groups == g) for g in sorted(set(self.groups))]

    def eligible(self, origin):
        return np.flatnonzero(self.first <= np.datetime64(origin) - np.timedelta64(56, "D"))

    def indices(self, origin):
        if not hasattr(self, "_index_cache"):
            self._index_cache = {}
        if origin in self._index_cache:
            return self._index_cache[origin]
        dates = pd.date_range(pd.Timestamp(origin) - pd.Timedelta(days=56), periods=63)
        result = np.array([self.columns[d] for d in dates.strftime("%Y-%m-%d")])
        self._index_cache[origin] = result
        return result

    def window(self, origin, rows=None, labels=False):
        rows = self.eligible(origin) if rows is None else np.asarray(rows)
        if not np.isin(rows, self.eligible(origin)).all():
            raise ValueError("Ineligible query series")
        cols = self.indices(origin)
        h = self.values[np.ix_(rows, cols[:56])]
        if not np.isfinite(h).all() or (h < 0).any():
            raise ValueError("Invalid history")
        scale = np.maximum(h.mean(axis=1, dtype=np.float64), 1).astype(np.float32)
        c1, c2 = self.factor1[np.ix_(rows, cols)], self.factor2[np.ix_(rows, cols)]
        dates = pd.date_range(pd.Timestamp(origin) - pd.Timedelta(days=56), periods=63)
        dow = np.broadcast_to(np.eye(7, dtype=np.float32)[dates.dayofweek], (len(rows), 63, 7))
        cov = np.concatenate([dow, c1[..., None], c2[..., None]], axis=2).astype(np.float32)
        if not np.isfinite(cov).all():
            raise ValueError("Invalid covariates")
        y = self.values[np.ix_(rows, cols[56:])] if labels else None
        keys = pd.DataFrame({"origin": origin, "series_id": np.repeat(self.ids[rows], 7),
                             "target_date": np.tile(dates[56:].strftime("%Y-%m-%d"), len(rows))})
        return rows, h, scale, cov, y, keys

    def tree(self, origin, labels=False):
        rows, h, scale, cov, y, keys = self.window(origin, labels=labels)
        n = h / scale[:, None]
        repeated = lambda a: np.repeat(a, 7)
        values = {"log_scale": repeated(np.log1p(scale)), "zero_fraction": repeated((h == 0).mean(1))}
        for period in (7, 14, 28, 56):
            values[f"mean_{period}"] = repeated(n[:, -period:].mean(1))
            values[f"std_{period}"] = repeated(n[:, -period:].std(1))
        for lag in (7, 14, 28, 56):
            values[f"lag_{lag}"] = n[:, 56-lag:63-lag].ravel()
        values["weekday_mean"] = n.reshape(-1, 8, 7).mean(1).ravel()
        values["last"] = repeated(n[:, -1])
        values["horizon"] = np.tile(np.arange(7), len(rows))
        for i, name in enumerate(self.static_names):
            values[name] = repeated(self.static[rows, i])
        values["factor1"] = cov[:, 56:, 7].ravel()
        values["factor2"] = cov[:, 56:, 8].ravel()
        # Full known future factor pattern is available to every supervised model.
        for factor in (7, 8):
            for day in range(7):
                values[f"future_{factor}_{day}"] = repeated(cov[:, 56+day, factor])
        x = pd.DataFrame(values, dtype=np.float32)
        return x, None if y is None else (y/scale[:, None]).ravel(), repeated(scale), keys

    def truth(self, origins):
        chunks = []
        for origin in origins:
            _, _, _, _, y, keys = self.window(origin, labels=True)
            keys["actual"] = y.ravel()
            chunks.append(keys)
        return pd.concat(chunks, ignore_index=True)


def check_prediction(frame):
    if frame.duplicated(KEYS).any() or frame[KEYS].isna().any().any():
        raise ValueError("Duplicate/missing keys")
    delta = (pd.to_datetime(frame.target_date)-pd.to_datetime(frame.origin)).dt.days
    if not delta.between(0, 6).all() or not frame.groupby(["origin", "series_id"]).size().eq(7).all():
        raise ValueError("Incomplete forecast week")
    for col in frame.columns.difference(KEYS):
        if not np.isfinite(frame[col].to_numpy(float)).all():
            raise ValueError(f"Nonfinite {col}")
