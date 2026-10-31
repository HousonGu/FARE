"""TimeFuse on the stored B/S/F candidates of Rossmann and Rohlik.

Uses the 22 TimeFuse meta features of the 56-day history (upstream
third_party/TimeFuse/meta_feature.py) and a linear fusor with a softmax over the
three candidates: seed 2021, five epochs, batch size 64. Development rows train
the fusor; test rows are scored once. The forecasting backbone is not refitted.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import exp.stored_candidates as S  # noqa: E402

TIMEFUSE_SOURCE = ROOT / "third_party" / "TimeFuse"


FEATURE_NAMES = (
    "mean", "std", "min", "max", "skewness", "kurtosis", "autocorrelation_mean",
    "stationarity", "rate_of_change_mean", "rate_of_change_std", "autoreg_coef_mean",
    "residual_std_mean", "frequency_mean", "frequency_peak", "spectral_entropy",
    "spectral_variation", "spectral_skewness", "spectral_kurtosis", "covariance_mean",
    "covariance_max", "covariance_min", "covariance_std",
)


def meta_row(history, counts):
    sys.path.insert(0, str(TIMEFUSE_SOURCE))
    try:
        from meta_feature import extract_meta_feature
    except ImportError as error:
        raise ImportError("TimeFuse source missing: run bash scripts/fetch_third_party.sh TimeFuse first") from error
    if history.shape != (56,) or not np.isfinite(history).all() or (history < 0).any():
        raise ValueError("History must be 56 finite non-negative days")
    if float(np.std(history)) == 0:
        counts["constant"] += 1
        row = np.zeros(len(FEATURE_NAMES))
        row[0] = row[2] = row[3] = float(history[0])
        return row
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            features = extract_meta_feature(history.reshape(56, 1))
    except Exception:
        counts["exception"] += 1
        return np.zeros(len(FEATURE_NAMES))
    row = np.array([features[n] for n in FEATURE_NAMES], float)
    bad = ~np.isfinite(row)
    counts["nonfinite"] += int(bad.sum())
    row[bad] = 0
    return row


def fuse_train(meta, forecasts, actual, device):
    import random
    import torch
    import torch.nn as nn
    torch.manual_seed(2021)
    random.seed(2021)
    np.random.seed(2021)
    mean = meta.mean(0)
    std = np.maximum(meta.std(0), 1e-8)
    model = nn.Linear(len(FEATURE_NAMES), 3).to(device)
    nn.init.kaiming_uniform_(model.weight, a=np.sqrt(5))
    opt = torch.optim.Adam(model.parameters(), lr=0.0005)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)
    loss_fn = nn.SmoothL1Loss(beta=0.01)
    x = torch.tensor((meta - mean) / std, dtype=torch.float32)
    yhat = torch.tensor(forecasts, dtype=torch.float32)
    y = torch.tensor(actual, dtype=torch.float32)
    model.train()
    for epoch in range(5):
        order = torch.randperm(len(x)).numpy()
        losses = []
        for start in range(0, len(order), 64):
            batch = order[start:start + 64]
            xb, pb, yb = x[batch].to(device), yhat[batch].to(device), y[batch].to(device)
            weights = torch.softmax(model(xb), dim=-1)
            fused = (weights.unsqueeze(-1) * pb).sum(1)
            loss = loss_fn(fused, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        sched.step()
        print(f"timefuse epoch {epoch + 1} {np.mean(losses):.6f}", flush=True)
    model.eval()
    return model, mean, std


def fuse_predict(model, meta, forecasts, mean, std, device):
    import torch
    x = torch.tensor((meta - mean) / std, dtype=torch.float32)
    yhat = torch.tensor(forecasts, dtype=torch.float32)
    out = []
    with torch.no_grad():
        for start in range(0, len(x), 64):
            w = torch.softmax(model(x[start:start + 64].to(device)), dim=-1)
            out.append((w.unsqueeze(-1) * yhat[start:start + 64].to(device)).sum(1).cpu().numpy())
    return np.vstack(out)


def run_timefuse(name):
    data, _g, _r, _f = S.dataset_modules(name)
    import torch
    source = S.SOURCE
    dest = S.OUT / name
    if S.verify_completed(dest):
        print("skip", dest, flush=True)
        return
    dest.mkdir(parents=True, exist_ok=True)
    dev, test = data.schedule["dev"], data.schedule["test"]
    blocks = {}
    counts = dict(constant=0, exception=0, nonfinite=0)
    for split, origins in (("dev", dev), ("test", test)):
        chunks = []
        for origin in origins:
            ref = pd.read_parquet(source / "references" / f"{origin}.parquet")
            back = pd.read_parquet(source / "chronos" / f"{origin}.parquet")
            frame = S.merge_keys(ref, back).sort_values(S.KEYS).reset_index(drop=True)
            cache = dest / f"{origin}_meta.npz"
            ids = frame.series_id.to_numpy()[::7]
            if cache.exists():
                with np.load(cache) as z:
                    meta = z["meta"]
                    if meta.shape != (len(ids), len(FEATURE_NAMES)) or not np.isfinite(meta).all():
                        raise ValueError("Invalid cached TimeFuse features")
                    if not np.array_equal(z["ids"], ids.astype(str)):
                        raise ValueError("Meta cache ids changed")
            else:
                h, _cov, keys = data.window(origin)
                # window() row order follows eligible(); align to sorted frame ids
                hist = {int(sid): row for sid, row in zip(data.ids[data.eligible(origin)], h)}
                local = dict(constant=0, exception=0, nonfinite=0)
                meta = np.vstack([meta_row(hist[int(sid)].astype(float), local) for sid in ids])
                np.savez_compressed(cache, meta=meta, ids=ids.astype(str))
                for k, v in local.items():
                    counts[k] += v
                print("timefuse meta", name, origin, len(ids), flush=True)
            parts = frame[["backbone_forecast", "shape_retrieval", "factor_retrieval"]].to_numpy().reshape(-1, 7, 3).transpose(0, 2, 1)
            actual = None
            if split == "dev":
                truth = data.truth(origin)[S.KEYS + ["actual"]]
                aligned = S.merge_keys(frame[S.KEYS], truth).sort_values(S.KEYS)
                if not aligned[S.KEYS].reset_index(drop=True).equals(frame[S.KEYS].reset_index(drop=True)):
                    raise ValueError("Dev actual alignment failed at " + origin)
                actual = aligned.actual.to_numpy().reshape(-1, 7)
            chunks.append((frame[S.KEYS].copy(), meta, parts, actual))
        blocks[split] = chunks
    device = torch.device(S.DEVICE)
    dev_blocks = blocks["dev"]
    fusor, mean, std = fuse_train(np.concatenate([x[1] for x in dev_blocks]),
                                  np.concatenate([x[2] for x in dev_blocks]),
                                  np.concatenate([x[3] for x in dev_blocks]), device)
    torch.save(fusor.state_dict(), dest / "model.pt")
    output = []
    for keys, meta, parts, _ in blocks["test"]:
        forecast = fuse_predict(fusor, meta, parts, mean, std, device)
        keys = keys.copy()
        keys["forecast"] = np.maximum(forecast.ravel(), 0)
        output.append(keys)
    pred = pd.concat(output, ignore_index=True)
    truth = pd.concat([data.truth(o)[S.KEYS + ["actual"]] for o in test], ignore_index=True)
    scored = S.merge_keys(truth, pred)
    S.write_json(dest / "fit.json", dict(adapter="22-feature TimeFuse same-candidate", seed=2021, epochs=5,
                                       mean=mean.tolist(), std=std.tolist(), replacement_counts=counts,
                                       features=list(FEATURE_NAMES)))
    S.finish(dest, scored, "forecast", dict(task="timefuse", dataset=name, variant="TimeFuse", seed=2021))


def main(argv=None):
    global TIMEFUSE_SOURCE
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=("rossmann", "rohlik"), required=True)
    p.add_argument("--raw", type=Path, help="raw CSV folder; default data/raw/DATASET")
    p.add_argument("--source", type=Path, help="completed Chronos-2 run with references/ and chronos/")
    p.add_argument("--out", type=Path, default=ROOT / "results/baselines/timefuse")
    p.add_argument("--device", default="cpu")
    p.add_argument("--timefuse-source", type=Path, default=TIMEFUSE_SOURCE)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    plan = S.configure(a.dataset, a.raw, a.source, a.out, a.device)
    TIMEFUSE_SOURCE = a.timefuse_source.expanduser().resolve()
    if a.dry_run:
        print(json.dumps(dict(plan, timefuse_source=str(TIMEFUSE_SOURCE)), indent=2))
        return 0
    S.check_inputs(p)
    if not (TIMEFUSE_SOURCE / "meta_feature.py").is_file():
        p.error("Missing TimeFuse meta_feature.py; run bash scripts/fetch_third_party.sh TimeFuse")
    S.validate_identity("timefuse", a.dataset, TIMEFUSE_SOURCE / "meta_feature.py")
    warnings.filterwarnings("ignore", category=FutureWarning)
    run_timefuse(a.dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
