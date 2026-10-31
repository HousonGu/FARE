"""Retrieval and error-estimation ablations on Rossmann and Rohlik.

Reads the stored references and Chronos-2 forecasts of the completed main run
(``--source``, default ``results/DATASET/chronos``) and refits the gates on the
development rows for one variant:

  no_factor          drop the factor reference F and its gate inputs
  no_disagreement    drop the two disagreement inputs
  A1_joint           F from the joint shape-and-factor Top-8 (references rebuilt)
  A2_random          F from 8 random library entries (references rebuilt, seeded)
  A3_global_errors   historical errors from the global past mean
  A4_no_shrink       per-series historical errors without shrinkage
  A5_constant_alpha  alpha reduced to a constant
  A6_constant_beta   beta reduced to a constant
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import exp.stored_candidates as S  # noqa: E402

VARIANTS = ("no_factor", "no_disagreement", "A1_joint", "A2_random", "A3_global_errors",
            "A4_no_shrink", "A5_constant_alpha", "A6_constant_beta")


def objective(theta, x, y, parts, active):
    k = x.shape[1]
    full = np.zeros(2 * k)
    full[active] = theta
    beta, alpha = expit(x @ full[:k]), expit(x @ full[k:])
    b, s, f = parts.T
    ref = s + alpha * (f - s)
    pred = b + beta * (ref - b)
    residual = pred - y
    smooth = np.sqrt(residual ** 2 + 1)
    g = residual / smooth / y.sum()
    grad = np.r_[x.T @ (g * beta * (1 - beta) * (ref - b)), x.T @ (g * beta * alpha * (1 - alpha) * (f - s))]
    penalized = np.ones(2 * k, bool)
    penalized[[0, k]] = False
    grad[penalized] += 2 * S.PENALTY * full[penalized]
    value = smooth.sum() / y.sum() + S.PENALTY * np.square(full[penalized]).sum()
    return value, grad[active]


def modified_history(frame, train, parts, variant):
    codes, ids = pd.factorize(frame.series_id, sort=False)
    sums, counts = np.zeros((len(ids), 3)), np.zeros(len(ids))
    total, count = np.zeros(3), 0
    result = np.empty((len(frame), 4))
    scale = np.maximum(frame.history_mean.to_numpy(float), 1)
    for origin in sorted(frame.origin.unique()):
        rows = np.flatnonzero(frame.origin.to_numpy() == origin)
        c = codes[rows]
        prior = total / count if count else np.zeros(3)
        if variant == "A3_global_errors":
            errors = np.broadcast_to(prior, (len(rows), 3))
        else:
            errors = np.divide(sums[c], counts[c, None], out=np.tile(prior, (len(rows), 1)), where=counts[c, None] > 0)
        result[rows, :3] = np.clip(errors, 0, 10)
        result[rows, 3] = np.log1p(counts[c])
        if train[rows].all():
            observed = np.clip(np.abs(frame.actual.to_numpy(float)[rows, None] - parts[rows]) / scale[rows, None], 0, 10)
            for j in range(3):
                sums[:, j] += np.bincount(c, weights=observed[:, j], minlength=len(ids))
            counts += np.bincount(c, minlength=len(ids))
            total += observed.sum(0)
            count += len(rows)
        elif train[rows].any():
            raise ValueError("Mixed origin split")
    return result


def fit_predict(x, parts, y, train, variant):
    from fare.fusion import fit_gate, predict_gate
    k = x.shape[1]
    if variant in ("A5_constant_alpha", "A6_constant_beta", "A3_global_errors", "A4_no_shrink"):
        active = np.arange(2 * k)
        if variant == "A5_constant_alpha":
            active = np.arange(k + 1)
        if variant == "A6_constant_beta":
            active = np.r_[0, np.arange(k, 2 * k)]
        result = minimize(objective, np.zeros(len(active)), args=(x[train], y[train], parts[train], active),
                          jac=True, method="L-BFGS-B", options=dict(maxiter=S.MAXITER, ftol=1e-9, gtol=1e-6))
        if not result.success or not np.isfinite(result.x).all():
            raise RuntimeError(f"{variant}: {result.message}")
        full = np.zeros(2 * k)
        full[active] = result.x
        pred, beta, alpha = predict_gate(x[~train], full, *parts[~train].T)
        return pred, dict(iterations=int(result.nit), objective=float(result.fun), active=active.tolist())
    # no_factor / no_disagreement / A1_joint / A2_random use the full two-layer gate on the prepared design
    single = variant == "no_factor"
    theta, record = fit_gate(x[train], y[train], *parts[train].T, single=single)
    pred = predict_gate(x[~train], theta, parts[~train, 0], parts[~train, 1], None if single else parts[~train, 2])[0]
    return pred, record


def run_ablation(name, variant, seed, ref_dir=None):
    data, gates, _ref, factors = S.dataset_modules(name)
    source = S.SOURCE
    if ref_dir is None:
        ref_dir = source / "references"
    dest = S.OUT / name / f"{variant}_{seed}"
    if S.verify_completed(dest):
        print("skip", dest, flush=True)
        return
    frame, train, test = S.load_frame(name, data, ref_dir, source / "chronos")
    x, parts, meta = gates.design(frame, train)
    y = frame.actual.to_numpy(float)
    if variant in ("A3_global_errors", "A4_no_shrink"):
        raw = modified_history(frame, train, parts, variant)
        mean = raw[train].mean(0)
        std = np.maximum(raw[train].std(0), 1e-8)
        x = x.copy()
        x[:, -4:] = (raw - mean) / std
    elif variant == "no_factor":
        keep = [0] + [i + 1 for i, n in enumerate(meta["names"]) if n not in factors]
        x = np.ascontiguousarray(x[:, keep])
    elif variant == "no_disagreement":
        drop = {"backbone_shape_disagreement", "factor_shape_disagreement"}
        keep = [0] + [i + 1 for i, n in enumerate(meta["names"]) if n not in drop]
        x = np.ascontiguousarray(x[:, keep])
    elif variant in ("A5_constant_alpha", "A6_constant_beta", "A1_joint", "A2_random"):
        pass
    else:
        raise ValueError(variant)
    pred, record = fit_predict(x, parts, y, train, variant)
    frame = frame.copy()
    frame.loc[~train, "forecast"] = pred
    scored = S.score_column(frame, train, data, test, "forecast")
    S.write_json(dest / "fit.json", dict(variant=variant, seed=seed, record=record, names=meta["names"]))
    S.finish(dest, scored, "forecast", dict(task="ablation", dataset=name, variant=variant, seed=int(seed),
                                           reference_dir=str(ref_dir)))


def rebuild_references(name, variant, seed):
    data, _gates, ref_mod, _factors = S.dataset_modules(name)
    dest = S.OUT / name / "references" / f"{variant}_{seed}"
    marker = dest / "complete.json"
    if S.verify_completed(dest):
        return dest
    original = ref_mod.select
    dest.mkdir(parents=True, exist_ok=True)
    origins = data.schedule["dev"] + data.schedule["test"]
    for origin in origins:
        path = dest / f"{origin}.parquet"
        if path.exists():
            continue
        local = int.from_bytes(hashlib.sha256(f"{seed}:{origin}".encode()).digest()[:8], "little")
        rng = np.random.default_rng(local)

        def replacement(shape, promo, holiday, rng=rng, original=original):
            chosen, dist = original(shape, promo, holiday)
            chosen = dict(chosen)
            if variant == "A1_joint":
                chosen["factor"] = chosen["joint"]
            else:
                chosen["factor"] = np.vstack([rng.choice(shape.shape[1], 8, replace=False) for _ in range(shape.shape[0])])
            return chosen, dist

        ref_mod.select = replacement
        try:
            frame = ref_mod.references(data, origin, 8)
        finally:
            ref_mod.select = original
        frame.to_parquet(path, index=False)
        print("rebuilt", name, variant, seed, origin, len(frame), flush=True)
    S.write_json(marker, dict(status="complete", variant=variant, seed=seed, weeks=len(origins), artifacts=S.artifact_list(dest)))
    return dest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=("rossmann", "rohlik"), required=True)
    p.add_argument("--variant", choices=VARIANTS, required=True)
    p.add_argument("--seed", type=int, default=20261005)
    p.add_argument("--raw", type=Path, help="raw CSV folder; default data/raw/DATASET")
    p.add_argument("--source", type=Path, help="completed Chronos-2 run with references/ and chronos/")
    p.add_argument("--out", type=Path, default=ROOT / "results/ablations/extended")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    plan = S.configure(a.dataset, a.raw, a.source, a.out, a.device)
    if a.dry_run:
        print(json.dumps(dict(plan, variant=a.variant, seed=a.seed), indent=2))
        return 0
    S.check_inputs(p)
    S.validate_identity("ablation", a.dataset)
    warnings.filterwarnings("ignore", category=FutureWarning)
    refs = rebuild_references(a.dataset, a.variant, a.seed) if a.variant in ("A1_joint", "A2_random") else None
    run_ablation(a.dataset, a.variant, a.seed, refs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
