"""Retrieval and error-estimation ablations A1-A6 on Favorita/M5.

Reads a completed Chronos-2 run of ``exp.retail`` (``--source``) and never
modifies it. Each variant changes one part of FARE, refits the two gates on
the development rows, and scores the test rows once:

  A1_joint           factor reference = joint shape-and-factor Top-8
  A2_random          factor reference = 8 random library entries (seeded)
  A3_global_errors   historical errors from the global past mean
  A4_no_shrink       per-series historical errors without shrinkage
  A5_constant_alpha  alpha reduced to a constant
  A6_constant_beta   beta reduced to a constant

A1/A2 rebuild the references from the same past-origin library; all other
variants reuse the stored references and Chronos-2 forecasts.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from data_provider.retail import (CONFIG, KEYS, RetailData, checked_merge, check_prediction,
                                  json_write, parquet_write, metric)
from exp.retail_gates import design

VARIANTS = ('A3_global_errors', 'A4_no_shrink', 'A5_constant_alpha', 'A6_constant_beta', 'A1_joint', 'A2_random')


def objective(theta, x, y, parts, active):
    k = x.shape[1]
    full = np.zeros(2*k)
    full[active] = theta
    beta, alpha = expit(x @ full[:k]), expit(x @ full[k:])
    b, s, f = parts.T
    ref = s + alpha*(f-s)
    pred = b + beta*(ref-b)
    residual = pred-y
    smooth = np.sqrt(residual**2+1)
    g = residual/smooth/y.sum()
    grad = np.r_[x.T @ (g*beta*(1-beta)*(ref-b)), x.T @ (g*beta*alpha*(1-alpha)*(f-s))]
    penalized = np.ones(2*k, bool)
    penalized[[0, k]] = False
    grad[penalized] += 2*CONFIG['gate_penalty']*full[penalized]
    value = smooth.sum()/y.sum() + CONFIG['gate_penalty']*np.square(full[penalized]).sum()
    return value, grad[active]


def modified_history(frame, train, parts, variant):
    """Past development labels only; retain per-series support for A3."""
    codes, ids = pd.factorize(frame.series_id, sort=False)
    sums, counts = np.zeros((len(ids), 3)), np.zeros(len(ids))
    total, count = np.zeros(3), 0
    result = np.empty((len(frame), 4))
    scale = np.maximum(frame.history_mean.to_numpy(float), 1)
    for origin in sorted(frame.origin.unique()):
        rows = np.flatnonzero(frame.origin.to_numpy() == origin)
        c = codes[rows]
        prior = total/count if count else np.zeros(3)
        if variant == 'A3_global_errors':
            errors = np.broadcast_to(prior, (len(rows), 3))
        else:
            errors = np.divide(sums[c], counts[c, None], out=np.tile(prior, (len(rows), 1)), where=counts[c, None] > 0)
        result[rows, :3] = np.clip(errors, 0, 10)
        result[rows, 3] = np.log1p(counts[c])
        if train[rows].all():
            observed = np.clip(np.abs(frame.actual.to_numpy(float)[rows, None]-parts[rows])/scale[rows, None], 0, 10)
            for j in range(3):
                sums[:, j] += np.bincount(c, weights=observed[:, j], minlength=len(ids))
            counts += np.bincount(c, minlength=len(ids))
            total += observed.sum(0)
            count += len(rows)
        elif train[rows].any():
            raise ValueError('Mixed origin split')
    return result


def make_design(frame, train, dataset, variant):
    x, parts, meta = design(frame, train, dataset)
    if variant in ('A3_global_errors', 'A4_no_shrink'):
        raw = modified_history(frame, train, parts, variant)
        mean, std = raw[train].mean(0), np.maximum(raw[train].std(0), 1e-8)
        x[:, -4:] = (raw-mean)/std
        meta['replacement_history'] = dict(mean=mean.tolist(), std=std.tolist(), variant=variant,
            support='per-series completed development rows; unchanged in A3',
            cold_start='global past mean, zero before first completed development origin')
    return x, parts, meta


def fit_variant(frame, train, dataset, variant):
    x, parts, meta = make_design(frame, train, dataset, variant)
    k = x.shape[1]
    active = np.arange(2*k)
    if variant == 'A5_constant_alpha':
        active = np.arange(k+1)
    if variant == 'A6_constant_beta':
        active = np.r_[0, np.arange(k, 2*k)]
    y = frame.actual.to_numpy(float)
    with threadpool_limits(limits=CONFIG['cpu_threads']):
        result = minimize(objective, np.zeros(len(active)), args=(x[train], y[train], parts[train], active),
            jac=True, method='L-BFGS-B',
            options=dict(maxiter=CONFIG['gate_maxiter'], ftol=CONFIG['gate_ftol'], gtol=CONFIG['gate_gtol']))
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f'{variant}: {result.message}')
    full = np.zeros(2*k)
    full[active] = result.x
    beta, alpha = expit(x[~train] @ full[:k]), expit(x[~train] @ full[k:])
    b, s, f = parts[~train].T
    out = frame.loc[~train, KEYS].copy()
    out['forecast'] = np.maximum(b+beta*(s+alpha*(f-s)-b), 0)
    check_prediction(out)
    return out, dict(parameters=full.tolist(), active_parameters=active.tolist(), parameter_count=len(active),
        design=meta, converged=True, iterations=int(result.nit), objective=float(result.fun),
        alpha_quantiles=np.quantile(alpha, [0, .1, .5, .9, 1]).tolist(),
        beta_quantiles=np.quantile(beta, [0, .1, .5, .9, 1]).tolist())


def verify_source(source):
    manifest = json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    spec = manifest['schedule']
    if {k: v for k, v in manifest['config'].items() if k != 'version'} != {k: v for k, v in CONFIG.items() if k != 'version'}:
        raise ValueError('Source configuration differs from configs/retail.json')
    if spec['dataset'] == 'm5':
        expected = pd.date_range('2015-05-25', periods=52, freq='7D').strftime('%Y-%m-%d').tolist()
        if spec['test'] != expected:
            raise ValueError('M5 source must cover the unified 52 weeks')
    for a, b in [('fit', 'dev'), ('dev', 'test')]:
        if pd.Timestamp(spec[a][-1])+pd.Timedelta(days=6) >= pd.Timestamp(spec[b][0]):
            raise ValueError('Overlapping splits')
    verified = []
    for stage in ('references', 'chronos', 'gates', 'score'):
        record = json.loads((source/(stage+'.complete.json')).read_text(encoding='utf-8'))
        for rel in record['artifacts']:
            p = (source/rel).resolve()
            if not p.is_relative_to(source.resolve()) or not p.is_file():
                raise ValueError(f'Missing or unsafe input: {rel}')
            verified.append(rel.replace('\\', '/'))
    required = ['gates_chronos/predictions.parquet']
    required += [f'{d}/{o}.parquet' for d in ('references', 'chronos') for o in spec['dev']+spec['test']]
    if not set(required) <= set(verified):
        raise ValueError('Completion markers omit required source artifacts')
    return spec, verified


def load_frame(source, refs, spec, data):
    chunks = []
    for origin in spec['dev']+spec['test']:
        ref = pd.read_parquet(refs/f'{origin}.parquet')
        b = pd.read_parquet(source/'chronos'/f'{origin}.parquet').rename(columns={'forecast': 'backbone_forecast'})
        expected = data.window(origin, labels=False)[-1]
        f = checked_merge(expected, checked_merge(ref, b))
        if not f.origin.eq(origin).all():
            raise ValueError('Wrong origin file')
        f = checked_merge(f, data.truth([origin])) if origin in spec['dev'] else f.assign(actual=np.nan)
        chunks.append(f)
    frame = pd.concat(chunks, ignore_index=True).sort_values(KEYS).reset_index(drop=True)
    return frame, frame.origin.isin(spec['dev']).to_numpy()


def rebuild_references(data, spec, dest, variant, seed):
    """Use the same past-origin library; only replace the factor reference selection."""
    from unittest.mock import patch
    import data_provider.retail_factors as factors
    from exp.retail_references import build
    select = factors.select
    for origin in spec['dev']+spec['test']:
        local_seed = int.from_bytes(hashlib.sha256(f'{seed}:{origin}'.encode()).digest()[:8], 'little')
        rng = np.random.default_rng(local_seed)

        def replacement(shape, promo, holiday):
            chosen, dist = select(shape, promo, holiday)
            chosen['factor'] = chosen['joint'] if variant == 'A1_joint' else np.vstack([
                rng.choice(shape.shape[1], 8, replace=False) for _ in range(shape.shape[0])])
            return chosen, dist
        # Keep the complete past-origin library when processing one query week.
        bank = spec['fit']+spec['dev']+spec['test']
        single = dict(fit=[o for o in bank if o < origin], dev=[], test=[origin])
        with patch.object(factors, 'select', replacement):
            build(data, single, dest)


def execute(source, out, variant, seed):
    source, out = source.resolve(), out.resolve()
    if out == source or out.is_relative_to(source):
        raise ValueError('Use a separate output directory')
    spec, verified = verify_source(source)
    identity = dict(source=str(source), inputs=sorted(verified), variant=variant, seed=seed, config=CONFIG)
    out.mkdir(parents=True, exist_ok=True)
    with (out/'RUNNING.lock').open('x') as lock:
        lock.write(str(os.getpid()))
    try:
        manifest = out/'manifest.json'
        if manifest.exists() and json.loads(manifest.read_text()) != identity:
            raise ValueError('Changed inputs/code; choose a new output directory')
        json_write(manifest, identity)
        done = out/'complete.json'
        if done.exists():
            for rel in json.loads(done.read_text())['artifacts']:
                if not (out/rel).is_file():
                    raise ValueError('Missing completed output')
            return
        started = time.time()
        data = RetailData(spec['dataset'])
        # Raw files and runtime must match the source run.
        from exp.retail import audit
        if not (out/'audit.json').exists():
            shutil.copy2(source/'audit.json', out/'audit.json')
        audit(data, spec, out)
        refs = source/'references'
        if variant in ('A1_joint', 'A2_random'):
            refs = out/'references'
            rebuild_references(data, spec, refs, variant, seed)
        frame, train = load_frame(source, refs, spec, data)
        prediction, fit = fit_variant(frame, train, spec['dataset'], variant)
        truth = data.truth(spec['test'])
        baseline = pd.read_parquet(source/'gates_chronos/predictions.parquet')[KEYS+['soft_history']]
        result = checked_merge(checked_merge(prediction, truth), baseline)
        overall = {name: metric(result.actual, result[col]) for name, col in [('variant', 'forecast'), ('full', 'soft_history')]}
        weekly = [dict(origin=o, variant=metric(p.actual, p.forecast), full=metric(p.actual, p.soft_history))
                  for o, p in result.groupby('origin', sort=True)]
        parquet_write(out/'predictions.parquet', prediction)
        json_write(out/'fit.json', fit)
        json_write(out/'metrics.json', dict(variant=variant, seed=seed, test_weeks=len(spec['test']), overall=overall,
            weekly=weekly, wape_difference=overall['variant']['wape']-overall['full']['wape'],
            elapsed_seconds=time.time()-started))
        artifacts = sorted(str(p.relative_to(out)) for p in out.rglob('*')
                           if p.is_file() and p.name not in ('RUNNING.lock', 'complete.json'))
        json_write(done, dict(artifacts=artifacts))
    finally:
        (out/'RUNNING.lock').unlink(missing_ok=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source', type=Path, required=True, help='completed exp.retail Chronos-2 run, e.g. results/m5/chronos/m5_full')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--variant', choices=VARIANTS, required=True)
    p.add_argument('--seed', type=int, default=20261005)
    a = p.parse_args()
    execute(a.source, a.out, a.variant, a.seed)
