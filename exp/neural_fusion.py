"""Two-level FARE gate for PatchTST/SparseTSF backbones with bounded memory.

Feature definitions and objective match the retail implementation. This driver
streams weeks and blocks the objective to avoid concatenating M5 DataFrames.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from threadpoolctl import threadpool_limits
from data_provider.panel import EXPECTED
from exp.io import KEYS, merge, metric, pool, save_parquet, write_json, seal, verify, verify_file


class History:
    def __init__(self, panel):
        self.panel = panel
        self.sums = np.zeros((len(panel.ids), 3), np.float64)
        self.counts = np.zeros(len(panel.ids), np.float64)
        self.global_sums = np.zeros(3, np.float64)
        self.global_count = 0
        self.latest = None

    def features(self, frame):
        if self.latest is not None and self.latest >= frame.origin.iloc[0]:
            raise ValueError('Historical target overlaps current forecast origin')
        codes = frame.series_id.map(self.panel.id_rows).to_numpy(dtype=np.int64)
        prior = self.global_sums/self.global_count if self.global_count else np.zeros(3)
        errors = (self.sums[codes]+14*prior)/(self.counts[codes, None]+14)
        return np.column_stack([np.clip(errors, 0, 10), np.log1p(self.counts[codes])])

    def update(self, frame, parts):
        origin = frame.origin.iloc[0]
        if origin not in self.panel.schedule['dev']:
            raise ValueError('Only completed development labels may update gate error state')
        codes = frame.series_id.map(self.panel.id_rows).to_numpy(dtype=np.int64)
        actual = frame.actual.to_numpy(np.float64)
        errors = np.clip(np.abs(actual[:, None]-parts)/np.maximum(frame.history_mean.to_numpy()[:, None], 1), 0, 10)
        if not np.isfinite(errors).all():
            raise ValueError('Invalid development error')
        for j in range(3):
            self.sums[:, j] += np.bincount(codes, weights=errors[:, j], minlength=len(self.panel.ids))
        self.counts += np.bincount(codes, minlength=len(self.panel.ids))
        self.global_sums += errors.sum(axis=0)
        self.global_count += len(frame)
        self.latest = frame.target_date.max()

    def record(self):
        return dict(sums=self.sums.tolist(), counts=self.counts.tolist(), global_sums=self.global_sums.tolist(),
                    global_count=self.global_count, latest=self.latest, shrink_rows=14)

    def restore(self, record):
        self.sums = np.asarray(record['sums'], np.float64)
        self.counts = np.asarray(record['counts'], np.float64)
        self.global_sums = np.asarray(record['global_sums'], np.float64)
        self.global_count, self.latest = record['global_count'], record['latest']


def base_features(frame, dataset):
    scale = np.maximum(frame.history_mean.to_numpy(), 1)
    names = ['log_history_mean', 'history_cv', 'history_zero_fraction']
    values = [np.log1p(frame.history_mean), np.clip(frame.history_std/scale, 0, 10), frame.history_zero_fraction]
    if dataset == 'favorita':
        extra = ['log_promo_distance', 'log_holiday_distance', 'log_promo_bank', 'log_holiday_bank', 'promo_bin', 'holiday']
        values += [np.log1p(frame.promo_distance), np.log1p(frame.holiday_distance),
                   np.log1p(frame.promo_bank_size), np.log1p(frame.holiday_bank_size),
                   np.minimum(frame.onpromotion.to_numpy(dtype=np.float64), 6)/6, frame.holiday_day]
        factors = set(extra)
        binary = {'holiday'}
    elif dataset == 'm5':
        extra = ['log_bank_size', 'snap', 'event']
        values += [np.log1p(frame.bank_size), frame.snap, frame.event]
        factors, binary = {'snap', 'event'}, {'snap', 'event'}
    else:
        one, two = ('promo', 'school_holiday') if dataset == 'rossmann' else ('discount', 'holiday')
        extra = ['log_bank_size', 'log_'+one+'_distance', 'log_'+two+'_distance', one, two]
        values += [np.log1p(frame.bank_size), np.log1p(frame[one+'_distance']),
                   np.log1p(frame[two+'_distance']), frame[one], frame[two]]
        factors, binary = set(extra[1:]), {one, two}
    names += extra
    names += ['backbone_shape_disagreement', 'factor_shape_disagreement']
    values += [np.clip(np.abs(frame.backbone_forecast-frame.shape_retrieval)/scale, 0, 10),
               np.clip(np.abs(frame.factor_retrieval-frame.shape_retrieval)/scale, 0, 10)]
    factors.add('factor_shape_disagreement')
    raw = np.column_stack(values).astype(np.float64)
    if not np.isfinite(raw).all():
        raise ValueError('Nonfinite gate features')
    return raw, names, factors, binary


def load_week(panel, backbone, origin, labels=False):
    path = backbone/'predictions'/f'{origin}.parquet'
    verify_file(path)
    frame = merge(panel.reference(origin), pd.read_parquet(path).rename(columns={'forecast': 'backbone_forecast'}))
    if labels:
        if origin not in panel.schedule['dev']:
            raise ValueError('Fusion fitting may read only development labels')
        frame = merge(frame, panel.truth(origin))
    parts = np.maximum(frame[['backbone_forecast', 'shape_retrieval', 'factor_retrieval']].to_numpy(np.float64), 0)
    if not np.isfinite(parts).all():
        raise ValueError('Invalid candidate predictions')
    return frame, parts


def build_design(panel, backbone, cache):
    marker = verify(cache)
    history = History(panel)
    if marker:
        meta = json.loads((cache/'design.json').read_text(encoding='utf-8'))
        history.restore(meta['history'])
        return meta, history
    cache.mkdir(parents=True, exist_ok=True)
    n = sum(len(panel.rows(o))*7 for o in panel.schedule['dev'])
    position = 0
    mean = m2 = None
    count = 0
    for origin in panel.schedule['dev']:
        frame, parts = load_week(panel, backbone, origin, labels=True)
        raw, names, factors, binary = base_features(frame, panel.dataset)
        raw = np.column_stack([raw, history.features(frame)])
        names += ['hist_backbone_error', 'hist_shape_error', 'hist_factor_error', 'hist_log_support']
        if position == 0:
            x = np.lib.format.open_memmap(cache/'raw.npy', mode='w+', dtype=np.float64, shape=(n, raw.shape[1]))
            p = np.lib.format.open_memmap(cache/'parts.npy', mode='w+', dtype=np.float64, shape=(n, 3))
            y = np.lib.format.open_memmap(cache/'actual.npy', mode='w+', dtype=np.float64, shape=(n,))
            mean, m2 = np.zeros(raw.shape[1]), np.zeros(raw.shape[1])
        size = len(frame)
        x[position:position+size], p[position:position+size], y[position:position+size] = raw, parts, frame.actual
        # Parallel-variance formula avoids cancellation for nearly constant columns.
        chunk_mean = raw.mean(axis=0)
        chunk_m2 = np.square(raw-chunk_mean).sum(axis=0)
        delta = chunk_mean-mean
        m2 += chunk_m2+delta*delta*count*size/(count+size)
        mean += delta*size/(count+size)
        count += size
        history.update(frame, parts)
        position += size
    if position != n:
        raise ValueError('Development key count changed')
    x.flush(); p.flush(); y.flush()
    del x, p, y
    std = np.maximum(np.sqrt(m2/count), 1e-8)
    continuous = [i for i, name in enumerate(names) if name not in binary]
    meta = dict(names=names, factors=sorted(factors), continuous=continuous, mean=mean.tolist(),
                std=std.tolist(), history=history.record(), development_rows=n)
    write_json(cache/'design.json', meta)
    seal(cache, [cache/n for n in ['raw.npy', 'parts.npy', 'actual.npy', 'design.json']])
    return meta, history


def matrix(raw, meta, keep):
    normalized = np.array(raw, np.float64, copy=True)
    columns = meta['continuous']
    normalized[:, columns] = (normalized[:, columns]-np.asarray(meta['mean'])[columns])/np.asarray(meta['std'])[columns]
    return np.column_stack([np.ones(len(normalized)), normalized[:, keep]])


def objective(theta, raw, actual, parts, meta, keep, single, config):
    k = len(keep)+1
    denominator = float(actual.sum())
    total = 0.0
    gradient = np.zeros_like(theta)
    for start in range(0, len(raw), config['gate_block_rows']):
        end = start+config['gate_block_rows']
        x = matrix(raw[start:end], meta, keep)
        b, s, f = parts[start:end].T
        beta = expit(x@theta[:k])
        if single:
            pred = b+beta*(s-b)
        else:
            alpha = expit(x@theta[k:])
            reference = s+alpha*(f-s)
            pred = b+beta*(reference-b)
        residual = pred-actual[start:end]
        smooth = np.sqrt(residual*residual+1)
        g = residual/smooth/denominator
        total += float(smooth.sum())/denominator
        if single:
            gradient += x.T@(g*beta*(1-beta)*(s-b))
        else:
            gradient[:k] += x.T@(g*beta*(1-beta)*(reference-b))
            gradient[k:] += x.T@(g*beta*alpha*(1-alpha)*(f-s))
    penalized = np.ones(len(theta), bool)
    penalized[0] = False
    if not single:
        penalized[k] = False
    total += config['gate_penalty']*np.square(theta[penalized]).sum()
    gradient[penalized] += 2*config['gate_penalty']*theta[penalized]
    return float(total), gradient


def fit_variant(raw, actual, parts, meta, variant, config):
    names = meta['names']
    drop = set()
    single = variant == 'with_shape_only'
    if variant == 'with_no_history':
        drop.update(names[-4:])
    elif single:
        drop.update(meta['factors'])
        drop.add('hist_factor_error')
    keep = [i for i, name in enumerate(names) if name not in drop]
    count = (len(keep)+1)*(1 if single else 2)
    theta = np.zeros(count)
    attempts, iterations = [], 0
    for attempt in range(config['gate_max_attempts']):
        result = minimize(objective, theta, args=(raw, actual, parts, meta, keep, single, config),
                          jac=True, method='L-BFGS-B', options=dict(maxiter=config['gate_maxiter'],
                          ftol=config['gate_ftol'], gtol=config['gate_gtol']))
        iterations += int(result.nit)
        attempts.append(dict(attempt=attempt+1, iterations=int(result.nit), message=str(result.message), objective=float(result.fun)))
        if result.success or not np.isfinite(result.x).all():
            break
        # A predeclared continuation of the same objective, never a test-based restart.
        theta = result.x.copy()
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError('Gate convergence failed: '+str(result.message))
    return dict(parameters=result.x.tolist(), feature_indices=keep, single=single,
                feature_names=[names[i] for i in keep], parameter_count=count,
                iterations=iterations, attempts=attempts, objective=float(result.fun), converged=True)


def apply_variant(raw, parts, meta, fitted):
    x = matrix(raw, meta, fitted['feature_indices'])
    theta = np.asarray(fitted['parameters'])
    k = x.shape[1]
    beta = expit(x@theta[:k])
    alpha = np.zeros(len(x)) if fitted['single'] else expit(x@theta[k:])
    reference = parts[:, 1]+alpha*(parts[:, 2]-parts[:, 1])
    return np.maximum(parts[:, 0]+beta*(reference-parts[:, 0]), 0), alpha, beta


def run_fusion(panel, backbone, destination, kind, seed, config):
    existing = verify(destination)
    if existing:
        return json.loads((destination/'metrics.json').read_text(encoding='utf-8'))
    destination.mkdir(parents=True, exist_ok=True)
    meta, history = build_design(panel, backbone, destination/'cache')
    fits_path = destination/'fit.json'
    if fits_path.exists():
        verify_file(fits_path)
        fitted = json.loads(fits_path.read_text(encoding='utf-8'))
    else:
        raw = np.load(destination/'cache/raw.npy', mmap_mode='r')
        actual = np.load(destination/'cache/actual.npy', mmap_mode='r')
        parts = np.load(destination/'cache/parts.npy', mmap_mode='r')
        fitted = dict(design=meta, variants={})
        with threadpool_limits(limits=config['cpu_threads']):
            for variant in config['variants'][1:]:
                print('fit gate', panel.dataset, kind, seed, variant, flush=True)
                fitted['variants'][variant] = fit_variant(raw, actual, parts, meta, variant, config)
        write_json(fits_path, fitted)
    weekly = {}
    for origin in panel.schedule['test']:
        path = destination/'predictions'/f'{origin}.parquet'
        if path.exists():
            verify_file(path)
            scored = merge(panel.truth(origin), pd.read_parquet(path))
        else:
            frame, parts = load_week(panel, backbone, origin)
            raw = np.column_stack([base_features(frame, panel.dataset)[0], history.features(frame)])
            prediction = frame[KEYS].copy()
            prediction['without'] = parts[:, 0]
            for variant, fit in fitted['variants'].items():
                values, alpha, beta = apply_variant(raw, parts, meta, fit)
                prediction[variant] = values
                if variant == 'with':
                    prediction['alpha'], prediction['beta'] = alpha, beta
            save_parquet(path, prediction)
            # Test truth is accessed only after every forecast has been calculated.
            scored = merge(panel.truth(origin), prediction)
        weekly[origin] = {variant: metric(scored.actual, scored[variant]) for variant in config['variants']}
    overall = {variant: pool([v[variant] for v in weekly.values()]) for variant in config['variants']}
    if (len(weekly), overall['without']['rows']) != getattr(panel, 'expected', EXPECTED[panel.dataset]):
        raise ValueError('Formal evaluation range changed')
    result = dict(status='complete', dataset=panel.dataset, backbone=kind, seed=seed,
                  overall=overall, weekly=weekly)
    write_json(destination/'metrics.json', result)
    files = [fits_path, destination/'metrics.json']
    files += list((destination/'predictions').glob('*.parquet'))
    seal(destination, files, dict(status='complete', dataset=panel.dataset, backbone=kind, seed=seed))
    return result
