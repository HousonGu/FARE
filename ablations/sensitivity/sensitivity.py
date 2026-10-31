"""One-at-a-time hyperparameter sensitivity of FARE's full two-gate fusion.

Reads the shared input folder from baselines/prepare_inputs.py (frozen
Chronos-2 B, references S/F, panel labels, main-model predictions and scaler). Backbones and references are never recomputed; for each
setting only the gate inputs and the two gates are rebuilt and refitted on
development labels, then scored once on the test rows.

Swept settings (others held at the paper values):
  lam  shrinkage pseudo-count in Eq. (9)            paper 14
  clip upper clip of errors and disagreements        paper 10
  eps  smoothing constant in sqrt(r^2 + eps^2)       paper 1
  reg  L2 weight on non-intercept gate parameters    paper 0.001
"""
from pathlib import Path
import argparse
import json
import os
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from threadpoolctl import threadpool_limits

KEYS = ['origin', 'series_id', 'target_date']
EXPECTED = {'favorita': (24, 25, 311850), 'm5': (52, 52, 11053322),
            'rossmann': (26, 26, 196630), 'rohlik': (26, 26, 283185)}
DEFAULT = dict(lam=14., clip=10., eps=1., reg=1e-3)
SWEEP = dict(lam=[0., 3.5, 7., 28., 56.], clip=[5., 20.], eps=[.1, 10.], reg=[1e-4, 1e-2])
OPTIONS = dict(maxiter=250, ftol=1e-9, gtol=1e-6)
CV_CLIP = 10.          # history_std / a stays clipped at 10 in every setting
BATCH = 262144
TAIL = ['backbone_shape_disagreement', 'factor_shape_disagreement',
        'hist_backbone_error', 'hist_shape_error', 'hist_factor_error', 'hist_log_support']
read_table = pd.read_parquet   # replaced by tests only


def configs():
    out = {'default': dict(DEFAULT)}
    for name, values in SWEEP.items():
        for value in values:
            setting = dict(DEFAULT); setting[name] = value
            out[f'{name}={value:g}'] = setting
    return out


# ----------------------------------------------------------------- I/O helpers
def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def canonical(frame):
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


def expected_frame(panel, values, origin):
    c = panel['days'].index(origin)
    if 'eligible_rows' in panel:
        rows = np.asarray(panel['eligible_rows'][origin], dtype=int)
    else:
        rows = np.flatnonzero(np.array(panel['first'], dtype='datetime64[D]') <= np.datetime64(origin) - np.timedelta64(56, 'D'))
    keys = pd.DataFrame(dict(origin=origin, series_id=np.repeat(np.asarray(panel['ids'], dtype=str)[rows], 7),
                             target_date=np.tile(panel['days'][c:c + 7], len(rows))))
    keys['actual'] = np.asarray(values[np.ix_(rows, np.arange(c, c + 7))], dtype=float).ravel()
    if not np.isfinite(keys.actual).all() or (keys.actual < 0).any():
        raise ValueError('Missing/invalid target')
    return keys


def base_features(frame, dataset):
    """Gate inputs that do not depend on the swept settings (same order as the paper model)."""
    scale = np.maximum(frame.history_mean.to_numpy(float), 1)
    names = ['log_history_mean', 'history_cv', 'history_zero_fraction']
    cols = [np.log1p(frame.history_mean), np.clip(frame.history_std / scale, 0, CV_CLIP), frame.history_zero_fraction]
    if dataset == 'favorita':
        names += ['log_promo_distance', 'log_holiday_distance', 'log_promo_bank', 'log_holiday_bank', 'promo_bin', 'holiday']
        cols += [np.log1p(frame.promo_distance), np.log1p(frame.holiday_distance),
                 np.log1p(frame.promo_bank_size), np.log1p(frame.holiday_bank_size),
                 np.minimum(frame.onpromotion, 6) / 6, frame.holiday_day]
        binary = ['holiday']
    elif dataset == 'm5':
        names += ['log_bank_size', 'snap', 'event']; cols += [np.log1p(frame.bank_size), frame.snap, frame.event]
        binary = ['snap', 'event']
    elif dataset == 'rossmann':
        names += ['log_bank_size', 'log_promo_distance', 'log_school_holiday_distance', 'promo', 'school_holiday']
        cols += [np.log1p(frame.bank_size), np.log1p(frame.promo_distance), np.log1p(frame.school_holiday_distance),
                 frame.promo, frame.school_holiday]
        binary = ['promo', 'school_holiday']
    else:
        names += ['log_bank_size', 'log_discount_distance', 'log_holiday_distance', 'discount', 'holiday']
        cols += [np.log1p(frame.bank_size), np.log1p(frame.discount_distance), np.log1p(frame.holiday_distance),
                 frame.discount, frame.holiday]
        binary = ['discount', 'holiday']
    raw = np.column_stack([np.asarray(c, dtype=float) for c in cols])
    if not np.isfinite(raw).all():
        raise ValueError('Nonfinite gate input')
    return raw, names, binary


# --------------------------------------------------------------- cached panel
def build_cache(inputs, dataset, cache):
    """Merge B/S/F, labels and setting-free features once per dataset."""
    inputs = Path(inputs).resolve(); cache = Path(cache); source = inputs / dataset
    if (cache / 'cache.json').exists():
        return json.loads((cache / 'cache.json').read_text(encoding='utf-8'))
    # The shared input folder lists every file in complete.json.
    manifest = json.loads((source / 'complete.json').read_text(encoding='utf-8'))['artifacts']
    for rel in manifest:
        if not (source / rel).is_file():
            raise ValueError('Missing input file: ' + rel)
    panel = json.loads((source / 'panel.json').read_text(encoding='utf-8'))
    values = np.load(source / 'values.npy', mmap_mode='r'); schedule = panel['schedule']
    nd, nt, expected_rows = EXPECTED[dataset]
    if len(schedule['dev']) != nd or len(schedule['test']) != nt:
        raise ValueError('Wrong dev/test schedule')
    origins = [(o, 'dev') for o in schedule['dev']] + [(o, 'test') for o in schedule['test']]
    days = [o for o, _ in origins]
    if any(pd.Timestamp(a) + pd.Timedelta(days=6) >= pd.Timestamp(b) for a, b in zip(days, days[1:])):
        raise ValueError('Overlapping forecast target weeks')
    sizes = [len(expected_frame(panel, values, o)) for o in days]
    test_rows = sum(n for n, (_, s) in zip(sizes, origins) if s == 'test')
    if test_rows != expected_rows:
        raise ValueError(f'Wrong test row count {test_rows} != {expected_rows}')
    n = sum(sizes); n_test = test_rows
    cache.mkdir(parents=True, exist_ok=True)
    index = {str(v): i for i, v in enumerate(panel['ids'])}
    arrays = names = binary = None; weeks = []; offset = 0; test_offset = 0
    for (origin, split), size in zip(origins, sizes):
        frame = exact_merge(canonical(read_table(source / 'references' / f'{origin}.parquet')),
                            canonical(read_table(source / 'backbone' / f'{origin}.parquet'))
                            .rename(columns={'forecast': 'backbone_forecast'})[KEYS + ['backbone_forecast']])
        frame = exact_merge(frame, expected_frame(panel, values, origin))
        frame = frame.sort_values(KEYS).reset_index(drop=True)
        if len(frame) != size:
            raise ValueError('Coverage differs from expected rows at ' + origin)
        parts = np.maximum(frame[['backbone_forecast', 'shape_retrieval', 'factor_retrieval']].to_numpy(float), 0)
        raw, names, binary = base_features(frame, dataset)
        if arrays is None:
            def mm(name, shape, dtype='float64'):
                return np.lib.format.open_memmap(cache / f'{name}.npy', mode='w+', dtype=dtype, shape=shape)
            arrays = dict(base=mm('base', (n, raw.shape[1])), parts=mm('parts', (n, 3)), y=mm('y', (n,)),
                          scale=mm('scale', (n,)), codes=mm('codes', (n,), 'int64'),
                          existing=mm('existing_full', (n_test,)))
        sl = slice(offset, offset + size)
        arrays['base'][sl] = raw; arrays['parts'][sl] = parts
        arrays['y'][sl] = frame.actual.to_numpy(float)
        arrays['scale'][sl] = np.maximum(frame.history_mean.to_numpy(float), 1)
        arrays['codes'][sl] = [index[v] for v in frame.series_id]
        if split == 'test':
            old = exact_merge(frame[KEYS], canonical(read_table(source / 'fare' / f'{origin}.parquet'))).sort_values(KEYS)
            if not np.allclose(old.backbone.to_numpy(float), parts[:, 0], rtol=1e-8, atol=1e-8):
                raise ValueError('Existing backbone differs from cached candidate B')
            arrays['existing'][test_offset:test_offset + size] = old.soft_history.to_numpy(float)
            test_offset += size
        weeks.append(dict(origin=origin, split=split, start=offset, stop=offset + size,
                          last_target=pd.to_datetime(frame.target_date).max().strftime('%Y-%m-%d')))
        offset += size
        print('cached', dataset, split, origin, size, flush=True)
    for array in arrays.values():
        array.flush()
    saved = None
    if (source / 'fare' / 'fit.json').exists():
        design = json.loads((source / 'fare' / 'fit.json').read_text(encoding='utf-8'))['design']
        if design['names'] != names + TAIL:
            raise ValueError('Existing model feature names differ')
        saved = dict(mean=design['mean'], std=design['std'])
    meta = dict(dataset=dataset, rows=n, test_rows=n_test, series=len(index), names=names + TAIL,
                binary=binary, weeks=weeks, saved_scaler=saved)
    write_json(cache / 'cache.json', meta)
    return meta


def load(cache):
    cache = Path(cache)
    return {k: np.load(cache / f'{k}.npy', mmap_mode='r')
            for k in ('base', 'parts', 'y', 'scale', 'codes', 'existing_full')}


# ------------------------------------------------------------ setting-dependent
def history_features(meta, data, lam, clip):
    """Shrunk forward errors (Eq. 8-9); test labels never update the state."""
    out = np.empty((meta['rows'], 4))
    sums = np.zeros((meta['series'], 3)); counts = np.zeros(meta['series'])
    total = np.zeros(3); count = 0; latest = None
    for week in meta['weeks']:
        if latest is not None and latest >= pd.Timestamp(week['origin']):
            raise ValueError('A historical target reaches the current origin')
        sl = slice(week['start'], week['stop']); c = np.asarray(data['codes'][sl])
        prior = total / count if count else np.zeros(3)
        if lam > 0:
            errors = (sums[c] + lam * prior) / (counts[c, None] + lam)
        else:   # unshrunk: series mean when available, past global mean for cold starts
            seen = counts[c, None] > 0
            errors = np.where(seen, sums[c] / np.maximum(counts[c, None], 1), prior)
        out[sl, :3] = np.clip(errors, 0, clip); out[sl, 3] = np.log1p(counts[c])
        if week['split'] == 'dev':
            parts = np.asarray(data['parts'][sl]); y = np.asarray(data['y'][sl])
            observed = np.clip(np.abs(y[:, None] - parts) / np.asarray(data['scale'][sl])[:, None], 0, clip)
            for j in range(3):
                sums[:, j] += np.bincount(c, weights=observed[:, j], minlength=len(counts))
            counts += np.bincount(c, minlength=len(counts))
            total += observed.sum(0); count += len(y)
            latest = pd.Timestamp(week['last_target'])
    return out


def design(meta, data, hist, clip, work, scaler=None):
    """Intercept + standardized inputs; scaler from development rows unless given."""
    work = Path(work); work.mkdir(parents=True, exist_ok=True)
    n_dev = sum(w['stop'] - w['start'] for w in meta['weeks'] if w['split'] == 'dev')
    width = len(meta['names']) + 1
    x = np.lib.format.open_memmap(work / 'x.npy', mode='w+', dtype='float64', shape=(meta['rows'], width))
    continuous = np.array([i for i, name in enumerate(meta['names']) if name not in meta['binary']])
    total = np.zeros(len(continuous)); square = np.zeros(len(continuous))
    for start in range(0, meta['rows'], BATCH):
        stop = min(start + BATCH, meta['rows'])
        parts = np.asarray(data['parts'][start:stop]); scale = np.asarray(data['scale'][start:stop])[:, None]
        disagreement = np.clip(np.abs(parts[:, [0, 2]] - parts[:, [1, 1]]) / scale, 0, clip)
        raw = np.column_stack([np.asarray(data['base'][start:stop]), disagreement, hist[start:stop]])
        x[start:stop, 0] = 1.; x[start:stop, 1:] = raw
        dev_stop = min(stop, n_dev)
        if start < dev_stop:
            block = raw[:dev_stop - start, continuous]
            total += block.sum(0); square += np.square(block).sum(0)
    mean = total / n_dev; std = np.maximum(np.sqrt(np.maximum(square / n_dev - mean ** 2, 0)), 1e-8)
    if scaler is not None:
        mean, std = np.asarray(scaler['mean']), np.asarray(scaler['std'])
    for start in range(0, meta['rows'], BATCH):
        stop = min(start + BATCH, meta['rows'])
        x[start:stop, continuous + 1] = (x[start:stop, continuous + 1] - mean) / std
    x.flush()
    if not np.isfinite(np.load(work / 'x.npy', mmap_mode='r')[::max(1, meta['rows'] // 1000)]).all():
        raise ValueError('Nonfinite design')
    return np.load(work / 'x.npy', mmap_mode='r'), n_dev, dict(mean=mean.tolist(), std=std.tolist())


def objective(theta, x, parts, y, eps, reg, batch=BATCH):
    """Eq. (14) with smoothing constant eps and L2 weight reg; theta = [beta | alpha]."""
    k = x.shape[1]; denominator = float(np.sum(y))
    value = 0.; gradient = np.zeros_like(theta)
    for start in range(0, len(y), batch):
        stop = min(start + batch, len(y))
        xx = np.asarray(x[start:stop]); pp = np.asarray(parts[start:stop]); yy = np.asarray(y[start:stop])
        beta = expit(xx @ theta[:k]); alpha = expit(xx @ theta[k:])
        b, s, f = pp.T; reference = s + alpha * (f - s); pred = b + beta * (reference - b)
        residual = pred - yy; smooth = np.sqrt(residual * residual + eps * eps)
        g = residual / smooth / denominator
        value += float(smooth.sum()) / denominator
        gradient[:k] += xx.T @ (g * beta * (1 - beta) * (reference - b))
        gradient[k:] += xx.T @ (g * beta * alpha * (1 - alpha) * (f - s))
    value += reg * (theta[1:k] @ theta[1:k] + theta[k + 1:] @ theta[k + 1:])
    gradient[1:k] += 2 * reg * theta[1:k]; gradient[k + 1:] += 2 * reg * theta[k + 1:]
    return value, gradient


def predict(theta, x, parts, batch=BATCH):
    k = x.shape[1]; n = len(parts)
    pred = np.empty(n); alpha = np.empty(n); beta = np.empty(n)
    for start in range(0, n, batch):
        stop = min(start + batch, n); xx = np.asarray(x[start:stop]); pp = np.asarray(parts[start:stop])
        beta[start:stop] = expit(xx @ theta[:k]); alpha[start:stop] = expit(xx @ theta[k:])
        b, s, f = pp.T
        pred[start:stop] = np.maximum(b + beta[start:stop] * (s + alpha[start:stop] * (f - s) - b), 0)
    return pred, alpha, beta


def wape(y, pred):
    return float(np.abs(np.asarray(y) - pred).sum() / np.sum(y))


def run_setting(meta, data, setting, work, threads, scaler=None):
    started = time.time()
    hist = history_features(meta, data, setting['lam'], setting['clip'])
    x, n_dev, used = design(meta, data, hist, setting['clip'], work, scaler)
    del hist
    with threadpool_limits(limits=threads):
        result = minimize(objective, np.zeros(2 * x.shape[1]),
                          args=(x[:n_dev], data['parts'][:n_dev], data['y'][:n_dev], setting['eps'], setting['reg']),
                          jac=True, method='L-BFGS-B', options=OPTIONS)
        theta = result.x
        dev_pred, _, _ = predict(theta, x[:n_dev], data['parts'][:n_dev])
        test_pred, alpha, beta = predict(theta, x[n_dev:], data['parts'][n_dev:])
    y_test = np.asarray(data['y'][n_dev:]); existing = np.asarray(data['existing_full'])
    weekly = []
    for week in meta['weeks']:
        if week['split'] != 'test':
            continue
        a, b = week['start'] - n_dev, week['stop'] - n_dev
        weekly.append(dict(origin=week['origin'], rows=b - a, wape=wape(y_test[a:b], test_pred[a:b]),
                           existing_full_wape=wape(y_test[a:b], existing[a:b])))
    abs_error = float(np.abs(y_test - test_pred).sum())
    return dict(dataset=meta['dataset'], setting=setting, converged=bool(result.success),
                optimizer_message=str(result.message), iterations=int(result.nit), objective=float(result.fun),
                parameters=theta.tolist(), dev_rows=n_dev, test_rows=len(y_test),
                dev_wape=wape(data['y'][:n_dev], dev_pred), test_wape=abs_error / float(y_test.sum()),
                test_mae=abs_error / len(y_test),
                backbone_test_wape=wape(y_test, np.asarray(data['parts'][n_dev:, 0])),
                existing_full_test_wape=wape(y_test, existing),
                max_abs_diff_vs_existing_full=float(np.max(np.abs(test_pred - existing))),
                mean_test_alpha=float(alpha.mean()), mean_test_beta=float(beta.mean()),
                scaler=('saved existing full-model scaler' if scaler is not None else 'development rows'),
                scaler_values=used, elapsed_seconds=time.time() - started), weekly


def run(inputs, dataset, out, threads=4, only=None):
    out = Path(out); root = out / dataset
    meta = build_cache(inputs, dataset, root / 'cache'); data = load(root / 'cache')
    todo = configs()
    if meta['saved_scaler'] is not None:   # reproduction check only, not part of the sweep
        todo['default@saved_scaler'] = dict(DEFAULT)
    for name, setting in todo.items():
        if only and name not in only:
            continue
        dest = root / name
        if (dest / 'result.json').exists():
            print('SKIP', dataset, name, flush=True); continue
        scaler = meta['saved_scaler'] if name == 'default@saved_scaler' else None
        result, weekly = run_setting(meta, data, setting, root / 'work', threads, scaler)
        result['name'] = name
        dest.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(weekly).to_csv(dest / 'weekly.csv', index=False)
        write_json(dest / 'result.json', result)
        print(f"DONE {dataset} {name} test_wape={100 * result['test_wape']:.4f}% "
              f"converged={result['converged']} {result['elapsed_seconds']:.0f}s", flush=True)
    work = root / 'work' / 'x.npy'
    if work.exists():
        work.unlink()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--inputs', type=Path, default=Path(__file__).resolve().parents[2] / 'data' / 'baselines',
                   help='Folder produced by baselines/prepare_inputs.py, one subfolder per dataset')
    p.add_argument('--dataset', choices=list(EXPECTED), required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--only', nargs='+', help='subset of setting names, e.g. default lam=0')
    a = p.parse_args()
    run(a.inputs, a.dataset, a.out, a.threads, a.only)
