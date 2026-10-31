"""Read stored B/S/F and panel labels; prepare bounded-memory gate inputs, never fit."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from io_utils import KEYS, canonical, exact_merge, seal, verify, write_json

EXPECTED = {'favorita': (24, 25, 311850), 'm5': (52, 52, 11053322),
            'rossmann': (26, 26, 196630), 'rohlik': (26, 26, 283185)}


class History:
    def __init__(self, ids):
        self.index = {str(v): i for i, v in enumerate(ids)}
        self.sums = np.zeros((len(ids), 3)); self.counts = np.zeros(len(ids))
        self.total = np.zeros(3); self.count = 0; self.latest = None

    def features(self, frame, parts, y, development):
        origin = pd.Timestamp(frame.origin.iloc[0])
        if self.latest is not None and self.latest >= origin:
            raise ValueError('A historical target reaches the current origin')
        codes = np.array([self.index[v] for v in frame.series_id])
        prior = self.total / self.count if self.count else np.zeros(3)
        errors = (self.sums[codes] + 14. * prior) / (self.counts[codes, None] + 14.)
        out = np.column_stack([np.clip(errors, 0, 10), np.log1p(self.counts[codes])])
        if development:
            observed = np.clip(np.abs(y[:, None] - parts) / np.maximum(frame.history_mean.to_numpy()[:, None], 1), 0, 10)
            for j in range(3):
                self.sums[:, j] += np.bincount(codes, weights=observed[:, j], minlength=len(self.counts))
            self.counts += np.bincount(codes, minlength=len(self.counts))
            self.total += observed.sum(0); self.count += len(y)
            self.latest = pd.to_datetime(frame.target_date).max()
        return out


def raw_features(frame, parts, history, dataset):
    scale = np.maximum(frame.history_mean.to_numpy(float), 1)
    names = ['log_history_mean', 'history_cv', 'history_zero_fraction']
    cols = [np.log1p(frame.history_mean), np.clip(frame.history_std / scale, 0, 10), frame.history_zero_fraction]
    if dataset == 'favorita':
        names += ['log_promo_distance', 'log_holiday_distance', 'log_promo_bank', 'log_holiday_bank', 'promo_bin', 'holiday']
        cols += [np.log1p(frame.promo_distance), np.log1p(frame.holiday_distance),
                 np.log1p(frame.promo_bank_size), np.log1p(frame.holiday_bank_size),
                 np.minimum(frame.onpromotion, 6) / 6, frame.holiday_day]
        binary = {'holiday'}
    elif dataset == 'm5':
        names += ['log_bank_size', 'snap', 'event']; cols += [np.log1p(frame.bank_size), frame.snap, frame.event]
        binary = {'snap', 'event'}
    elif dataset == 'rossmann':
        names += ['log_bank_size', 'log_promo_distance', 'log_school_holiday_distance', 'promo', 'school_holiday']
        cols += [np.log1p(frame.bank_size), np.log1p(frame.promo_distance), np.log1p(frame.school_holiday_distance), frame.promo, frame.school_holiday]
        binary = {'promo', 'school_holiday'}
    else:
        names += ['log_bank_size', 'log_discount_distance', 'log_holiday_distance', 'discount', 'holiday']
        cols += [np.log1p(frame.bank_size), np.log1p(frame.discount_distance), np.log1p(frame.holiday_distance), frame.discount, frame.holiday]
        binary = {'discount', 'holiday'}
    names += ['backbone_shape_disagreement', 'factor_shape_disagreement',
              'hist_backbone_error', 'hist_shape_error', 'hist_factor_error', 'hist_log_support']
    cols += [np.clip(np.abs(parts[:, 0] - parts[:, 1]) / scale, 0, 10),
             np.clip(np.abs(parts[:, 2] - parts[:, 1]) / scale, 0, 10)]
    raw = np.column_stack(cols + [history])
    if not np.isfinite(raw).all():
        raise ValueError('Nonfinite gate input')
    return raw, names, [i for i, name in enumerate(names) if name not in binary]


def expected_frame(panel, values, origin):
    c = panel['days'].index(origin)
    if 'eligible_rows' in panel:
        rows = np.asarray(panel['eligible_rows'][origin], dtype=int)
    else:
        rows = np.flatnonzero(np.array(panel['first'], dtype='datetime64[D]') <= np.datetime64(origin) - np.timedelta64(56, 'D'))
    keys = pd.DataFrame(dict(origin=origin, series_id=np.repeat(np.asarray(panel['ids'], dtype=str)[rows], 7),
                             target_date=np.tile(panel['days'][c:c+7], len(rows))))
    keys['actual'] = np.asarray(values[np.ix_(rows, np.arange(c, c+7))], dtype=float).ravel()
    if not np.isfinite(keys.actual).all() or (keys.actual < 0).any():
        raise ValueError('Missing/invalid target; evaluation keys must not be silently filtered')
    return keys


def prepare(inputs, dataset, out, profile_series=None):
    inputs = Path(inputs).resolve(); out = Path(out).resolve(); source = inputs / dataset
    if out == source or out.is_relative_to(source):
        raise ValueError('Prepared output must be outside the read-only input folder')
    # The shared input folder lists every file in complete.json.
    manifest = json.loads((source / 'complete.json').read_text(encoding='utf-8'))['artifacts']
    for rel in manifest:
        p = (source / rel).resolve()
        if not p.is_relative_to(source) or not p.is_file():
            raise ValueError('Missing input file: ' + rel)
    for required in ('fare/fit.json',):
        if required not in manifest:
            raise ValueError('Prepare the inputs with --fare-predictions and --fare-fit first')
    identity = dict(dataset=dataset, inputs=str(source),
                    profile_series=profile_series, protocol='candidate-paths-v1')
    if (out / 'complete.json').exists():
        if verify(out) != identity:
            raise ValueError('Completed preparation identity differs; use another output directory')
        print('SKIP prepared inputs already complete', flush=True); return
    if out.exists():
        raise ValueError('Partial preparation exists; retain it and use a new output directory')
    panel = json.loads((source / 'panel.json').read_text(encoding='utf-8'))
    values = np.load(source / 'values.npy', mmap_mode='r'); schedule = panel['schedule']
    nd, nt, expected_rows = EXPECTED[dataset]
    if len(schedule['dev']) != nd or len(schedule['test']) != nt:
        raise ValueError('Wrong source schedule (in particular, M5 must have 52 test origins)')
    origins = schedule['dev'] + schedule['test']
    if any(pd.Timestamp(a) + pd.Timedelta(days=6) >= pd.Timestamp(b) for a, b in zip(origins, origins[1:])):
        raise ValueError('Overlapping forecast target weeks')
    chosen = set(map(str, panel['ids'][:profile_series])) if profile_series else None
    sizes = {}
    for origin in origins:
        frame = expected_frame(panel, values, origin)
        sizes[origin] = len(frame if chosen is None else frame[frame.series_id.isin(chosen)])
    if not profile_series and sum(sizes[o] for o in schedule['test']) != expected_rows:
        raise ValueError('Wrong full test row count')
    out.mkdir(parents=True); arrays = {}; info = {}; history = History(panel['ids'])
    scaler_sum = scaler_square = None; continuous = names = None
    for split in ('dev', 'test'):
        n = sum(sizes[o] for o in schedule[split]); offset = 0; weeks = []; key_chunks = []
        fields = [('parts', 3), ('y', 0)] + ([('baseline', 0)] if split == 'test' else [])
        arrays[split] = {
            name: np.lib.format.open_memmap(out / f'{name}_{split}.npy', mode='w+', dtype='float64',
                                            shape=(n, width) if width else (n,))
            for name, width in fields}
        for origin in schedule[split]:
            frame = exact_merge(canonical(pd.read_parquet(source / 'references' / f'{origin}.parquet')),
                                canonical(pd.read_parquet(source / 'backbone' / f'{origin}.parquet')).rename(columns={'forecast': 'backbone_forecast'})[KEYS + ['backbone_forecast']])
            frame = exact_merge(frame, expected_frame(panel, values, origin))
            if chosen is not None:
                frame = frame[frame.series_id.isin(chosen)]
            frame = frame.sort_values(KEYS).reset_index(drop=True)
            if frame.empty:
                continue
            parts = np.maximum(frame[['backbone_forecast', 'shape_retrieval', 'factor_retrieval']].to_numpy(float), 0)
            y = frame.actual.to_numpy(float)
            # Held-out targets are never passed to the historical feature accumulator.
            hist = history.features(frame, parts, y if split == 'dev' else None, split == 'dev')
            raw, names, continuous = raw_features(frame, parts, hist, dataset)
            if 'x' not in arrays[split]:
                arrays[split]['x'] = np.lib.format.open_memmap(out / f'x_{split}.npy', mode='w+', dtype='float64', shape=(n, raw.shape[1] + 1))
            end = offset + len(frame); sl = slice(offset, end)
            arrays[split]['x'][sl, 0] = 1.; arrays[split]['x'][sl, 1:] = raw
            arrays[split]['parts'][sl] = parts; arrays[split]['y'][sl] = y
            if split == 'dev':
                ss = raw[:, continuous].sum(0); sq = np.square(raw[:, continuous]).sum(0)
                scaler_sum = ss if scaler_sum is None else scaler_sum + ss
                scaler_square = sq if scaler_square is None else scaler_square + sq
            else:
                old = canonical(pd.read_parquet(source / 'fare' / f'{origin}.parquet'))
                old = old if chosen is None else old[old.series_id.isin(chosen)]
                paired = exact_merge(frame[KEYS], old).sort_values(KEYS)
                if not np.allclose(paired.backbone, parts[:, 0], rtol=1e-8, atol=1e-8):
                    raise ValueError('Existing baseline backbone differs from cached candidate B')
                arrays[split]['baseline'][sl] = paired.soft_history
            key_chunks.append(frame[KEYS]); weeks.append(dict(origin=origin, start=offset, stop=end)); offset = end
        if offset != n or not n:
            raise ValueError('Prepared coverage differs from expected coverage')
        pd.concat(key_chunks, ignore_index=True).to_parquet(out / f'keys_{split}.parquet', index=False)
        info[split] = dict(rows=n, weeks=weeks)
    mean = scaler_sum / info['dev']['rows']
    std = np.sqrt(np.maximum(scaler_square / info['dev']['rows'] - mean ** 2, 0)); std = np.maximum(std, 1e-8)
    scaler_source = 'development-only cached candidate features; no model fit'
    if (source / 'fare' / 'fit.json').exists() and not profile_series:
        saved = json.loads((source / 'fare' / 'fit.json').read_text())['design']
        if saved['names'] != names:
            raise ValueError('Existing model feature names differ')
        mean, std = np.asarray(saved['mean']), np.asarray(saved['std'])
        scaler_source = 'unchanged existing full-model scaler'
    for split in ('dev', 'test'):
        x = arrays[split]['x']
        for start in range(0, len(x), 65536):
            stop = min(start + 65536, len(x)); block = x[start:stop].copy()
            block[:, np.asarray(continuous) + 1] = (block[:, np.asarray(continuous) + 1] - mean) / std
            x[start:stop] = block
        for array in arrays[split].values():
            array.flush()
    write_json(out / 'metadata.json', dict(identity=identity, names=names, continuous=continuous,
               scaler=dict(mean=mean.tolist(), std=std.tolist(), source=scaler_source), splits=info,
               production=profile_series is None, history_frozen_after_dev=True,
               candidates='unchanged B/S/F; every original gate input retained in all new variants'))
    seal(out, identity); print('PREPARED', dataset, info['dev']['rows'], info['test']['rows'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, default=Path(__file__).resolve().parents[2] / 'data' / 'baselines',
                   help='Folder produced by baselines/prepare_inputs.py, one subfolder per dataset')
    p.add_argument('--dataset', choices=EXPECTED, required=True); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    prepare(a.inputs, a.dataset, a.out)
