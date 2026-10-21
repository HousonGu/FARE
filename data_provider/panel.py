"""Frozen full-panel access; supervised labels are restricted to the fit split."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from exp.io import KEYS, canonical, merge, verify

EXPECTED = {'favorita': (25, 311850), 'm5': (52, 11053322),
            'rossmann': (26, 196630), 'rohlik': (26, 283185)}


class Panel:
    def __init__(self, folder, check=True):
        self.folder = Path(folder)
        if check and not verify(self.folder):
            raise ValueError('Prepared panel is incomplete')
        self.meta = json.loads((self.folder / 'panel.json').read_text(encoding='utf-8'))
        self.dataset = self.meta['dataset']
        self.schedule = self.meta['schedule']
        self.ids = np.asarray(self.meta['ids'], dtype=str)
        if len(set(self.ids)) != len(self.ids):
            raise ValueError('Duplicate series ids')
        self.id_rows = {v: i for i, v in enumerate(self.ids)}
        self.days = self.meta['days']
        self.columns = {v: i for i, v in enumerate(self.days)}
        self.first = np.asarray(self.meta['first'], dtype='datetime64[D]')
        self.values = np.load(self.folder / 'values.npy', mmap_mode='r')
        if self.values.shape != (len(self.ids), len(self.days)):
            raise ValueError('Panel shape differs')
        self._indices = {}
        self.validate_schedule()

    def validate_schedule(self):
        s = self.schedule
        for key in ['fit', 'train', 'tune', 'dev', 'test']:
            if not s[key] or s[key] != sorted(set(s[key])):
                raise ValueError('Unordered or empty split ' + key)
            dates = pd.to_datetime(s[key])
            if any(d.weekday() != 0 for d in dates) or any((b-a).days < 7 for a, b in zip(dates, dates[1:])):
                raise ValueError('Non-Monday or overlapping origins')
        if s['train'] + s['tune'] != s['fit']:
            raise ValueError('Train/tune must partition fit')
        for a, b in [('train', 'tune'), ('fit', 'dev'), ('dev', 'test')]:
            if pd.Timestamp(s[a][-1]) + pd.Timedelta(days=6) >= pd.Timestamp(s[b][0]):
                raise ValueError('Overlapping split labels')

    def rows(self, origin):
        if 'eligible_rows' in self.meta:
            result = np.asarray(self.meta['eligible_rows'][origin], dtype=np.int64)
        else:
            result = np.flatnonzero(self.first <= np.datetime64(origin) - np.timedelta64(56, 'D'))
        if len(np.unique(result)) != len(result) or (result < 0).any() or (result >= len(self.ids)).any():
            raise ValueError('Invalid frozen eligibility')
        return result

    def indices(self, origin):
        if origin not in self._indices:
            days = pd.date_range(pd.Timestamp(origin) - pd.Timedelta(days=56), periods=63).strftime('%Y-%m-%d')
            self._indices[origin] = np.asarray([self.columns[d] for d in days])
        return self._indices[origin]

    def keys(self, origin, rows=None):
        rows = self.rows(origin) if rows is None else np.asarray(rows)
        return pd.DataFrame(dict(origin=origin, series_id=np.repeat(self.ids[rows], 7),
            target_date=np.tile(pd.date_range(origin, periods=7).strftime('%Y-%m-%d'), len(rows))))

    def context(self, origin, rows=None):
        rows = self.rows(origin) if rows is None else np.asarray(rows)
        if not np.isin(rows, self.rows(origin)).all():
            raise ValueError('Ineligible inference series')
        x = np.asarray(self.values[np.ix_(rows, self.indices(origin)[:56])], np.float32)
        if not np.isfinite(x).all() or (x < 0).any():
            raise ValueError('Invalid history; no zero imputation')
        return x

    def truth(self, origin):
        result = self.keys(origin)
        result['actual'] = np.asarray(self.values[np.ix_(self.rows(origin), self.indices(origin)[56:])], np.float64).ravel()
        if not np.isfinite(result.actual).all() or (result.actual < 0).any():
            raise ValueError('Invalid labels; cannot change evaluation rows')
        return result

    def reference(self, origin):
        result = pd.read_parquet(self.folder / 'references' / f'{origin}.parquet')
        return merge(self.keys(origin), result)

    def audit(self):
        coverage = {}
        for split in ['fit', 'dev', 'test']:
            coverage[split] = {}
            for origin in self.schedule[split]:
                self.context(origin)
                self.truth(origin)
                coverage[split][origin] = len(self.rows(origin)) * 7
                if split != 'fit':
                    self.reference(origin)
        expected = EXPECTED[self.dataset]
        if (len(coverage['test']), sum(coverage['test'].values())) != expected:
            raise ValueError('Formal test range changed')
        return dict(dataset=self.dataset, series=len(self.ids), shape=list(self.values.shape),
                    coverage=coverage, test_rows=expected[1], test_origins=expected[0])


class Windows:
    def __init__(self, panel, origins):
        if not set(origins) <= set(panel.schedule['fit']):
            raise ValueError('Backbone training/selection may read only fit labels')
        self.panel = panel
        self.origins = list(origins)
        self.rows = [panel.rows(o) for o in origins]
        self.ends = np.cumsum([len(r) for r in self.rows])
        self.flat_rows = np.concatenate(self.rows)
        self.columns = np.asarray([panel.indices(o) for o in origins])

    def __len__(self):
        return int(self.ends[-1])

    def batch(self, indices):
        indices = np.asarray(indices)
        which = np.searchsorted(self.ends, indices, side='right')
        values = np.asarray(self.panel.values[self.flat_rows[indices, None], self.columns[which]], np.float32)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError('Invalid training windows')
        x, y = values[:, :56], values[:, 56:]
        scale = np.maximum(x.mean(axis=1, dtype=np.float64), 1).astype(np.float32)
        return x / scale[:, None], y / scale[:, None], scale

    def targets(self, indices):
        indices = np.asarray(indices)
        which = np.searchsorted(self.ends, indices, side='right')
        return np.asarray(self.panel.values[self.flat_rows[indices, None], self.columns[which, 56:]], np.float64)
