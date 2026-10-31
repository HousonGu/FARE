"""Read-only frozen panels for independent conventional forecasting baselines.

The original sales matrix, schedule and eligibility metadata are copied byte for
byte.  Predictors receive 56 observations and seven known-factor rows; access to
supervised labels is explicitly limited to the original fit split.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from io_utils import verify

KEYS = ['origin', 'series_id', 'target_date']
EXPECTED = {'favorita': (25, 311850), 'm5': (52, 11053322),
            'rossmann': (26, 196630), 'rohlik': (26, 283185)}


class Panel:
    def __init__(self, folder, check=True):
        self.folder = Path(folder)
        if check and not verify(self.folder):
            raise ValueError('Prepared conventional panel is incomplete')
        self.meta = json.loads((self.folder / 'panel.json').read_text(encoding='utf-8'))
        self.dataset = self.meta['dataset']
        self.schedule = self.meta['schedule']
        self.ids = np.asarray(self.meta['ids'], dtype=str)
        self.days = self.meta['days']
        self.id_rows = {v: i for i, v in enumerate(self.ids)}
        self.columns = {v: i for i, v in enumerate(self.days)}
        if len(self.id_rows) != len(self.ids) or len(self.columns) != len(self.days):
            raise ValueError('Duplicate series or day keys')
        self.first = np.asarray(self.meta['first'], dtype='datetime64[D]')
        self.values = np.load(self.folder / 'values.npy', mmap_mode='r')
        if self.values.shape != (len(self.ids), len(self.days)):
            raise ValueError('Frozen sales panel shape differs')
        self.cov_meta = json.loads((self.folder / 'covariates.json').read_text(encoding='utf-8'))
        self.covariate_names = list(self.cov_meta['names'])
        self.covariates = np.load(self.folder / 'covariates.npy', mmap_mode='r')
        if self.covariates.shape != (*self.values.shape, len(self.covariate_names)):
            raise ValueError('Known-factor panel shape differs')
        self._indices = {}
        self._origins = set(self.schedule['fit'] + self.schedule['dev'] + self.schedule['test'])
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
        if origin not in self._origins:
            raise ValueError('Origin outside the frozen schedule: ' + str(origin))
        if 'eligible_rows' in self.meta:
            result = np.asarray(self.meta['eligible_rows'][origin], dtype=np.int64)
        else:
            result = np.flatnonzero(self.first <= np.datetime64(origin) - np.timedelta64(56, 'D'))
        if len(np.unique(result)) != len(result) or (result < 0).any() or (result >= len(self.ids)).any():
            raise ValueError('Invalid frozen eligibility')
        return result

    def _rows(self, origin, rows):
        eligible = self.rows(origin)
        if rows is None:
            return eligible
        result = np.asarray(rows)
        if result.ndim != 1 or result.dtype.kind not in 'iu':
            raise ValueError('Rows must be one-dimensional integer indices')
        if len(np.unique(result)) != len(result) or not np.isin(result, eligible).all():
            raise ValueError('Duplicate or ineligible inference series')
        return result.astype(np.int64, copy=False)

    def indices(self, origin):
        if origin not in self._origins:
            raise ValueError('Origin outside the frozen schedule: ' + str(origin))
        if origin not in self._indices:
            days = pd.date_range(pd.Timestamp(origin) - pd.Timedelta(days=56), periods=63).strftime('%Y-%m-%d')
            self._indices[origin] = np.asarray([self.columns[d] for d in days], dtype=np.int64)
        return self._indices[origin]

    def keys(self, origin, rows=None):
        rows = self._rows(origin, rows)
        return pd.DataFrame(dict(origin=origin, series_id=np.repeat(self.ids[rows], 7),
            target_date=np.tile(pd.date_range(origin, periods=7).strftime('%Y-%m-%d'), len(rows))))

    def context(self, origin, rows=None):
        rows = self._rows(origin, rows)
        result = np.asarray(self.values[np.ix_(rows, self.indices(origin)[:56])], dtype=np.float64)
        if not np.isfinite(result).all() or (result < 0).any():
            raise ValueError('Invalid observed history; no zero imputation allowed')
        return result

    def future_covariates(self, origin, rows=None):
        rows = self._rows(origin, rows)
        result = np.asarray(self.covariates[np.ix_(rows, self.indices(origin)[56:],
                                                np.arange(len(self.covariate_names)))], dtype=np.float32)
        if not np.isfinite(result).all():
            raise ValueError('Missing known future factors; evaluation rows cannot be dropped')
        return result

    def _labels(self, origin, rows):
        result = np.asarray(self.values[np.ix_(rows, self.indices(origin)[56:])], dtype=np.float64)
        if not np.isfinite(result).all() or (result < 0).any():
            raise ValueError('Invalid labels; evaluation rows cannot change')
        return result

    def training_labels(self, origin, rows=None):
        if origin not in self.schedule['fit']:
            raise ValueError('Model training/selection may read only fit labels')
        return self._labels(origin, self._rows(origin, rows))

    def truth(self, origin):
        rows = self.rows(origin)
        result = self.keys(origin, rows)
        result['actual'] = self._labels(origin, rows).ravel()
        return result

    def audit(self):
        coverage, actual_sums = {}, {}
        for split in ['fit', 'dev', 'test']:
            coverage[split], actual_sums[split] = {}, {}
            for origin in self.schedule[split]:
                rows = self.rows(origin)
                self.context(origin, rows)
                labels = self._labels(origin, rows)
                factors = self.covariates[np.ix_(rows, self.indices(origin),
                                               np.arange(len(self.covariate_names)))]
                if not np.isfinite(factors).all():
                    raise ValueError('Missing factors in an eligible 63-day window')
                for i, name in enumerate(self.covariate_names):
                    allowed = self.cov_meta['encodings'][name]['values']
                    if not np.isin(factors[..., i], allowed).all():
                        raise ValueError('Invalid factor encoding: ' + name)
                coverage[split][origin] = int(len(rows) * 7)
                actual_sums[split][origin] = float(labels.sum(dtype=np.float64))
        expected = EXPECTED[self.dataset]
        if (len(coverage['test']), sum(coverage['test'].values())) != expected:
            raise ValueError('Formal test origin/row range changed')
        return dict(dataset=self.dataset, series=len(self.ids), shape=list(self.values.shape),
                    covariate_names=self.covariate_names, coverage=coverage,
                    actual_sums=actual_sums,
                    test_rows=expected[1], test_origins=expected[0],
                    missing_eligible_factors=0, target_imputation='none',
                    old_predictions_included=False)
