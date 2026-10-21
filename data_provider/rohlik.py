"""Official Rohlik Sales V2 panel, with missing observations retained as NaN."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

KEYS = ['origin', 'series_id', 'target_date']
DISCOUNTS = [f'type_{i}_discount' for i in range(7)]
FILES = ['sales_train.csv', 'sales_test.csv', 'inventory.csv', 'calendar.csv']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


class RohlikData:
    def __init__(self, raw):
        self.raw = Path(raw)
        # Check public forecast schema, but never use competition test values or solution.
        test_columns = pd.read_csv(self.raw / 'sales_test.csv', nrows=0).columns
        if not set(DISCOUNTS).issubset(test_columns):
            raise ValueError('Required planned discount fields absent from official test schema')
        frame = pd.read_csv(self.raw / 'sales_train.csv',
                            usecols=['unique_id', 'date', 'warehouse', 'sales'] + DISCOUNTS)
        meta = pd.read_csv(self.raw / 'inventory.csv',
                           usecols=['unique_id', 'warehouse', 'L2_category_name_en'])
        calendar = pd.read_csv(self.raw / 'calendar.csv', usecols=['warehouse', 'date', 'holiday'])
        for f in [frame, calendar]:
            f['date'] = pd.to_datetime(f.date, errors='raise')
            if f.date.isna().any() or not f.date.eq(f.date.dt.normalize()).all():
                raise ValueError('Missing/non-daily date')
        if frame.duplicated(['unique_id', 'date']).any() or meta.unique_id.duplicated().any():
            raise ValueError('Duplicate inventory/date or metadata keys')
        if calendar.duplicated(['warehouse', 'date']).any():
            raise ValueError('Duplicate calendar keys')
        if frame[['unique_id', 'warehouse']].isna().any().any():
            raise ValueError('Missing series/warehouse')
        if not np.isfinite(frame.sales.dropna()).all() or (frame.sales.dropna() < 0).any():
            raise ValueError('Invalid sales; do not silently clip targets')
        if not calendar.holiday.isin([0, 1]).all():
            raise ValueError('Missing/nonbinary holiday')
        values = frame[DISCOUNTS].to_numpy(float)
        if np.isinf(values).any():
            raise ValueError('Infinite discount')
        # Missing any discount field remains unknown, never silently treated as no promo.
        frame['Discount'] = np.where(np.isfinite(values).all(1), (np.maximum(values, 0).max(1) > 0).astype(float), np.nan)
        negative_discounts = int((values < 0).sum())
        missing_discounts = int(np.isnan(values).any(1).sum())
        frame = frame.drop(columns=DISCOUNTS)
        frame = frame.merge(meta.rename(columns={'warehouse': 'meta_warehouse'}), on='unique_id', how='left', validate='many_to_one')
        if frame[['meta_warehouse', 'L2_category_name_en']].isna().any().any():
            raise ValueError('Missing warehouse/category metadata')
        if not frame.warehouse.eq(frame.meta_warehouse).all():
            raise ValueError('Inventory/observation warehouse mismatch')
        if frame.groupby('unique_id').warehouse.nunique().max() != 1:
            raise ValueError('Series changes warehouse')
        frame = frame.merge(calendar, on=['warehouse', 'date'], how='left', validate='many_to_one')
        if frame.holiday.isna().any():
            raise ValueError('Missing calendar for observed dates')
        self.ids = np.array(sorted(frame.unique_id.unique()), dtype=np.int64)
        self.dates = pd.date_range(frame.date.min(), frame.date.max())
        self.arr = {}
        for column, name in [('sales', 'Sales'), ('Discount', 'Discount')]:
            self.arr[name] = frame.pivot(index='unique_id', columns='date', values=column).reindex(index=self.ids, columns=self.dates).to_numpy(float)
        metadata = meta.set_index('unique_id').loc[self.ids]
        cal = calendar.pivot(index='warehouse', columns='date', values='holiday').reindex(columns=self.dates)
        self.arr['Holiday'] = cal.reindex(index=metadata.warehouse).to_numpy(float)
        self.groups = [np.flatnonzero((metadata.warehouse.to_numpy() == w) & (metadata.L2_category_name_en.to_numpy() == c))
                       for w, c in sorted(set(zip(metadata.warehouse, metadata.L2_category_name_en)))]
        origins = pd.date_range(self.dates[0] + pd.Timedelta(days=56), self.dates[-1] - pd.Timedelta(days=6), freq='W-MON').strftime('%Y-%m-%d').tolist()
        if len(origins) < 61:
            raise ValueError('Need at least 9 fit + 26 dev + 26 test origins')
        self.schedule = dict(fit=origins[:-52], dev=origins[-52:-26], test=origins[-26:])
        before = frame[frame.date < pd.Timestamp(self.schedule['test'][0])]
        self.audit = dict(dataset='rohlik_sales_v2_official', series=len(self.ids), rows=len(frame),
            start=str(self.dates[0].date()), end=str(self.dates[-1].date()),
            missing_sales=int(frame.sales.isna().sum()), zero_sales=int(frame.sales.eq(0).sum()),
            missing_panel_days=int(len(self.ids) * len(self.dates) - len(frame)),
            negative_discount_values=negative_discounts, missing_discount_rows=missing_discounts,
            target='anonymized availability-adjusted volume, pcs or kg; no second availability correction',
            future_assumption='discount plan and public holiday as benchmark-provided known covariates; no historical plan snapshots',
            excluded_inputs=['availability', 'total_orders', 'sell_price_main', 'solution', 'test_weights'],
            grouping=['warehouse', 'L2_category_name_en'], groups=len(self.groups), max_group_size=max(map(len, self.groups)),
            pretest_factor_support=before.groupby(['Discount', 'holiday']).size().rename('rows').reset_index().to_dict('records'),
            raw_sha256={name: digest(self.raw / name) for name in FILES}, schedule=self.schedule,
            fit_tune=self.schedule['fit'][-8:])
        self.audit['coverage'] = [dict(origin=o, split=split, eligible_series=len(self.eligible(o)),
                                      planned_rows=len(self.eligible(o)) * 7)
                                  for split in ['dev', 'test'] for o in self.schedule[split]]
        if any(r['eligible_series'] == 0 for r in self.audit['coverage']):
            raise ValueError('A required dev/test origin has no eligible series')

    def columns(self, origin):
        pos = int((pd.Timestamp(origin) - self.dates[0]).days)
        if pos < 56 or pos + 7 > len(self.dates):
            raise ValueError('Window out of range')
        return np.arange(pos - 56, pos + 7)

    def eligible(self, origin, reference=False):
        c = self.columns(origin)
        valid = np.isfinite(self.arr['Sales'][:, c[:56]]).all(1)
        for field in ['Discount', 'Holiday']:
            valid &= np.isfinite(self.arr[field][:, c]).all(1)
        if reference:
            valid &= np.isfinite(self.arr['Sales'][:, c[56:]]).all(1)
        return np.flatnonzero(valid)

    def window(self, origin, rows=None):
        rows = self.eligible(origin) if rows is None else np.asarray(rows)
        c = self.columns(origin)
        h = self.arr['Sales'][np.ix_(rows, c[:56])]
        cov = {name: self.arr[name][np.ix_(rows, c)] for name in ['Discount', 'Holiday']}
        keys = pd.DataFrame(dict(origin=origin, series_id=np.repeat(self.ids[rows], 7),
                                 target_date=np.tile(self.dates[c[56:]].strftime('%Y-%m-%d'), len(rows))))
        return h, cov, keys

    def truth(self, origin):
        rows = self.eligible(origin)
        keys = self.window(origin, rows)[-1]
        keys['actual'] = self.arr['Sales'][np.ix_(rows, self.columns(origin)[56:])].ravel()
        return keys
