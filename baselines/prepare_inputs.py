"""Assemble the shared input folder used by the comparison methods and controls.

For one dataset this script combines
  * the prepared sales panel and references from ``python -m exp.prepare_panel``,
  * the two known future factors read from the raw competition CSV files,
  * the frozen Chronos-2 forecasts written by the main FARE runner, and
  * optionally, the main FARE test predictions and gate design,
into ``OUT``: values.npy, panel.json, covariates.npy, covariates.json, references/,
backbone/, fare/ and a complete.json listing every file.
Sales values are copied unchanged; no prediction or label is imputed.
"""
import argparse
import contextlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'conventional'))
from data import Panel  # noqa: E402
from io_utils import KEYS, merge, save_parquet, seal, verify, write_json  # noqa: E402

DATASETS = ('favorita', 'm5', 'rossmann', 'rohlik')
DISCOUNTS = [f'type_{i}_discount' for i in range(7)]


class Sources:
    """Read the raw CSV files of one dataset and record their sizes."""
    def __init__(self, raw_root):
        self.raw_root = Path(raw_root).resolve()
        self.records = {}

    @contextlib.contextmanager
    def open(self, dataset, name):
        path = self.raw_root / dataset / name
        if not path.is_file():
            raise FileNotFoundError('Raw input unavailable: ' + str(path))
        with path.open('rb') as stream:
            yield stream

    def csv(self, dataset, name, **kwargs):
        with self.open(dataset, name) as stream:
            result = pd.read_csv(stream, **kwargs)
        self.records[f'{dataset}/{name}'] = dict(bytes=(self.raw_root / dataset / name).stat().st_size)
        return result


def _pivot(frame, id_column, date_column, field, meta):
    if frame.duplicated([id_column, date_column]).any():
        raise ValueError('Duplicate known-factor keys')
    frame = frame.copy()
    frame[id_column] = frame[id_column].astype(str)
    frame[date_column] = pd.to_datetime(frame[date_column]).dt.strftime('%Y-%m-%d')
    return frame.pivot(index=id_column, columns=date_column, values=field).reindex(
        index=meta['ids'], columns=meta['days']).to_numpy(dtype=np.float32)


def favorita(source, meta):
    frame = source.csv('favorita', 'train.csv',
                       usecols=['date', 'store_nbr', 'family', 'onpromotion'], parse_dates=['date'])
    if frame.onpromotion.isna().any() or (frame.onpromotion < 0).any():
        raise ValueError('Unknown/negative Favorita promotion')
    keys = frame[['store_nbr', 'family']].drop_duplicates().sort_values(['store_nbr', 'family'])
    keys['series_id'] = [f'f{i:04d}' for i in range(len(keys))]
    frame = frame.merge(keys, on=['store_nbr', 'family'], validate='many_to_one')
    frame['promo_bin'] = np.select([frame.onpromotion >= 6, frame.onpromotion > 0], [2, 1], default=0)
    stores = source.csv('favorita', 'stores.csv')
    holidays = source.csv('favorita', 'holidays_events.csv', parse_dates=['date'])
    # Planned holidays only: no unplanned Event, Work Day, or transferred original dates.
    planned = holidays.loc[(~holidays.transferred) & holidays.type.isin(
        ['Holiday', 'Transfer', 'Additional', 'Bridge'])]
    national = stores[['store_nbr']].merge(planned.loc[planned.locale.eq('National'), ['date']], how='cross')
    regional = stores[['store_nbr', 'state']].merge(
        planned.loc[planned.locale.eq('Regional'), ['date', 'locale_name']],
        left_on='state', right_on='locale_name')[['store_nbr', 'date']]
    local = stores[['store_nbr', 'city']].merge(
        planned.loc[planned.locale.eq('Local'), ['date', 'locale_name']],
        left_on='city', right_on='locale_name')[['store_nbr', 'date']]
    observed = pd.concat([national, regional, local], ignore_index=True).drop_duplicates()
    observed['holiday'] = np.int8(1)
    frame = frame.merge(observed, on=['store_nbr', 'date'], how='left', validate='many_to_one')
    # Absence from the scheduled holiday calendar means no holiday; no sales value is filled here.
    frame['holiday'] = frame.holiday.fillna(0)
    arrays = [_pivot(frame, 'series_id', 'date', name, meta) for name in ['promo_bin', 'holiday']]
    return arrays, dict(names=['promo_bin', 'holiday'], encodings={
        'promo_bin': dict(values=[0, 1, 2], definition='0: onpromotion=0; 1: 1..5; 2: >=6'),
        'holiday': dict(values=[0, 1], definition='Scheduled observed National/Regional/Local holiday for store; types Holiday/Transfer/Additional/Bridge; transferred=False')})


def m5(source, meta):
    keys = source.csv('m5', 'sales_train_evaluation.csv', usecols=['id', 'state_id']).set_index('id')
    if keys.index.duplicated().any() or set(keys.index) != set(meta['ids']):
        raise ValueError('M5 series identity differs from the prepared panel')
    calendar = source.csv('m5', 'calendar.csv', parse_dates=['date'])
    if calendar.date.duplicated().any():
        raise ValueError('Duplicate M5 calendar dates')
    calendar['day'] = calendar.date.dt.strftime('%Y-%m-%d')
    calendar = calendar.set_index('day').reindex(meta['days'])
    event = (calendar.event_name_1.notna() | calendar.event_name_2.notna()).to_numpy(dtype=np.float32)
    states = keys.loc[meta['ids'], 'state_id'].to_numpy()
    snap = np.empty((len(states), len(meta['days'])), dtype=np.float32)
    for state in sorted(set(states)):
        values = calendar[f'snap_{state}'].to_numpy(dtype=np.float32)
        if not np.isin(values, [0, 1]).all():
            raise ValueError('Missing/nonbinary SNAP schedule')
        snap[states == state] = values
    # All panel dates must exist; missing calendar rows must not become event=0.
    if calendar.date.isna().any():
        raise ValueError('Missing M5 calendar dates')
    arrays = [snap, np.broadcast_to(event, snap.shape)]
    return arrays, dict(names=['snap', 'event'], encodings={
        'snap': dict(values=[0, 1], definition='Calendar snap_CA/snap_TX/snap_WI selected by original state_id'),
        'event': dict(values=[0, 1], definition='event_name_1.notna() OR event_name_2.notna()')})


def rossmann(source, meta):
    frame = source.csv('rossmann', 'train.csv', usecols=['Store', 'Date', 'Promo', 'SchoolHoliday'])
    for name in ['Promo', 'SchoolHoliday']:
        if not frame[name].isin([0, 1]).all():
            raise ValueError('Unknown/nonbinary Rossmann factor ' + name)
    arrays = [_pivot(frame, 'Store', 'Date', name, meta) for name in ['Promo', 'SchoolHoliday']]
    return arrays, dict(names=['Promo', 'SchoolHoliday'], encodings={
        'Promo': dict(values=[0, 1], definition='Original Promo binary flag'),
        'SchoolHoliday': dict(values=[0, 1], definition='Original SchoolHoliday binary flag')},
        excluded_inputs=['Open', 'Customers', 'StateHoliday'])


def rohlik(source, meta):
    schema = source.csv('rohlik', 'sales_test.csv', nrows=0)
    if not set(DISCOUNTS).issubset(schema):
        raise ValueError('Planned discount fields absent from public test schema')
    frame = source.csv('rohlik', 'sales_train.csv', usecols=['unique_id', 'date', 'warehouse'] + DISCOUNTS)
    discounts = frame[DISCOUNTS].to_numpy(dtype=np.float64)
    if np.isinf(discounts).any():
        raise ValueError('Infinite Rohlik discount')
    frame['Discount'] = np.where(np.isfinite(discounts).all(axis=1),
        (np.maximum(discounts, 0).max(axis=1) > 0).astype(float), np.nan)
    inventory = source.csv('rohlik', 'inventory.csv', usecols=['unique_id', 'warehouse'])
    if inventory.unique_id.duplicated().any():
        raise ValueError('Duplicate Rohlik inventory IDs')
    check = frame[['unique_id', 'warehouse']].drop_duplicates().merge(
        inventory, on='unique_id', suffixes=('', '_inventory'), validate='one_to_one', how='left')
    if check.warehouse_inventory.isna().any() or not check.warehouse.eq(check.warehouse_inventory).all():
        raise ValueError('Rohlik warehouse metadata mismatch')
    calendar = source.csv('rohlik', 'calendar.csv', usecols=['warehouse', 'date', 'holiday'])
    if calendar.duplicated(['warehouse', 'date']).any() or not calendar.holiday.isin([0, 1]).all():
        raise ValueError('Duplicate/unknown Rohlik holiday calendar')
    frame = frame.merge(calendar, on=['warehouse', 'date'], how='left', validate='many_to_one')
    if frame.holiday.isna().any():
        raise ValueError('Missing Rohlik holiday for observed dates')
    discount = _pivot(frame, 'unique_id', 'date', 'Discount', meta)
    # Warehouse holidays apply to every series and day, including unobserved sales dates.
    calendar['date'] = pd.to_datetime(calendar.date).dt.strftime('%Y-%m-%d')
    cal = calendar.pivot(index='warehouse', columns='date', values='holiday').reindex(columns=meta['days'])
    inventory['unique_id'] = inventory.unique_id.astype(str)
    warehouses = inventory.set_index('unique_id').loc[meta['ids'], 'warehouse']
    holiday = cal.reindex(index=warehouses).to_numpy(dtype=np.float32)
    return [discount, holiday], dict(names=['Discount', 'Holiday'], encodings={
        'Discount': dict(values=[0, 1], definition='1 iff max(max(type_0_discount..type_6_discount,0))>0; any missing field remains NaN'),
        'Holiday': dict(values=[0, 1], definition='Public calendar.holiday for the series warehouse/date')},
        excluded_inputs=['availability', 'total_orders', 'sell_price_main'],
        missing_discount_rows=int(np.isnan(discounts).any(axis=1).sum()),
        negative_discount_values=int((discounts < 0).sum()))


def _forecast_week(path, keys):
    frame = pd.read_parquet(path).rename(columns={'backbone_forecast': 'forecast'})
    if not set(KEYS + ['forecast']) <= set(frame):
        raise ValueError('Backbone file needs origin, series_id, target_date and forecast: ' + str(path))
    frame = merge(keys, frame[KEYS + ['forecast']])
    values = frame.forecast.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite backbone forecast: ' + str(path))
    frame['forecast'] = np.maximum(values, 0)
    return frame[KEYS + ['forecast']]


def prepare(args):
    dataset, panel_dir, out = args.dataset, args.panel.resolve(), args.out.resolve()
    if out.exists():
        raise ValueError('Use a new output directory: ' + str(out))
    if not verify(panel_dir):
        raise ValueError('Prepared panel is missing or incomplete; run python -m exp.prepare_panel first')
    meta = json.loads((panel_dir / 'panel.json').read_text(encoding='utf-8'))
    if meta['dataset'] != dataset:
        raise ValueError('Prepared panel belongs to ' + meta['dataset'])
    meta['schedule'] = dict(meta['schedule'], dataset=dataset)
    out.mkdir(parents=True)
    shutil.copy2(panel_dir / 'values.npy', out / 'values.npy')
    source = Sources(args.raw_root)
    arrays, cov_meta = globals()[dataset](source, meta)
    covariates = np.stack(arrays, axis=-1).astype(np.float32, copy=False)
    np.save(out / 'covariates.npy', covariates, allow_pickle=False)
    cov_meta.update(dataset=dataset, shape=list(covariates.shape), dtype=str(covariates.dtype),
                    raw_inputs=source.records)
    write_json(out / 'covariates.json', cov_meta)
    write_json(out / 'panel.json', meta)
    panel = Panel(out, check=False)
    origins = meta['schedule']['dev'] + meta['schedule']['test']
    for origin in origins:
        keys = panel.keys(origin)
        reference = panel_dir / 'references' / f'{origin}.parquet'
        (out / 'references').mkdir(exist_ok=True)
        shutil.copy2(reference, out / 'references' / reference.name)
        save_parquet(out / 'backbone' / f'{origin}.parquet',
                     _forecast_week(args.chronos / f'{origin}.parquet', keys))
    if args.fare_predictions is not None:
        fare = pd.read_parquet(args.fare_predictions)
        for key in KEYS:
            fare[key] = fare[key].astype(str)
        for origin in meta['schedule']['test']:
            week = merge(panel.keys(origin), fare.loc[fare.origin.eq(origin), KEYS + ['backbone', 'soft_history']])
            save_parquet(out / 'fare' / f'{origin}.parquet', week)
    if args.fare_fit is not None:
        design = json.loads(args.fare_fit.read_text(encoding='utf-8'))['design']
        write_json(out / 'fare' / 'fit.json', dict(design=dict(names=design['names'], mean=design['mean'],
                                                              std=design['std'])))
    write_json(out / 'audit.json', panel.audit())
    files = [p for p in out.rglob('*') if p.is_file()]
    seal(out, files, dict(dataset=dataset, panel_source=str(panel_dir)))
    print(dataset, 'inputs prepared in', out, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', choices=DATASETS, required=True)
    parser.add_argument('--panel', type=Path, required=True, help='Output of python -m exp.prepare_panel')
    parser.add_argument('--raw-root', type=Path, required=True, help='Parent of the raw CSV folders')
    parser.add_argument('--chronos', type=Path, required=True,
                        help='Folder of weekly Chronos-2 forecasts {origin}.parquet from the main runner')
    parser.add_argument('--fare-predictions', type=Path,
                        help='Main-runner test predictions with backbone and soft_history columns')
    parser.add_argument('--fare-fit', type=Path, help='Main-runner gate fit.json holding the design scaler')
    parser.add_argument('--out', type=Path, required=True)
    prepare(parser.parse_args())


if __name__ == '__main__':
    main()
