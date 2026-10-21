"""Development-only exact whole-history retrieval and fixed 50:50 fusion.

Same family/item across stores, normalized 56-day L2, equal-weight Top-8.
Candidate targets finish strictly before the query origin. No future covariates.
"""

import pandas as pd

from data_provider.paths import ROOT, RAW_ROOT

def load_matrix(dataset):
    base = RAW_ROOT / dataset
    if dataset == 'favorita':
        raw = pd.read_csv(base / 'train.csv', usecols=['date', 'store_nbr', 'family', 'sales'])
        keys = raw[['store_nbr', 'family']].drop_duplicates().sort_values(['store_nbr', 'family'])
        keys['item_id'] = [f'f{i:04d}' for i in range(len(keys))]
        raw = raw.merge(keys, validate='many_to_one')
        matrix = raw.pivot(index='item_id', columns='date', values='sales')
        groups = keys.set_index('item_id')['family']
        first = pd.Series(pd.Timestamp('1900-01-01'), index=matrix.index)
    else:
        cal = pd.read_csv(base / 'calendar.csv', parse_dates=['date'])
        lookup = dict(zip(cal.d, cal.date.dt.strftime('%Y-%m-%d')))
        raw = pd.read_csv(base / 'sales_train_evaluation.csv')
        matrix = raw.set_index('id')[[c for c in raw if c.startswith('d_')]].rename(columns=lookup)
        groups = raw.set_index('id').item_id
        prices = pd.read_csv(base / 'sell_prices.csv', usecols=['store_id', 'item_id', 'wm_yr_wk'])
        weeks = prices.groupby(['store_id', 'item_id']).wm_yr_wk.min().reset_index()
        weeks['first'] = weeks.wm_yr_wk.map(cal.groupby('wm_yr_wk').date.min())
        first = raw[['id', 'store_id', 'item_id']].merge(weeks, validate='one_to_one').set_index('id')['first']
    return matrix, groups, first
