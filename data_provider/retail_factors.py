"""Matched-information Favorita retrieval diagnostic, development only.

Future promotions are benchmark-provided/ex-post; holidays are planned calendar.
Natural and excluded banks use identical candidate sets across all methods.
"""

from data_provider.paths import RAW_ROOT

import numpy as np

import pandas as pd

from data_provider.holidays import favorita_holiday_store_dates


def factors(index, columns):
    base = RAW_ROOT / 'favorita'
    raw = pd.read_csv(base / 'train.csv', usecols=['date', 'store_nbr', 'family', 'onpromotion'], parse_dates=['date'])
    keys = raw[['store_nbr', 'family']].drop_duplicates().sort_values(['store_nbr', 'family'])
    keys['item_id'] = [f'f{i:04d}' for i in range(len(keys))]
    holiday = favorita_holiday_store_dates(pd.read_csv(base / 'stores.csv'),
        pd.read_csv(base / 'holidays_events.csv', parse_dates=['date'])).assign(holiday=True)
    raw = raw.merge(keys, validate='many_to_one').merge(holiday, on=['store_nbr', 'date'], how='left', validate='many_to_one')
    raw['holiday'] = raw.holiday.eq(True).astype(float)
    raw['promo_bin'] = np.select([raw.onpromotion >= 6, raw.onpromotion > 0], [2, 1], default=0)
    raw['date'] = raw.date.dt.strftime('%Y-%m-%d')
    matrices = [raw.pivot(index='item_id', columns='date', values=c).loc[index, columns].to_numpy()
                for c in ['promo_bin', 'holiday']]
    return matrices


from fare.retrieval import select
