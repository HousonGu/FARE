"""Exact same-key merge for the pinned Chronos prediction adapters."""
KEYS = ['origin', 'series_id', 'target_date']


def merge(left, right):
    result = left.merge(right, on=KEYS, how='outer', validate='one_to_one', indicator=True)
    if not result._merge.eq('both').all():
        raise ValueError('Prediction keys differ')
    return result.drop(columns='_merge')
