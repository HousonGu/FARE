"""Score complete same-key predictions on the test origins."""
import pandas as pd
from data_provider.retail import CONFIG, KEYS, checked_merge, check_prediction, metric, json_write

MAIN = {'backbone','soft_history','soft_no_history','soft_history_no_factor',
        'soft_history_no_disagreement','shape_half','factor_half','global_static','history_hard','flat_softmax'}


def families(backbone='all'):
    all_families = ['chronos','lightgbm']+[f'tft_seed_{s}' for s in CONFIG['tft_seeds']]
    return all_families if backbone == 'all' else [f for f in all_families if f == backbone or (backbone == 'tft' and f.startswith('tft_seed_'))]


def score(data,schedule,folder,backbone='all'):
    predictions = {}
    for family in families(backbone):
        frame = pd.read_parquet(folder/f'gates_{family}'/'predictions.parquet')
        expected = MAIN if family=='chronos' else {'backbone','soft_history'}
        if set(frame)-set(KEYS) != expected: raise ValueError(f'Missing/extra methods: {family}')
        predictions[family] = frame
    for family,frame in predictions.items():
        check_prediction(frame)
        if sorted(frame.origin.unique()) != schedule['test']: raise ValueError(f'Wrong schedule: {family}')
    truth = data.truth(schedule['test'])
    # Favorita promotion-holiday subgroup, defined from known factors only.
    subgroup = None
    if data.dataset=='favorita':
        refs = pd.concat([pd.read_parquet(folder/'references'/f'{o}.parquet')[KEYS+['onpromotion','holiday_day']]
                          for o in schedule['test']],ignore_index=True)
        subgroup = checked_merge(truth[KEYS],refs)
        subgroup['selected'] = (subgroup.onpromotion>=6)&(subgroup.holiday_day==1)
        truth = checked_merge(truth,subgroup[KEYS+['selected']])
    overall,weekly,groups = [],[],[]
    for family,frame in predictions.items():
        merged = checked_merge(frame,truth)
        for method in [c for c in frame if c not in KEYS]:
            name = family+'/'+method
            overall.append(dict(method=name,**metric(merged.actual,merged[method])))
            for origin,part in merged.groupby('origin',sort=True):
                weekly.append(dict(origin=origin,method=name,**metric(part.actual,part[method])))
            if subgroup is not None:
                part = merged[merged.selected]
                groups.append(dict(method=name,group='promotion>=6 and planned holiday',**metric(part.actual,part[method])))
    json_write(folder/'metrics.json',dict(overall=overall,weekly=weekly,subgroup=groups))
