"""Seed-preserving tables; full completion requires all 24 backbone fits."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from data_provider.panel import EXPECTED
from exp.io import write_json, verify, seal, sha


def summarize(out, config, dataset='all', model='all'):
    out = Path(out)
    config = dict(config)
    if dataset != 'all': config['datasets'] = [dataset]
    if model != 'all': config['backbones'] = [model]
    records, missing, rows = [], [], []
    for dataset in config['datasets']:
        for kind in config['backbones']:
            found = []
            for seed in config['seeds']:
                directory = out/dataset/'fusion'/kind/str(seed)
                if not verify(directory):
                    missing.append(dict(dataset=dataset, backbone=kind, seed=seed))
                    continue
                record = json.loads((directory/'metrics.json').read_text(encoding='utf-8'))
                if (len(record['weekly']), record['overall']['without']['rows']) != EXPECTED[dataset]:
                    raise ValueError('Wrong formal test range')
                records.append(record)
                found.append(record)
            if len(found) != len(config['seeds']):
                continue
            for variant in config['variants']:
                wape = np.asarray([r['overall'][variant]['wape'] for r in found])
                mae = np.asarray([r['overall'][variant]['mae'] for r in found])
                rows.append(dict(dataset=dataset, backbone=kind, variant=variant, seeds=len(found),
                    test_origins=EXPECTED[dataset][0], test_rows=EXPECTED[dataset][1],
                    wape_mean=float(wape.mean()), wape_sample_std=float(wape.std(ddof=1)),
                    mae_mean=float(mae.mean()), mae_sample_std=float(mae.std(ddof=1))))
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/'summary.json', dict(status='complete' if not missing else 'incomplete',
        metrics_units=dict(wape='ratio, multiply by 100 for percent', mae='original daily sales units'),
        summary=rows, missing=missing))
    if rows:
        pd.DataFrame(rows).to_csv(out/'summary.csv', index=False)
        pd.DataFrame([r for r in rows if r['variant'] in ['without', 'with']]).to_csv(out/'paper_with_without.csv', index=False)
        single = [dict(dataset=r['dataset'], backbone=r['backbone'], seed=r['seed'], variant=v, **m)
                  for r in records for v, m in r['overall'].items()]
        pd.DataFrame(single).to_csv(out/'per_seed_metrics.csv', index=False)
        pairs = [dict(dataset=r['dataset'], backbone=r['backbone'], seed=r['seed'],
            without_wape=r['overall']['without']['wape'], with_wape=r['overall']['with']['wape'],
            **{k: v for k, v in r['paired'].items() if k != 'weekly_wape_delta'}) for r in records]
        pd.DataFrame(pairs).to_csv(out/'paired_improvements.csv', index=False)
    if not missing:
        files = [out/n for n in ['summary.json', 'summary.csv', 'paper_with_without.csv', 'per_seed_metrics.csv', 'paired_improvements.csv']]
        files += [out/d/'fusion'/k/str(s)/'complete.json' for d in config['datasets'] for k in config['backbones'] for s in config['seeds']]
        seal(out, files, dict(status='complete', datasets=len(config['datasets']),
             backbones=len(config['backbones']), seeds=len(config['seeds']), paired_runs=len(records),
             all_summary_rows=len(rows)))
    expected = len(config['datasets']) * len(config['backbones']) * len(config['seeds'])
    print('summary:', len(records), '/', expected, 'paired runs;', len(missing), 'missing', flush=True)
    return not missing
