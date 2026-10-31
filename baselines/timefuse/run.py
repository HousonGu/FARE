"""TimeFuse on the same three stored candidates (Chronos-2 B, shape S, factor F).

Meta features are computed from each query's 56-day history, the fusor is trained
on development weeks only, and test forecasts are written once.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'baselines' / 'conventional'))
from data import Panel  # noqa: E402
from io_utils import KEYS, merge, metric, pool, save_parquet, write_json  # noqa: E402
from fusor import FEATURE_NAMES, ModelFusor, SEED, EPOCHS, meta_row, train, predict  # noqa: E402

DATASETS = ('favorita', 'm5', 'rossmann', 'rohlik')


def candidates(panel, origin):
    ref = pd.read_parquet(panel.folder/'references'/f'{origin}.parquet')[KEYS+['shape_retrieval', 'factor_retrieval']]
    backbone = pd.read_parquet(panel.folder/'backbone'/f'{origin}.parquet')
    frame = merge(ref, backbone).sort_values(KEYS).reset_index(drop=True)
    parts = frame[['forecast', 'shape_retrieval', 'factor_retrieval']].to_numpy(dtype=np.float64)
    if not np.isfinite(parts).all() or (parts < 0).any():
        raise ValueError('Invalid stored candidates ' + origin)
    return frame


def run(panel, dest, device):
    dest.mkdir(parents=True, exist_ok=True)
    counts = dict(constant=0, exception=0, nonfinite=0)
    lookup = {v: i for i, v in enumerate(panel.ids)}
    blocks = {}
    for split in ('dev', 'test'):
        chunks = []
        for origin in panel.schedule[split]:
            frame = candidates(panel, origin)
            ids = frame.series_id.to_numpy()[::7]
            rows = np.array([lookup[v] for v in ids])
            cache = dest/f'{origin}_meta.npz'
            if cache.exists():
                with np.load(cache, allow_pickle=False) as z:
                    meta = z['meta']
                    if not np.array_equal(z['ids'], ids.astype(str)):
                        raise ValueError('Meta cache keys differ')
                    for k, v in zip(counts, z['counts']):
                        counts[k] += int(v)
            else:
                local = dict(constant=0, exception=0, nonfinite=0)
                histories = panel.values[np.ix_(rows, panel.indices(origin)[:56])]
                meta = np.vstack([meta_row(h.astype(float), local) for h in histories])
                tmp = dest/f'{origin}_meta.tmp.npz'
                np.savez_compressed(tmp, meta=meta, ids=ids.astype(str), counts=np.array(list(local.values())))
                tmp.replace(cache)
                for k, v in local.items():
                    counts[k] += v
            if meta.shape != (len(ids), len(FEATURE_NAMES)) or not np.isfinite(meta).all():
                raise ValueError('Invalid meta features')
            parts = frame[['forecast', 'shape_retrieval', 'factor_retrieval']].to_numpy().reshape(-1, 7, 3).transpose(0, 2, 1)
            actual = None
            if split == 'dev':
                actual = merge(frame[KEYS], panel.truth(origin)).actual.to_numpy().reshape(-1, 7)
            chunks.append((frame[KEYS], meta, parts, actual))
            print('timefuse features', panel.dataset, split, origin, len(ids), flush=True)
        blocks[split] = chunks
    model_path, fit_path = dest/'model.pt', dest/'fit.json'
    if model_path.exists() and fit_path.exists():
        saved = json.loads(fit_path.read_text(encoding='utf-8'))
        fusor = ModelFusor(len(FEATURE_NAMES), 3).to(torch.device(device))
        fusor.load_state_dict(torch.load(model_path, map_location=torch.device(device), weights_only=True))
        fusor.eval()
        mean, std = np.asarray(saved['mean'], float), np.asarray(saved['std'], float)
    else:
        dev = blocks['dev']
        fusor, mean, std = train(np.concatenate([x[1] for x in dev]), np.concatenate([x[2] for x in dev]),
                                 np.concatenate([x[3] for x in dev]), torch.device(device))
        torch.save(fusor.state_dict(), model_path)
        write_json(fit_path, dict(mean=mean.tolist(), std=std.tolist(), replacement_counts=counts,
                                  seed=SEED, epochs=EPOCHS))
    records = []
    for keys, meta, parts, _ in blocks['test']:
        _, forecast = predict(fusor, meta, parts, mean, std, torch.device(device))
        keys = keys.copy()
        keys['forecast'] = np.maximum(forecast.ravel(), 0)
        origin = keys.origin.iloc[0]
        save_parquet(dest/'predictions'/f'{origin}.parquet', keys)
        scored = merge(panel.truth(origin), keys)
        records.append(dict(origin=origin, **metric(scored.actual, scored.forecast)))
    write_json(dest/'metrics.json', dict(dataset=panel.dataset, method='timefuse', per_origin=records,
                                         pooled=pool(records)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT/'data/baselines',
                        help='Folder produced by baselines/prepare_inputs.py, one subfolder per dataset')
    parser.add_argument('--dataset', choices=DATASETS, action='append')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    for dataset in args.dataset or DATASETS:
        run(Panel(args.data_root/dataset), args.out/dataset/'timefuse', args.device)


if __name__ == '__main__':
    main()
