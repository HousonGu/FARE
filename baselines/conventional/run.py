"""Conventional baselines: seasonal naive, ETS, SARIMA, Ridge and XGBoost."""
import os
_cpus = str(os.cpu_count() or 1)
# Set before NumPy/XGBoost load their thread pools. A parent value of 1 must not stick.
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = _cpus
os.environ['NUMBA_NUM_THREADS'] = '1'
import argparse
import gc
import json
from pathlib import Path
import signal
import sys
import time
import traceback
import uuid
import numpy as np
import pandas as pd
from data import Panel, EXPECTED
from io_utils import (HERE, KEYS, environment, merge, metric, pool, save_csv,
                      save_parquet, seal, verify,
                      verify_file, write_json)
from models import candidates, fit_model, load_model, predict_model

ROOT = HERE.parents[1]


class StopRequested(KeyboardInterrupt):
    pass


def stopped(signum, frame):
    raise StopRequested('Termination requested; resume the same output directory')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def trained(panel, method, params, origins, folder, config, seed):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if verify(folder, 'trained.json'):
        return load_model(method, folder, config)
    started = time.monotonic()
    predictor = fit_model(panel, method, params, origins, folder, config, seed=seed)
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix in ['.json', '.npz', '.ubj']
             and p.name != 'trained.json' and '.tmp' not in p.name]
    if not files:
        raise ValueError('Model stage did not save a reloadable artifact')
    seal(folder, files, dict(method=method, seed=seed, fit_seconds=time.monotonic()-started), 'trained.json')
    return predictor


def score_origins(panel, predictor, origins, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    for origin in origins:
        path = folder / f'{origin}.parquet'
        if path.is_file():
            verify_file(path)
            predictions = pd.read_parquet(path)
        else:
            values = np.asarray(predict_model(predictor, panel, origin), dtype=np.float64)
            if values.shape != (len(panel.rows(origin)), 7):
                raise ValueError('Prediction shape differs from frozen keys')
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError('Prediction is nonfinite or negative')
            predictions = panel.keys(origin)
            predictions['forecast'] = values.ravel()
            save_parquet(path, predictions)
        scored = merge(panel.truth(origin), predictions)
        record = dict(origin=origin, **metric(scored.actual, scored.forecast))
        records.append(record)
        print(panel.dataset, predictor.method, origin, record['rows'],
              f"WAPE={record['wape']:.8f}", flush=True)
    save_csv(folder / 'weekly_metrics.csv', pd.DataFrame(records))
    write_json(folder / 'metrics.json', pool(records))
    seal(folder, [folder/'weekly_metrics.csv', folder/'metrics.json'] +
         [folder/f'{origin}.parquet' for origin in origins],
         dict(origins=list(origins), status='scored'))
    return pool(records)


def select(panel, method, folder, config):
    folder = Path(folder)
    path = folder / 'selection.json'
    if path.exists():
        verify_file(path)
        return read_json(path)['selected']
    results = []
    for index, params in enumerate(candidates(method, config)):
        trial = folder / f'trial_{index:02d}'
        predictor = trained(panel, method, params, panel.schedule['train'], trial/'model',
                            config, config['selection_seed'])
        values = score_origins(panel, predictor, panel.schedule['tune'], trial/'validation')
        results.append(dict(index=index, params=params, validation=values))
        del predictor
        gc.collect()
    selected = min(results, key=lambda r: (r['validation']['wape'], r['index']))
    write_json(path, dict(method=method, selected=selected, trials=results,
                         train=panel.schedule['train'], tune=panel.schedule['tune']))
    return selected


def task(panel, method, folder, config):
    selected = select(panel, method, folder/'search', config)
    seeds = config['seeds'] if method == 'xgboost' else [config['selection_seed']]
    for seed in seeds:
        target = folder / str(seed)
        if verify(target):
            continue
        predictor = trained(panel, method, selected['params'], panel.schedule['fit'], target/'model', config, seed)
        values = score_origins(panel, predictor, panel.schedule['test'], target/'predictions')
        if (len(panel.schedule['test']), values['rows']) != EXPECTED[panel.dataset]:
            raise ValueError('Formal test coverage changed')
        write_json(target/'metrics.json', dict(dataset=panel.dataset, method=method, seed=seed, **values))
        seal(target, [target/'metrics.json', target/'model/trained.json', target/'predictions/complete.json'],
             dict(status='formal_new_baseline_complete', dataset=panel.dataset, method=method, seed=seed))
        del predictor
        gc.collect()


def initialize(out, config, data_root):
    expected = dict(protocol=config, environment=environment(),
                    data=[d for d in config['datasets'] if (data_root/d/'complete.json').exists()])
    manifest = out/'run_manifest.json'
    if manifest.exists():
        return
    if out.exists() and any(out.iterdir()):
        raise ValueError('Refusing to write into a nonempty directory without our run_manifest.json')
    out.mkdir(parents=True, exist_ok=True)
    write_json(manifest, expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['audit', 'all'], default='all')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=ROOT/'data/baselines',
                        help='Folder produced by baselines/prepare_inputs.py, one subfolder per dataset')
    parser.add_argument('--dataset', choices=list(EXPECTED))
    parser.add_argument('--method', choices=['seasonal_naive', 'ets', 'sarima', 'ridge', 'xgboost'])
    parser.add_argument('--threads', type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('threads must be positive')
    config = read_json(HERE/'config.json')
    config['n_jobs'] = config['xgb_nthread'] = args.threads
    out, data_root = args.out.resolve(), args.data_root.resolve()
    if out == HERE or out.is_relative_to(data_root):
        raise ValueError('Output must not be a source or data directory')
    initialize(out, config, data_root)
    lock = out/'RUNNING.lock'
    token = uuid.uuid4().hex
    try:
        with lock.open('x', encoding='utf-8') as stream:
            json.dump(dict(pid=os.getpid(), token=token), stream)
    except FileExistsError:
        raise RuntimeError('RUNNING.lock exists. Confirm its PID has stopped, then move that lock aside; do not delete an active lock.')
    for signame in ['SIGTERM', 'SIGINT', 'SIGBREAK']:
        if hasattr(signal, signame):
            signal.signal(getattr(signal, signame), stopped)
    try:
        datasets = [args.dataset] if args.dataset else config['datasets']
        methods = [args.method] if args.method else config['methods']
        audits = {}
        for dataset in datasets:
            panel = Panel(data_root/dataset)
            if args.stage == 'audit':
                audits[dataset] = panel.audit()
                write_json(out/'audit.json', audits)
            else:
                for method in methods:
                    task(panel, method, out/dataset/method, config)
        write_json(out/'status.json', dict(status='stage_complete', stage=args.stage))
        return 0
    except (StopRequested, KeyboardInterrupt):
        write_json(out/'status.json', dict(status='interrupted_resumable', stage=args.stage))
        return 75
    except Exception as error:
        write_json(out/'status.json', dict(status='failed_not_complete', error=str(error), traceback=traceback.format_exc()))
        raise
    finally:
        if lock.exists() and read_json(lock).get('token') == token:
            lock.unlink()


if __name__ == '__main__':
    sys.exit(main())
