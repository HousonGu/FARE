"""RAF and LightGBM residual correction on the frozen Chronos-2 forecasts. No backbone fit."""
import argparse
import gc
import json
import os
from pathlib import Path
import signal
import numpy as np
import pandas as pd
from data import Panel, EXPECTED
from backbone import FrozenBackbone, load_pipeline, predict_context
from io_utils import HERE, merge, write_json, environment, verify_file, save_parquet, metric, pool, seal

ROOT = HERE.parents[1]
STOP = False


def request_stop(*_):
    global STOP
    STOP = True


def checkpoint_stop():
    if STOP:
        raise InterruptedError('Stop requested; resume same output directory')


def predicted_file(panel, origin, path, compute):
    if path.exists():
        verify_file(path)
        return merge(panel.keys(origin), pd.read_parquet(path))
    chunks = []
    for rows in compute('rows'):
        checkpoint_stop()
        values = compute(rows)
        frame = panel.keys(origin, rows)
        frame['forecast'] = np.asarray(values).ravel()
        if not np.isfinite(frame.forecast).all() or (frame.forecast < 0).any():
            raise ValueError('Invalid prediction')
        chunks.append(frame)
    if not chunks:
        raise ValueError('Empty evaluation origin')
    frame = merge(panel.keys(origin), pd.concat(chunks, ignore_index=True))
    save_parquet(path, frame)
    return frame


def run_raf(panel, out, args):
    from raf import Chronos2Embedder, build_retrieval_plan, augment_from_plan
    import torch
    checkpoint_stop()
    pipeline = load_pipeline(args.model_path, args.device)
    embedder = Chronos2Embedder(model_path=args.model_path, pipeline=pipeline,
                                device=args.device, batch_size=args.batch_size, dtype='float32')
    plan = build_retrieval_plan(panel, embedder, out/'retrieval_plan', origins=panel.schedule['test'],
                                rows_per_chunk=16, distance_device=args.device)
    for origin in panel.schedule['test']:
        checkpoint_stop()
        rows_all = panel.rows(origin)

        def compute(rows):
            if isinstance(rows, str):
                return [rows_all[i:i+args.batch_size] for i in range(0, len(rows_all), args.batch_size)]
            batch = augment_from_plan(panel, origin, rows, plan)
            pred = predict_context(pipeline, panel, origin, rows, batch.context119,
                                   past_covariates=batch.past_covariates119, batch_size=args.batch_size)
            return np.maximum(batch.denormalize(pred), 0)
        predicted_file(panel, origin, out/'raf'/'predictions'/f'{origin}.parquet', compute)
        print(panel.dataset, 'raf', origin, flush=True)
    del embedder, pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_residual(panel, provider, out, args, seeds):
    from residual import train_residual, predict_residual
    for seed in seeds:
        checkpoint_stop()
        root = out/f'residual_seed_{seed}'
        predictor = train_residual(panel, provider, root/'model', seed, threads=args.threads,
                                   batch_rows=args.feature_batch_rows, rounds=300,
                                   cache_dir=out/'residual_matrix')
        for origin in panel.schedule['test']:
            rows_all = panel.rows(origin)

            def compute(rows):
                if isinstance(rows, str):
                    return [rows_all[i:i+args.feature_batch_rows]
                            for i in range(0, len(rows_all), args.feature_batch_rows)]
                return predict_residual(predictor, panel, provider, origin, rows)
            predicted_file(panel, origin, root/'predictions'/f'{origin}.parquet', compute)
            print(panel.dataset, 'residual', seed, origin, flush=True)


def score_task(panel, provider, root, method):
    records = []
    for origin in panel.schedule['test']:
        truth = panel.truth(origin)
        if method == 'backbone':
            prediction = provider(origin, panel.rows(origin)).ravel()
        else:
            path = root/'predictions'/f'{origin}.parquet'
            if not path.exists():
                return None
            verify_file(path)
            truth = merge(truth, pd.read_parquet(path))
            prediction = truth.forecast.to_numpy()
        records.append(dict(origin=origin, **metric(truth.actual.to_numpy(), prediction)))
    combined = pool(records)
    if (len(records), combined['rows']) != EXPECTED[panel.dataset]:
        raise ValueError('Test coverage changed')
    write_json(root/'metrics.json', dict(method=method, dataset=panel.dataset, per_origin=records, pooled=combined))
    files = [root/'metrics.json'] + list((root/'predictions').glob('*'))
    seal(root, files, extra={'method': method, 'dataset': panel.dataset, 'rows': combined['rows']})
    return combined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['audit', 'all'], default='all')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=ROOT/'data/baselines',
                        help='Folder produced by baselines/prepare_inputs.py, one subfolder per dataset')
    parser.add_argument('--dataset', choices=list(EXPECTED), action='append')
    parser.add_argument('--method', choices=['both', 'raf', 'residual'], default='both')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--feature-batch-rows', type=int, default=2048)
    parser.add_argument('--model-path', type=Path, default=ROOT/'models/chronos-2',
                        help='Local amazon/chronos-2 folder')
    args = parser.parse_args()
    if min(args.threads, args.batch_size, args.feature_batch_rows) < 1:
        parser.error('Batch sizes and threads must be positive')
    config = json.loads((HERE/'config.json').read_text())
    datasets = args.dataset or list(EXPECTED)
    args.out.mkdir(parents=True, exist_ok=True)
    for event in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(event, request_stop)
    identity = {'config': config, 'environment': environment(), 'device': args.device,
                'threads': args.threads, 'batch_size': args.batch_size,
                'feature_batch_rows': args.feature_batch_rows, 'mode': args.stage}
    manifest = args.out/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError('Changed mode/code/environment/config; use a new output directory')
    lock = args.out/'RUNNING.lock'
    with lock.open('x') as handle:
        handle.write(str(os.getpid()))
    try:
        write_json(manifest, identity)
        os.environ['OMP_NUM_THREADS'] = str(args.threads)
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=args.threads):
            for name in datasets:
                checkpoint_stop()
                panel = Panel(args.data_root/name)
                provider = FrozenBackbone(panel)
                if args.stage == 'audit':
                    write_json(args.out/name/'audit.json', panel.audit())
                    continue
                root = args.out/name
                methods = ['backbone']
                if args.method in ['both', 'raf']:
                    run_raf(panel, root, args)
                    methods.append('raf')
                if args.method in ['both', 'residual']:
                    run_residual(panel, provider, root, args, config['residual_seeds'])
                    methods += [f'residual_seed_{s}' for s in config['residual_seeds']]
                for method in methods:
                    score_task(panel, provider, root/method, method)
        return 0
    except InterruptedError as exc:
        print(str(exc), flush=True)
        return 75
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    raise SystemExit(main())
