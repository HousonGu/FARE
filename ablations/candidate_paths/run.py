"""Fit the candidate-path controls (learned B+S, learned B+F, constant alpha and beta) and score test weeks."""
from pathlib import Path
import argparse
import json
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from io_utils import VARIANTS, metric, seal, verify, write_json
from model import fit, predict


def run(prepared, out, variants=VARIANTS, threads=4):
    prepared = Path(prepared).resolve(); out = Path(out).resolve()
    if out == prepared or out.is_relative_to(prepared):
        raise ValueError('Use a separate output directory; prepared inputs are read-only')
    prepared_identity = verify(prepared)
    meta = json.loads((prepared / 'metadata.json').read_text())
    if not variants or len(set(variants)) != len(variants) or any(v not in VARIANTS for v in variants):
        raise ValueError('Only bs_path, bf_path, both_constant are authorized training tasks')
    xdev = np.load(prepared / 'x_dev.npy', mmap_mode='r'); xtest = np.load(prepared / 'x_test.npy', mmap_mode='r')
    pdev = np.load(prepared / 'parts_dev.npy', mmap_mode='r'); ptest = np.load(prepared / 'parts_test.npy', mmap_mode='r')
    ydev = np.load(prepared / 'y_dev.npy', mmap_mode='r'); ytest = np.load(prepared / 'y_test.npy', mmap_mode='r')
    keyfile = prepared / 'keys_test.parquet'
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for variant in variants:
        identity = dict(protocol='candidate-paths-v1', prepared=prepared_identity,
                        variant=variant,
                        production=meta['production'])
        dest = out / variant
        if (dest / 'complete.json').exists():
            if verify(dest) != identity:
                raise ValueError('Completed task has another identity; do not overwrite it')
            print('SKIP existing completed', variant, flush=True)
            rows.append(json.loads((dest / 'metrics.json').read_text())); continue
        if dest.exists():
            raise ValueError('Partial task exists; inspect its status and use a new output folder, no automatic rerun')
        dest.mkdir(); lock = dest / 'RUNNING.lock'; lock.write_text('running', encoding='utf-8')
        started = time.time()
        write_json(dest / 'status.json', dict(status='running', variant=variant))
        try:
            theta, record = fit(xdev, pdev, ydev, variant, threads=threads)
            np.save(dest / 'parameters.npy', theta); write_json(dest / 'fit.json', record)
            # Emit one test origin at a time. No fitting operation sees a test label.
            import pyarrow as pa
            import pyarrow.parquet as pq
            writer = None; weekly = []
            try:
                for week in meta['splits']['test']['weeks']:
                    sl = slice(week['start'], week['stop'])
                    keys = pd.read_parquet(keyfile, filters=[('origin', '==', week['origin'])])
                    with threadpool_limits(limits=threads):
                        prediction, alpha, beta = predict(theta, xtest[sl], ptest[sl], variant)
                    if len(keys) != len(prediction):
                        raise ValueError('Prediction/keys coverage mismatch')
                    keys = keys.reset_index(drop=True)
                    keys['actual'] = ytest[sl]; keys['forecast'] = prediction
                    keys['alpha'] = alpha; keys['beta'] = beta
                    keys['weight_backbone'] = 1 - beta; keys['weight_shape'] = beta * (1 - alpha)
                    keys['weight_factor'] = beta * alpha
                    keys['backbone'] = ptest[sl, 0]
                    table = pa.Table.from_pandas(keys, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(dest / 'predictions.parquet', table.schema, compression='zstd')
                    writer.write_table(table)
                    weekly.append(dict(origin=week['origin'], variant=metric(ytest[sl], prediction)))
            finally:
                if writer is not None:
                    writer.close()
            def total(method):
                scores = [w[method] for w in weekly]
                count = sum(s['rows'] for s in scores); sales = sum(s['actual_sum'] for s in scores)
                ae = sum(s['absolute_error_sum'] for s in scores)
                return dict(rows=count, actual_sum=sales, absolute_error_sum=ae,
                            wape=ae / sales if sales else None, mae=ae / count)
            overall = {'variant': total('variant')}
            result = dict(dataset=meta['identity']['dataset'], variant=variant, protocol=identity['protocol'],
                          overall=overall, weekly=weekly,
                          interpretation=('candidate output-path intervention; removed candidate information remains in z'
                                          if variant != 'both_constant' else 'both gate logits are learned constants'))
            write_json(dest / 'metrics.json', result)
            write_json(dest / 'status.json', dict(status='complete', variant=variant))
            seal(dest, identity); rows.append(result)
            print('DONE', variant, flush=True)
        except BaseException as exc:
            write_json(dest / 'status.json', dict(status='failed', error=str(exc), variant=variant))
            raise
        finally:
            lock.unlink(missing_ok=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    p.add_argument('--threads', type=int, default=4)
    a = p.parse_args()
    if a.threads < 1:
        p.error('--threads must be positive')
    run(a.prepared, a.out, a.variants, a.threads)
