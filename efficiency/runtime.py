"""Measured stages of FARE, TimeFuse and RAF at one test origin.

Each stage calls the same functions as the experiment runners:

  reference  FARE references S/F for one origin (exp.*_references), with every
             preceding completed origin in the library. Shared by FARE and TimeFuse.
  fusion     FARE: gate inputs, gate fit on development rows, test prediction
             (exp.*_gates, fare.fusion). TimeFuse: meta features, fusor training
             and prediction (baselines/timefuse). Both read the stored B/S/F
             candidates of the shared input folder.
  raf        RAF retrieval plan and Chronos-2 prediction (baselines/residual_raf).

``raw_bank`` is an identity hook between the library concatenation and the
distance computation; ``profiler.py --bank-fraction`` wraps it to subsample the
library for the scaling study.
"""
from __future__ import annotations

import builtins
from contextlib import contextmanager
import copy
import importlib
import inspect
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _path in (ROOT / 'src', ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
METHOD = os.environ.get('EFFICIENCY_METHOD', 'fare').lower()
DATASET = os.environ.get('EFFICIENCY_DATASET', 'm5').lower()
KEYS = ['origin', 'series_id', 'target_date']
_BACKEND = None


def backend():
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    if METHOD == 'raf':
        sys.path.insert(0, str(ROOT / 'baselines' / 'residual_raf'))
        raf = importlib.import_module('raf')
        backbone = importlib.import_module('backbone')
        data = importlib.import_module('data')
        result = dict(raf=raf, backbone=backbone, panel_class=data.Panel)
    else:
        if DATASET in ('favorita', 'm5'):
            reference = importlib.import_module('exp.retail_references')
            gates = importlib.import_module('exp.retail_gates')
            data_module = importlib.import_module('data_provider.retail')
            reference_function = reference.build
            select_owner = importlib.import_module('data_provider.retail_factors')
        elif DATASET in ('rossmann', 'rohlik'):
            reference = importlib.import_module(f'exp.{DATASET}_references')
            gates = importlib.import_module(f'exp.{DATASET}_gates')
            data_module = importlib.import_module(f'data_provider.{DATASET}')
            reference_function = reference.references
            select_owner = reference
        else:
            raise ValueError('Unknown dataset: ' + DATASET)
        sys.path.insert(0, str(ROOT / 'baselines' / 'conventional'))
        panel_class = importlib.import_module('data').Panel
        result = dict(reference=reference, gates=gates, data=data_module, panel_class=panel_class,
                      reference_function=reference_function, design_function=gates.design,
                      fit_function=gates.fit_gate, predict_function=gates.predict_gate,
                      select_owner=select_owner, select_function=select_owner.select)
        if METHOD == 'timefuse':
            sys.path.insert(0, str(ROOT / 'baselines' / 'timefuse'))
            result['timefuse'] = importlib.import_module('fusor')
        elif METHOD != 'fare':
            raise ValueError('Unknown method: ' + METHOD)
    _BACKEND = result
    return result


def source_functions():
    source = backend()
    if METHOD == 'raf':
        functions = {'build_retrieval_plan': source['raf'].build_retrieval_plan,
                     'augment_from_plan': source['raf'].augment_from_plan,
                     'predict_context': source['backbone'].predict_context}
    else:
        functions = {name: source[key] for name, key in (
            ('references', 'reference_function'), ('design', 'design_function'),
            ('fit_gate', 'fit_function'), ('predict_gate', 'predict_function'), ('select', 'select_function'))}
        if METHOD == 'timefuse':
            functions.update({name: getattr(source['timefuse'], name) for name in ('meta_row', 'train', 'predict')})
    return {name: dict(file=str(Path(inspect.getsourcefile(function)).resolve().relative_to(ROOT)),
                       signature=str(inspect.signature(function)))
            for name, function in functions.items()}


def raw_bank(history, future, factor1, factor2):
    """Identity hook after the library concatenation and before distances, axis=0."""
    if not (len(history) == len(future) == len(factor1) == len(factor2)):
        raise ValueError('Unaligned reference arrays')
    return history, future, factor1, factor2


def select_reference(shape, factor1, factor2):
    return backend()['select_function'](shape, factor1, factor2)


@contextmanager
def bank_hook(module):
    """Route the four-array map(np.concatenate, ...) of the reference builder through raw_bank."""
    sentinel = object()
    previous = module.__dict__.get('map', sentinel)
    select_owner = backend()['select_owner']
    previous_select = select_owner.select

    def mapped(function, iterable, *others):
        if function is np.concatenate and not others:
            arrays = tuple(builtins.map(function, iterable))
            if len(arrays) != 4 or arrays[0].ndim != 2 or arrays[0].shape[1] != 56:
                raise ValueError('Reference library concatenation shape changed')
            return raw_bank(*arrays)
        return builtins.map(function, iterable, *others)

    module.__dict__['map'] = mapped
    select_owner.select = select_reference
    try:
        yield
    finally:
        select_owner.select = previous_select
        if previous is sentinel:
            del module.__dict__['map']
        else:
            module.__dict__['map'] = previous


def load_reference_data():
    source = backend()
    if DATASET in ('favorita', 'm5'):
        return source['data'].RetailData(DATASET)
    from data_provider.paths import RAW_ROOT
    cls = source['data'].RossmannData if DATASET == 'rossmann' else source['data'].RohlikData
    return cls(RAW_ROOT / DATASET)


def reference_run(data, schedule, origin, output, query_limit=None):
    """Build references for one query origin with every preceding completed origin in the library."""
    source = backend()
    output.mkdir(parents=True, exist_ok=True)
    original_eligible = data.eligible
    if query_limit:
        def limited(value, *args, **kwargs):
            rows = original_eligible(value, *args, **kwargs)
            return rows[:query_limit] if value == origin and not kwargs.get('reference', False) else rows
        data.eligible = limited
    try:
        os.environ['EFFICIENCY_QUERY_COUNT'] = str(len(data.eligible(origin)))
        with bank_hook(source['reference']):
            if DATASET in ('favorita', 'm5'):
                prior = [o for split in ('fit', 'dev', 'test') for o in schedule[split]
                         if pd.Timestamp(o) + pd.Timedelta(days=6) < pd.Timestamp(origin)]
                source['reference_function'](data, dict(fit=prior, dev=[], test=[origin]), output)
                frame = pd.read_parquet(output / f'{origin}.parquet')
            else:
                frame = source['reference_function'](data, origin, batch=8)
                frame.to_parquet(output / f'{origin}.parquet', index=False)
        return frame
    finally:
        if query_limit:
            del data.__dict__['eligible']


def canonical(frame):
    frame = frame.copy()
    for name in KEYS:
        frame[name] = frame[name].astype(str)
    return frame


def load_candidates(panel, origin, smoke_queries=None):
    """Stored references S/F and Chronos-2 forecasts B; development rows carry labels."""
    chunks = []
    selected = set(panel.ids[:smoke_queries]) if smoke_queries else None
    for current in panel.schedule['dev'] + [origin]:
        reference = canonical(pd.read_parquet(panel.folder / 'references' / f'{current}.parquet'))
        backbone = canonical(pd.read_parquet(panel.folder / 'backbone' / f'{current}.parquet'))
        backbone = backbone.rename(columns={'forecast': 'backbone_forecast'})
        frame = reference.merge(backbone[KEYS + ['backbone_forecast']], on=KEYS,
                                how='outer', validate='one_to_one', indicator=True)
        if not frame._merge.eq('both').all():
            raise ValueError('Reference/backbone candidate keys differ')
        frame = frame.drop(columns='_merge')
        if selected is not None:
            frame = frame[frame.series_id.isin(selected)]
        if current in panel.schedule['dev']:
            truth = canonical(panel.truth(current))
            frame = frame.merge(truth, on=KEYS, how='left', validate='one_to_one')
            if frame.actual.isna().any():
                raise ValueError('Missing development labels')
        else:
            frame['actual'] = np.nan
        chunks.append(frame)
    result = pd.concat(chunks, ignore_index=True).sort_values(KEYS).reset_index(drop=True)
    if not result.origin.eq(origin).any() or not result.origin.isin(panel.schedule['dev']).any():
        raise ValueError('Empty development or query rows')
    os.environ['EFFICIENCY_QUERY_COUNT'] = str(len(result[result.origin.eq(origin)]) // 7)
    return result


def design_fare(frame, train, dataset):
    function = backend()['design_function']
    return function(frame, train, dataset) if dataset in ('favorita', 'm5') else function(frame, train)


def fit_fare(x, actual, backbone, shape, factor):
    return backend()['fit_function'](x, actual, backbone, shape, factor)


def predict_fare(x, parameters, backbone, shape, factor):
    return backend()['predict_function'](x, parameters, backbone, shape, factor)


def timefuse_features(panel, frame):
    """TimeFuse meta features from the 56-day histories, as in baselines/timefuse/run.py."""
    source = backend()['timefuse']
    counts = dict(constant=0, exception=0, nonfinite=0)
    meta = []
    for origin, week in frame.groupby('origin', sort=True):
        ids = week.series_id.to_numpy()[::7]
        rows = np.array([panel.id_rows[v] for v in ids])
        histories = panel.values[np.ix_(rows, panel.indices(origin)[:56])]
        meta.append(np.vstack([source.meta_row(h.astype(float), counts) for h in histories]))
    parts = frame[['backbone_forecast', 'shape_retrieval', 'factor_retrieval']].to_numpy().reshape(-1, 7, 3).transpose(0, 2, 1)
    actual = frame.actual.to_numpy().reshape(-1, 7)
    return np.vstack(meta), parts, actual, counts


def timefuse_dev_features(panel, frame):
    return timefuse_features(panel, frame)


def timefuse_test_features(panel, frame):
    return timefuse_features(panel, frame)


def fit_timefuse(meta, parts, actual, device):
    import torch
    return backend()['timefuse'].train(meta, parts, actual, torch.device(device))


def predict_timefuse(model, meta, parts, mean, std, device):
    import torch
    return backend()['timefuse'].predict(model, meta, parts, mean, std, torch.device(device))


def raf_plan(panel, pipeline, model_path, output, origin, device, batch_size):
    source = backend()['raf']
    embedder = source.Chronos2Embedder(model_path=model_path, pipeline=pipeline,
                                      device=device, batch_size=batch_size, dtype='float32')
    return source.build_retrieval_plan(panel, embedder, output, origins=[origin],
                                       rows_per_chunk=16, distance_device=device, resume=False)


def raf_predict(panel, pipeline, plan, origin, batch_size):
    source = backend()
    rows = panel.rows(origin)
    chunks = []
    for start in range(0, len(rows), batch_size):
        selected = rows[start:start+batch_size]
        augmented = source['raf'].augment_from_plan(panel, origin, selected, plan)
        prediction = source['backbone'].predict_context(pipeline, panel, origin, selected,
            augmented.context119, past_covariates=augmented.past_covariates119, batch_size=batch_size)
        chunks.append(np.maximum(augmented.denormalize(prediction), 0))
    return np.concatenate(chunks)


def limited_raf_panel(original, maximum, origin):
    panel = copy.copy(original)
    selected = original.rows(origin)[:maximum]
    panel.ids = original.ids[selected]
    panel.first = original.first[selected]
    panel.values = np.asarray(original.values[selected])
    panel.covariates = np.asarray(original.covariates[selected])
    panel.id_rows = {str(value): index for index, value in enumerate(panel.ids)}
    panel.meta = copy.deepcopy(original.meta)
    panel.meta['ids'] = panel.ids.tolist()
    panel.meta['first'] = panel.first.astype(str).tolist()
    panel.meta['eligible_rows'] = {origin: np.flatnonzero(np.isin(selected, original.rows(origin))).tolist()
        for split in ('fit', 'dev', 'test') for origin in panel.schedule[split]}
    panel._indices = {}
    return panel


def run(args):
    if args.method != METHOD or args.dataset != DATASET:
        raise ValueError('EFFICIENCY_METHOD/EFFICIENCY_DATASET must match the entry arguments before profiling starts')
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('Use a fresh output directory: ' + str(output))
    source = backend()
    panel = source['panel_class'](args.inputs / DATASET)
    if args.phase == 'raf' and args.smoke_queries:
        tests = panel.schedule['test']
        selected_origin = {'first': tests[0], 'middle': tests[len(tests)//2], 'last': tests[-1]}[args.position]
        panel = limited_raf_panel(panel, args.smoke_queries, selected_origin)
    schedule = panel.schedule
    test = schedule['test']
    origin = {'first': test[0], 'middle': test[len(test)//2], 'last': test[-1]}[args.position]
    output.mkdir(parents=True)
    metadata = dict(method=METHOD, dataset=DATASET, phase=args.phase, position=args.position,
        origin=origin, smoke_queries=args.smoke_queries, full_query_set=args.smoke_queries is None,
        cached_backbone=args.phase == 'fusion', inputs=str((args.inputs / DATASET).resolve()),
        source_functions=source_functions(),
        internal_gate_optimizer_threads=4 if METHOD == 'fare' and args.phase == 'fusion' else None,
        note='Efficiency measurements only; these predictions are not used for accuracy.')
    if args.phase == 'reference':
        if METHOD == 'raf':
            raise ValueError('Use --phase raf for RAF')
        data = load_reference_data()
        if hasattr(data, 'schedule') and any(data.schedule[split] != schedule[split] for split in ('fit', 'dev', 'test')):
            raise ValueError('Raw-data schedule differs from the shared input folder')
        frame = reference_run(data, schedule, origin, output / 'references', args.smoke_queries)
        metadata['daily_prediction_rows'] = len(frame)
    elif args.phase == 'fusion':
        if METHOD == 'raf':
            raise ValueError('RAF has no fusion fit')
        frame = load_candidates(panel, origin, args.smoke_queries)
        train = frame.origin.isin(schedule['dev']).to_numpy()
        if METHOD == 'fare':
            x, parts, design = design_fare(frame, train, DATASET)
            parameters, fitted = fit_fare(x[train], frame.actual.to_numpy()[train], *parts[train].T)
            prediction, beta, alpha = predict_fare(x[~train], parameters, *parts[~train].T)
            metadata.update(design=design, fit=fitted)
        else:
            dev = timefuse_dev_features(panel, frame.loc[train])
            test_arrays = timefuse_test_features(panel, frame.loc[~train])
            model, mean, std = fit_timefuse(*dev[:3], args.device)
            weights, weekly_prediction = predict_timefuse(model, *test_arrays[:2], mean, std, args.device)
            prediction = weekly_prediction.ravel()
            metadata['replacement_counts'] = dict(dev=dev[3], test=test_arrays[3])
        keys = frame.loc[~train, KEYS].copy()
        keys['forecast'] = np.maximum(prediction, 0)
        keys.to_parquet(output / 'predictions.parquet', index=False)
        metadata.update(development_rows=int(train.sum()), daily_prediction_rows=int((~train).sum()))
    else:
        if METHOD != 'raf':
            raise ValueError('The raf phase requires --method raf')
        os.environ['EFFICIENCY_QUERY_COUNT'] = str(len(panel.rows(origin)))
        pipeline = source['backbone'].load_pipeline(args.model_path, args.device)
        plan = raf_plan(panel, pipeline, args.model_path, output / 'retrieval_plan', origin,
                        args.device, args.batch_size)
        prediction = raf_predict(panel, pipeline, plan, origin, args.batch_size)
        keys = panel.keys(origin)
        keys['forecast'] = prediction.ravel()
        keys.to_parquet(output / 'predictions.parquet', index=False)
        metadata['daily_prediction_rows'] = len(keys)
    metadata['query_count'] = int(os.environ['EFFICIENCY_QUERY_COUNT'])
    (output / 'measurement.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2,
                                                       default=str, allow_nan=False), encoding='utf-8')
    print(json.dumps({key: metadata[key] for key in ('method', 'dataset', 'phase', 'origin', 'query_count')}, ensure_ascii=False))


# Resolve the imports and signatures during profiler.py --check, without reading data.
if os.environ.get('EFFICIENCY_METHOD'):
    source_functions()
