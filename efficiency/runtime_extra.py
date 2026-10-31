"""Measured one-time and per-origin stages of the further cost-table methods.

Methods and phases (the experiment files are called unchanged):

  backbone  backbone        Chronos-2 forecast B for one test origin (56-day input)
            backbone_dev    Chronos-2 forecast B for every development origin
  fare      dev_reference   FARE/TimeFuse shape and factor references for every dev origin
  raf       raf_plan        RAF retrieval plan for first/middle/last test origins at once
            raf_test        RAF augmentation and 119-day inference for one origin
  residual  fit | test      LightGBM residual corrector on frozen Chronos-2
  pir       fit | test      PIR revision of frozen Chronos-2
  pfrp      fit | test      PFRP with its own DLinear backbone
  saraf     fit | test      SARAF with its own linear backbone

"fit" phases cover everything a method needs before the test period and run once
per repetition. "test" phases load the fit output of the same repetition and
forecast one test origin. Every timed step is a module-level function, so the
profiler wraps it in place. Data loading and artifact re-verification happen
outside the timed functions.

Only one seed (the first seed of each configuration) is timed, because the
protocols repeat identical work per seed.
"""
from __future__ import annotations

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
PROJECT = HERE.parent
sys.path[:0] = [str(PROJECT), str(PROJECT / 'src')]
METHOD = os.environ.get('EFFICIENCY_METHOD', 'pir').lower()
DATASET = os.environ.get('EFFICIENCY_DATASET', 'm5').lower()
EXTRA_METHODS = ('backbone', 'residual', 'pir', 'pfrp', 'saraf')
EXTRA_PHASES = ('dev_reference', 'raf_plan', 'raf_test', 'backbone', 'backbone_dev', 'fit', 'test')
BASE_DATASETS = ('favorita', 'm5')
KEYS = ['origin', 'series_id', 'target_date']
_BACKEND = None


class NoBudget:
    """Budget objects only stop long batches; measured runs never stop early."""
    def tick(self):
        return None


def data_root(method):
    return Path(os.environ.get('EFFICIENCY_INPUTS', PROJECT / 'data/baselines')) / DATASET


def backend():
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    if METHOD in ('backbone', 'residual', 'raf'):
        root = (PROJECT / 'baselines/residual_raf').resolve()
        sys.path.insert(0, str(root))
        result = dict(root=root, data=importlib.import_module('data'),
                      backbone=importlib.import_module('backbone'))
        if METHOD == 'residual':
            result['residual'] = importlib.import_module('residual')
        if METHOD == 'raf':
            result['raf'] = importlib.import_module('raf')
    elif METHOD in ('pir', 'pfrp', 'saraf'):
        root = (PROJECT / 'baselines' / METHOD).resolve()
        if not (root / 'engine.py').is_file():
            raise FileNotFoundError('Method source missing: ' + str(root))
        sys.path.insert(0, str(root))
        result = dict(root=root, data=importlib.import_module('data'),
                      engine=importlib.import_module('engine'),
                      author=importlib.import_module('author'),
                      io_utils=importlib.import_module('io_utils'),
                      config=json.loads((root / 'config.json').read_text(encoding='utf-8')))
        if METHOD in ('pir', 'saraf'):
            result['retrieval'] = importlib.import_module('retrieval')
    elif METHOD == 'fare':
        import runtime as fare_runtime
        result = dict(fare=fare_runtime)
    else:
        raise ValueError('Unknown method: ' + METHOD)
    _BACKEND = result
    return result


def source_functions():
    """Functions this module dispatches to, with signatures."""
    source = backend()
    if METHOD == 'backbone':
        functions = dict(load_pipeline=source['backbone'].load_pipeline,
                         predict_context=source['backbone'].predict_context)
    elif METHOD == 'residual':
        functions = dict(train_residual=source['residual'].train_residual,
                         predict_residual=source['residual'].predict_residual,
                         frozen_backbone=source['backbone'].FrozenBackbone)
    elif METHOD == 'raf':
        functions = dict(build_retrieval_plan=source['raf'].build_retrieval_plan,
                         augment_from_plan=source['raf'].augment_from_plan,
                         retrieval_plan=source['raf'].RetrievalPlan,
                         predict_context=source['backbone'].predict_context)
    elif METHOD == 'pir':
        engine = source['engine']
        functions = dict(make_bank=engine.make_bank, cache_week=engine.cache_week, train=engine.train,
                         arrays_for=engine.arrays_for, tensors=engine.tensors,
                         RetailPIR=source['author'].RetailPIR)
    elif METHOD == 'pfrp':
        engine = source['engine']
        functions = dict(train=engine.train, memory=engine.memory, construct=engine.construct,
                         forecast=engine.forecast, author_api=source['author'].author_api)
    elif METHOD == 'saraf':
        engine = source['engine']
        functions = dict(Bank=source['retrieval'].Bank, retrieve_week=source['retrieval'].retrieve_week,
                         TrainingWindows=engine.TrainingWindows, train=engine.train,
                         predict_origin=engine.predict_origin, construct=source['author'].construct)
    else:
        fare = source['fare'].backend()
        functions = dict(references=fare['reference_function'])
    result = {}
    for name, function in functions.items():
        try:
            signature = str(inspect.signature(function))
        except (TypeError, ValueError):
            signature = '(class)'
        result[name] = dict(file=inspect.getsourcefile(function), signature=signature)
    return result


# ----------------------------------------------------------------- helpers
def position_origin(schedule, position):
    test = schedule['test']
    return {'first': test[0], 'middle': test[len(test) // 2], 'last': test[-1]}[position]


def limit_rows(panel, maximum):
    """Smoke only: keep the first eligible series of every origin."""
    original = panel.rows

    def limited(origin):
        return original(origin)[:maximum]
    panel.rows = limited
    return panel


def set_queries(count):
    os.environ['EFFICIENCY_QUERY_COUNT'] = str(int(count))
    os.environ['EFFICIENCY_QUERY_COUNT'] = str(int(count))


def torch_setup(threads, deterministic=False):
    import torch
    torch.set_num_threads(int(threads))
    if deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.use_deterministic_algorithms(True)
    from threadpoolctl import threadpool_limits
    return threadpool_limits(limits=int(threads))


def smoke_config(config, smoke):
    """Explicitly shortened protocol for smoke runs; full runs keep the configuration."""
    config = copy.deepcopy(config)
    if not smoke:
        return config
    for key in ('epochs', 'encoder_epochs', 'fusion_epochs'):
        if key in config:
            config[key] = 1
    if 'medoids' in config:
        config['medoids'] = max(config.get('topk', 8), 16)
        config['medoid_max_iter'] = 2
    if 'tune_weeks' in config:
        config['tune_weeks'] = 1
    return config


def save_forecast(keys, values, output):
    keys = keys.copy()
    keys['forecast'] = np.maximum(np.asarray(values, dtype=float).ravel(), 0)
    keys.to_parquet(output / 'predictions.parquet', index=False)
    return len(keys)


# ----------------------------------------------------------------- Chronos-2 backbone B
def load_addon_panel(panels, smoke):
    """Smoke keeps every series for residual: stored-B merges require complete keys."""
    source = backend()
    panel = source['data'].Panel(Path(panels) / DATASET)
    return limit_rows(panel, smoke) if smoke and METHOD != 'residual' else panel


def predict_backbone(panel, pipeline, origin, chunk, batch_size):
    """predict_context on the 56-day input, i.e. how B is produced."""
    source = backend()
    rows = panel.rows(origin)
    parts = []
    for start in range(0, len(rows), chunk):
        selected = rows[start:start + chunk]
        parts.append(np.maximum(source['backbone'].predict_context(
            pipeline, panel, origin, selected, panel.context(origin, selected), batch_size=batch_size), 0))
    return np.concatenate(parts)


def predict_backbone_dev(panel, pipeline, origins, chunk, batch_size, output):
    total = 0
    for origin in origins:
        values = predict_backbone(panel, pipeline, origin, chunk, batch_size)
        frame = panel.keys(origin, panel.rows(origin))
        frame['forecast'] = values.ravel()
        frame.to_parquet(output / f'{origin}.parquet', index=False)
        total += len(frame)
    return total


def compare_with_cached_b(panel, inputs, origin, values):
    """Not timed: agreement between recomputed and stored B, as in the residual_raf smoke check."""
    try:
        provider = backend()['backbone'].FrozenBackbone(panel, inputs)
        cached = provider(origin, panel.rows(origin))
    except Exception as error:  # Comparison is informative only.
        return dict(status='not_compared', reason=f'{type(error).__name__}: {error}')
    difference = np.abs(cached - values)
    return dict(status='compared', rows=int(cached.size), max_absolute_difference=float(difference.max()),
                mean_absolute_difference=float(difference.mean()))


# ----------------------------------------------------------------- FARE dev references
def fare_dev_references(data, schedule, output):
    """Reference builders for every development origin (shared by FARE and TimeFuse)."""
    fare = backend()['fare']
    source = fare.backend()
    output.mkdir(parents=True, exist_ok=True)
    rows = 0
    if DATASET in BASE_DATASETS:
        source['reference_function'](data, dict(fit=schedule['fit'], dev=schedule['dev'], test=[]), output)
        for origin in schedule['dev']:
            rows += len(pd.read_parquet(output / f'{origin}.parquet', columns=['origin']))
    else:
        for origin in schedule['dev']:
            frame = source['reference_function'](data, origin, batch=8)
            frame.to_parquet(output / f'{origin}.parquet', index=False)
            rows += len(frame)
    return rows


# ----------------------------------------------------------------- RAF, plan once
def raf_plan_origins(panel, pipeline, model_path, output, origins, device, batch_size):
    """build_retrieval_plan; the fixed fit library is encoded once for all origins."""
    source = backend()
    embedder = source['raf'].Chronos2Embedder(model_path=model_path, pipeline=pipeline,
                                              device=device, batch_size=batch_size, dtype='float32')
    return source['raf'].build_retrieval_plan(panel, embedder, output, origins=list(origins),
                                              rows_per_chunk=16, distance_device=device, resume=False)


def raf_predict_origin(panel, pipeline, plan, origin, batch_size):
    """Same per-batch augmentation and Chronos-2 call as run_raf in baselines/residual_raf/run.py."""
    source = backend()
    rows = panel.rows(origin)
    chunks = []
    for start in range(0, len(rows), batch_size):
        selected = rows[start:start + batch_size]
        augmented = source['raf'].augment_from_plan(panel, origin, selected, plan)
        prediction = source['backbone'].predict_context(pipeline, panel, origin, selected,
            augmented.context119, past_covariates=augmented.past_covariates119, batch_size=batch_size)
        chunks.append(np.maximum(augmented.denormalize(prediction), 0))
    return np.concatenate(chunks)


def limited_raf_panel(original, maximum, origins):
    """Smoke only: the same series subset for every requested query origin."""
    panel = copy.copy(original)
    selected = np.unique(np.concatenate([original.rows(o)[:maximum] for o in origins]))
    panel.ids = original.ids[selected]
    panel.first = original.first[selected]
    panel.values = np.asarray(original.values[selected])
    panel.covariates = np.asarray(original.covariates[selected])
    panel.id_rows = {str(value): index for index, value in enumerate(panel.ids)}
    panel.meta = copy.deepcopy(original.meta)
    panel.meta['ids'] = panel.ids.tolist()
    panel.meta['first'] = panel.first.astype(str).tolist()
    panel.meta['eligible_rows'] = {o: np.flatnonzero(np.isin(selected, original.rows(o))).tolist()
        for split in ('fit', 'dev', 'test') for o in panel.schedule[split]}
    panel._indices = {}
    return panel


# ----------------------------------------------------------------- LightGBM residual
def residual_fit(panel, provider, output, seed, threads, batch_rows, rounds):
    """train_residual: dev feature matrix, 3 leaf trials, full-dev refit."""
    return backend()['residual'].train_residual(panel, provider, output / 'model', seed, threads=threads,
        batch_rows=batch_rows, rounds=rounds, cache_dir=output / 'residual_matrix')


def residual_load(panel, provider, fit_dir, seed, threads, batch_rows, rounds):
    """Reopen the completed fit; train_residual returns from its manifest."""
    return backend()['residual'].train_residual(panel, provider, fit_dir / 'model', seed, threads=threads,
        batch_rows=batch_rows, rounds=rounds, cache_dir=fit_dir / 'residual_matrix')


def residual_predict(predictor, panel, provider, origin):
    return backend()['residual'].predict_residual(predictor, panel, provider, origin)


# ----------------------------------------------------------------- PIR
def pir_bank(panel, output, budget):
    return backend()['engine'].make_bank(panel, output, budget)


def pir_dev_retrieval(panel, origins, output, bank, config, device, budget):
    for origin in origins:
        backend()['engine'].cache_week(panel, origin, output, bank, config, device, budget)


def pir_train(panel, output, config, device, budget, seed, phase, origins, epochs, tune):
    return backend()['engine'].train(panel, output, config, device, budget, seed, phase, origins, epochs, tune)


def pir_load_bank(fit_dir):
    folder = Path(fit_dir) / 'bank'
    return np.load(folder / 'keys.npy', mmap_mode='r'), np.load(folder / 'values.npy', mmap_mode='r')


def pir_test_retrieval(panel, origin, output, bank, config, device, budget):
    backend()['engine'].cache_week(panel, origin, output, bank, config, device, budget)


def pir_predict(panel, origin, output, fit_dir, config, device, seed):
    """engine.predict restricted to one origin: identical model, tensors and scaling."""
    import torch
    source = backend()
    engine = source['engine']
    model = source['author'].RetailPIR(config).to(device)
    model.load_state_dict(engine.load_torch(Path(fit_dir) / f'seed_{seed}' / 'refit' / 'model.pt', device))
    model.eval()
    with torch.no_grad():
        arrays, scale, keys = engine.arrays_for(panel, origin, output, False)
        chunks = []
        for start in range(0, len(arrays[0]), config['batch_size']):
            index = np.arange(start, min(start + config['batch_size'], len(arrays[0])))
            args, _ = engine.tensors(arrays, index, device, origin)
            chunks.append(model.revise(*args)[0].cpu().numpy()[:, :, 0] * scale[index])
    return keys, np.concatenate(chunks)


# ----------------------------------------------------------------- PFRP
def pfrp_encoder(encoder, windows, config, device, budget, folder, seed):
    backend()['engine'].train(encoder, windows, config, device, budget, folder, 'encoder', seed,
                              config['encoder_epochs'], encoder=True)


def pfrp_memory(encoder, windows, config, device, budget, folder):
    backend()['engine'].memory(encoder, windows, config, device, budget, folder)


def pfrp_train(model, windows, panel, config, device, budget, folder, seed):
    backend()['engine'].train(model, windows, config, device, budget, folder, 'pfrp', seed,
                              config['fusion_epochs'], tune=(panel, panel.schedule['tune']))


def pfrp_load(fit_folder, config, device):
    engine = backend()['engine']
    model = engine.construct(fit_folder, config, device)
    model.load_state_dict(engine.load_torch(Path(fit_folder) / 'pfrp.best.pt', device)['model'])
    return model


def pfrp_predict(model, panel, origin, config, device):
    """forecast(); the decoder-isolation audit check is not part of the method."""
    x, _, keys = panel.window(origin, labels=False)
    return keys, backend()['engine'].forecast(model, x, device, config['batch_size'], decoder_check=False)


# ----------------------------------------------------------------- SARAF
def saraf_bank(panel, config, output, budget):
    return backend()['retrieval'].Bank(panel, config, output, budget)


def saraf_training_windows(bank, origins, seed, folder, device, budget):
    return backend()['engine'].TrainingWindows(bank, origins, seed, folder, device, budget)


def saraf_train(model, windows, bank, seed, folder, device, budget, config):
    backend()['engine'].train(model, windows, bank, seed, folder, device, budget, config, 'saraf')


def saraf_load_bank(panel, config, fit_dir):
    """Bank.__init__ without rebuilding the stored fit-period bank."""
    import torch
    retrieval = backend()['retrieval']
    bank = retrieval.Bank.__new__(retrieval.Bank)
    bank.panel = panel
    bank.c = config
    bank.folder = Path(fit_dir) / 'bank'
    if not (bank.folder / 'complete.json').is_file():
        raise FileNotFoundError('SARAF fit bank missing: ' + str(bank.folder))
    bank.origins = panel.schedule['train']
    bank.calibration_count = len(bank.origins) // 2
    bank.minimum_query = bank.origins[bank.calibration_count]
    _, tool = backend()['author'].author_api()
    bank.tool = tool(56, 7, 1, n_period=1, topm=config['topm'], freq='d',
                     time_aware_weight=config['time_aware_weight'], mmr_candidate_pool=config['candidate_pool'])
    bank.x = np.load(bank.folder / 'x.npy', mmap_mode='r')
    bank.y = np.load(bank.folder / 'y.npy', mmap_mode='r')
    bank.valid = np.load(bank.folder / 'valid.npy')
    params = np.load(bank.folder / 'scaler.npz')
    bank.mean = params['mean']
    bank.scale = params['scale']
    bank.meta = json.loads((bank.folder / 'bank.json').read_text(encoding='utf-8'))
    bank.tool.dataset_stationarity = bank.meta['stationarity']
    bank.tool.adaptive_sigma = bank.tool.compute_adaptive_sigma(bank.meta['stationarity'])
    bank.tool.adaptive_lambda = bank.tool.compute_adaptive_lambda(bank.meta['stationarity'])
    bank.marks = torch.tensor(np.stack([retrieval.date_marks(o) for o in bank.origins]))
    return bank


def saraf_load_model(fit_folder, config, device):
    source = backend()
    model = source['author'].construct(config, device, control=False)
    model.load_state_dict(source['engine'].load_torch(Path(fit_folder) / 'saraf.best.pt', device)['model'])
    return model


def saraf_predict(model, bank, origin, seed, folder, device, budget, config):
    """predict_origin: same-series retrieval for this origin and forecasting."""
    return backend()['engine'].predict_origin(model, bank, origin, seed, folder, device, budget, config)


# ----------------------------------------------------------------- dispatch
def method_panel(smoke):
    """Smoke keeps every series for PIR: its stored-B merge requires complete keys."""
    source = backend()
    folder = data_root(METHOD)
    panel = source['data'].Panel(folder)
    return limit_rows(panel, smoke) if smoke and METHOD != 'pir' else panel


def fit_dir_of(args):
    if args.fit_dir is None:
        raise ValueError('--fit-dir is required for this phase')
    fit = Path(args.fit_dir).resolve()
    if not (fit / 'measurement.json').is_file():
        raise FileNotFoundError('Fit output is incomplete: ' + str(fit))
    return fit


def run(args):
    if args.method != METHOD or args.dataset != DATASET:
        raise ValueError('EFFICIENCY_METHOD/EFFICIENCY_DATASET must match the entry arguments before profiling starts')
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('Use a fresh output directory: ' + str(output))
    output.mkdir(parents=True)
    smoke = args.smoke_queries
    threads = int(os.environ.get('EFFICIENCY_THREADS', args.threads))
    budget = NoBudget()
    metadata = dict(method=METHOD, dataset=DATASET, phase=args.phase, position=args.position,
        smoke_queries=smoke, formal_efficiency=smoke is None, threads=threads, device=args.device,
        source_functions=source_functions(), portable_native_adaptation=True,
        note='Efficiency measurements only; these predictions are not used for accuracy.')
    phase, method = args.phase, METHOD

    if method == 'fare' and phase == 'dev_reference':
        fare = backend()['fare']
        panel = fare.backend()['panel_class'](args.inputs / DATASET)
        schedule = panel.schedule
        data = fare.load_reference_data()
        if hasattr(data, 'schedule') and any(data.schedule[s] != schedule[s] for s in ('fit', 'dev', 'test')):
            raise ValueError('Raw-loader schedule differs from the candidate cache')
        if smoke:
            original = data.eligible
            def limited(value, *a, **k):
                rows = original(value, *a, **k)
                return rows if k.get('reference', False) else rows[:smoke]
            data.eligible = limited
        set_queries(sum(len(data.eligible(o)) for o in schedule['dev']))
        metadata['daily_prediction_rows'] = fare_dev_references(data, schedule, output / 'references')
        metadata['origins'] = schedule['dev']

    elif method == 'raf' and phase in ('raf_plan', 'raf_test'):
        source = backend()
        panel = source['data'].Panel(Path(args.inputs) / DATASET)
        origins = [position_origin(panel.schedule, p) for p in ('first', 'middle', 'last')]
        if smoke:
            panel = limited_raf_panel(panel, smoke, origins)
        pipeline = source['backbone'].load_pipeline(args.model_path, args.device)
        if phase == 'raf_plan':
            set_queries(sum(len(panel.rows(o)) for o in origins))
            raf_plan_origins(panel, pipeline, args.model_path, output / 'retrieval_plan', origins,
                             args.device, args.batch_size)
            metadata.update(origins=origins, plan_dir=str(output / 'retrieval_plan'))
        else:
            fit = fit_dir_of(args)
            plan = source['raf'].RetrievalPlan(fit / 'retrieval_plan')
            origin = position_origin(panel.schedule, args.position)
            set_queries(len(panel.rows(origin)))
            values = raf_predict_origin(panel, pipeline, plan, origin, args.batch_size)
            metadata.update(origin=origin, plan_dir=str(fit / 'retrieval_plan'),
                            daily_prediction_rows=save_forecast(panel.keys(origin, panel.rows(origin)), values, output))

    elif method == 'backbone':
        source = backend()
        panel = load_addon_panel(args.inputs, smoke)
        inputs = Path(args.inputs)
        pipeline = source['backbone'].load_pipeline(args.model_path, args.device)
        chunk = int(os.environ.get('EFFICIENCY_B_CHUNK', 4096))
        if phase == 'backbone':
            origin = position_origin(panel.schedule, args.position)
            set_queries(len(panel.rows(origin)))
            values = predict_backbone(panel, pipeline, origin, chunk, args.batch_size)
            metadata.update(origin=origin, b_check=compare_with_cached_b(panel, inputs, origin, values),
                            daily_prediction_rows=save_forecast(panel.keys(origin, panel.rows(origin)), values, output))
        elif phase == 'backbone_dev':
            origins = panel.schedule['dev']
            set_queries(sum(len(panel.rows(o)) for o in origins))
            (output / 'dev_backbone').mkdir()
            metadata.update(origins=origins, daily_prediction_rows=predict_backbone_dev(
                panel, pipeline, origins, chunk, args.batch_size, output / 'dev_backbone'))
        else:
            raise ValueError('backbone supports --phase backbone or backbone_dev')
        metadata['chunk_rows'] = chunk

    elif method == 'residual':
        source = backend()
        panel = load_addon_panel(args.inputs, smoke)
        inputs = Path(args.inputs)
        provider = source['backbone'].FrozenBackbone(panel, inputs)
        config = json.loads((source['root'] / 'config.json').read_text(encoding='utf-8'))
        seed = config['residual_seeds'][0]
        rounds = 5 if smoke else config['residual_rounds']
        batch_rows = int(os.environ.get('EFFICIENCY_FEATURE_BATCH_ROWS', 2048))
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=threads):
            if phase == 'fit':
                set_queries(sum(len(panel.rows(o)) for o in panel.schedule['dev']))
                residual_fit(panel, provider, output, seed, threads, batch_rows, rounds)
            elif phase == 'test':
                fit = fit_dir_of(args)
                origin = position_origin(panel.schedule, args.position)
                set_queries(len(panel.rows(origin)))
                predictor = residual_load(panel, provider, fit, seed, threads, batch_rows, rounds)
                values = residual_predict(predictor, panel, provider, origin)
                metadata.update(origin=origin, fit_dir=str(fit),
                                daily_prediction_rows=save_forecast(panel.keys(origin, panel.rows(origin)), values, output))
            else:
                raise ValueError('residual supports --phase fit or test')
        metadata.update(seed=seed, rounds=rounds, feature_batch_rows=batch_rows)

    elif method in ('pir', 'pfrp', 'saraf'):
        source = backend()
        config = smoke_config(source['config'], smoke)
        seed = config['seeds'][0]
        panel = method_panel(smoke)
        metadata.update(seed=seed, config=config, data=str(data_root(method)))
        with torch_setup(config['cpu_threads'], deterministic=method == 'pir'):
            import torch
            if method == 'pir':
                dev = panel.schedule['dev']
                cut = len(dev) - config['tune_weeks']
                if phase == 'fit':
                    set_queries(sum(len(panel.rows(o)) for o in dev))
                    bank = pir_bank(panel, output, budget)
                    pir_dev_retrieval(panel, dev, output, bank, config, args.device, budget)
                    selection = pir_train(panel, output, config, args.device, budget, seed, 'select',
                                          dev[:cut], config['epochs'], dev[cut:])
                    pir_train(panel, output, config, args.device, budget, seed, 'refit', dev,
                              selection['best_epoch'], [])
                    metadata['best_epoch'] = selection['best_epoch']
                else:
                    fit = fit_dir_of(args)
                    origin = position_origin(panel.schedule, args.position)
                    set_queries(len(panel.rows(origin)))
                    bank = pir_load_bank(fit)
                    pir_test_retrieval(panel, origin, output, bank, config, args.device, budget)
                    keys, values = pir_predict(panel, origin, output, fit, config, args.device, seed)
                    metadata.update(origin=origin, fit_dir=str(fit), daily_prediction_rows=save_forecast(keys, values, output))
            elif method == 'pfrp':
                folder = output / f'seed_{seed}'
                if phase == 'fit':
                    folder.mkdir(parents=True)
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    train = panel.schedule['train']
                    cut = len(train) // 2
                    memory_windows = source['data'].Windows(panel, train[:cut])
                    fusion_windows = source['data'].Windows(panel, train[cut:])
                    set_queries(len(memory_windows) + len(fusion_windows))
                    _, pcl = source['author'].author_api()
                    encoder = pcl['PastSeriesEncoder'](56).to(args.device)
                    pfrp_encoder(encoder, memory_windows, config, args.device, budget, folder, seed)
                    pfrp_memory(encoder, memory_windows, config, args.device, budget, folder)
                    torch.manual_seed(seed)
                    model = source['engine'].construct(folder, config, args.device)
                    pfrp_train(model, fusion_windows, panel, config, args.device, budget, folder, seed)
                else:
                    fit = fit_dir_of(args)
                    origin = position_origin(panel.schedule, args.position)
                    set_queries(len(panel.rows(origin)))
                    model = pfrp_load(fit / f'seed_{seed}', config, args.device)
                    keys, values = pfrp_predict(model, panel, origin, config, args.device)
                    metadata.update(origin=origin, fit_dir=str(fit), daily_prediction_rows=save_forecast(keys, values, output))
            else:
                folder = output / f'seed_{seed}'
                folder.mkdir(parents=True)
                if phase == 'fit':
                    train = panel.schedule['train']
                    origins = train[len(train) // 2:]
                    set_queries(sum(len(panel.rows(o)) for o in origins))
                    bank = saraf_bank(panel, config, output, budget)
                    windows = saraf_training_windows(bank, origins, seed, folder, args.device, budget)
                    torch.manual_seed(seed)
                    model = source['author'].construct(config, args.device, control=False)
                    saraf_train(model, windows, bank, seed, folder, args.device, budget, config)
                else:
                    fit = fit_dir_of(args)
                    origin = position_origin(panel.schedule, args.position)
                    set_queries(len(panel.rows(origin)))
                    bank = saraf_load_bank(panel, config, fit)
                    model = saraf_load_model(fit / f'seed_{seed}', config, args.device)
                    values, keys = saraf_predict(model, bank, origin, seed, folder, args.device, budget, config)
                    metadata.update(origin=origin, fit_dir=str(fit), daily_prediction_rows=save_forecast(keys, values, output))
    else:
        raise ValueError(f'Unsupported method/phase combination: {method}/{phase}')

    metadata['query_count'] = int(os.environ.get('EFFICIENCY_QUERY_COUNT', 0))
    (output / 'measurement.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2,
                                                       default=str, allow_nan=False), encoding='utf-8')
    print(json.dumps({key: metadata.get(key) for key in ('method', 'dataset', 'phase', 'origin', 'query_count')},
                     ensure_ascii=False))


# Validate real imports and signatures during profiler --check, without reading data.
# RAF and FARE also register their source roots here, because their cost specs wrap these modules.
if os.environ.get('EFFICIENCY_METHOD', '').lower() in EXTRA_METHODS + ('raf', 'fare'):
    source_functions()
