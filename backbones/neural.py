"""Official model construction, fit-only selection, exact epoch resume and inference."""
from pathlib import Path
from types import SimpleNamespace
import itertools
import json
import os
import random
import time
import numpy as np
import torch
from data_provider.panel import Windows
from exp.io import save_parquet, write_json, seal_file, verify_file, metric, merge


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure(config, device):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(config['cpu_threads'])
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no silent CPU fallback')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def candidates(kind, config):
    settings = config[kind]
    if kind == 'patchtst':
        return [dict(d_model=d, patch_len=p, learning_rate=lr, weight_decay=0.0)
                for d, p, lr in itertools.product(settings['d_models'], settings['patch_lengths'], settings['learning_rates'])]
    return [dict(learning_rate=lr, weight_decay=wd)
            for lr, wd in itertools.product(settings['learning_rates'], settings['weight_decays'])]


def make_model(kind, params, config):
    if kind == 'patchtst':
        from vendor.patchtst.model import Model
        s = config[kind]
        model = Model(SimpleNamespace(enc_in=1, seq_len=56, pred_len=7,
            e_layers=s['layers'], n_heads=s['heads'], d_model=params['d_model'],
            d_ff=4*params['d_model'], dropout=s['dropout'], fc_dropout=0.0,
            head_dropout=s['dropout'], individual=False, patch_len=params['patch_len'],
            stride=s['stride'], padding_patch='end', revin=True, affine=True,
            subtract_last=False, decomposition=False, kernel_size=25))
    elif kind == 'sparsetsf':
        from vendor.sparsetsf.model import Model
        s = config[kind]
        if 56 % s['period_len'] or 7 % s['period_len']:
            raise ValueError('SparseTSF period must divide both context and horizon')
        model = Model(SimpleNamespace(enc_in=1, seq_len=56, pred_len=7,
            period_len=s['period_len'], d_model=32, model_type=s['model_type']))
    else:
        raise ValueError(kind)
    return model


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)
    seal_file(path)


def cpu_weights(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def target_mean(windows, batch_size):
    total, count = 0.0, 0
    for start in range(0, len(windows), batch_size):
        y = windows.targets(np.arange(start, min(start+batch_size, len(windows))))
        total += float(y.sum())
        count += y.size
    return max(total/count, 1e-8)


def validate(model, windows, device, config):
    model.eval()
    error, actual, rows = 0.0, 0.0, 0
    with torch.inference_mode():
        for start in range(0, len(windows), config['prediction_batch_size']):
            indices = np.arange(start, min(start+config['prediction_batch_size'], len(windows)))
            x, y, scale = windows.batch(indices)
            pred = model(torch.from_numpy(x[..., None]).to(device)).squeeze(-1).cpu().numpy()
            prediction = np.maximum(pred.astype(np.float64)*scale[:, None], 0)
            truth = windows.targets(indices)
            if not np.isfinite(prediction).all():
                raise RuntimeError('Nonfinite validation prediction')
            error += float(np.abs(truth-prediction).sum())
            actual += float(truth.sum())
            rows += truth.size
    if actual <= 0:
        raise ValueError('Zero tune target sum')
    return dict(wape=error/actual, mae=error/rows, rows=rows)


def train(model, windows, params, device, config, epochs, seed, checkpoint, tune=None):
    """Resume only committed epochs, including optimizer and dropout RNG states."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    start_epoch, stale, best, best_epoch, stopped, records = 0, 0, float('inf'), 0, False, []
    denominator = None
    if checkpoint.exists():
        verify_file(checkpoint)
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if state['params'] != params or state['seed'] != seed:
            raise ValueError('Checkpoint configuration changed')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        for values in optimizer.state.values():
            for key, value in values.items():
                if torch.is_tensor(value) and key != 'step':
                    values[key] = value.to(device)
        start_epoch, stale, best, best_epoch = state['epoch'], state['stale'], state['best'], state['best_epoch']
        stopped, records, denominator = state['stopped'], state['records'], state['denominator']
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'])
        if device.startswith('cuda'):
            torch.cuda.set_rng_state_all(state['cuda_rng'])
    if denominator is None:
        denominator = target_mean(windows, config['prediction_batch_size'])
    for epoch in range(start_epoch, epochs):
        if stopped:
            break
        tick = time.perf_counter()
        order = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(len(windows))
        model.train()
        total, count = 0.0, 0
        for start in range(0, len(order), config['batch_size']):
            x, y, scale = windows.batch(order[start:start+config['batch_size']])
            x = torch.from_numpy(x[..., None]).to(device)
            y = torch.from_numpy(y).to(device)
            scale_tensor = torch.from_numpy(scale[:, None]).to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x).squeeze(-1)
            loss = (torch.abs(pred-y)*scale_tensor).mean()/denominator
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite backbone training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach())*len(x)
            count += len(x)
        score = validate(model, tune, device, config) if tune is not None else None
        if score is not None:
            if score['wape'] < best:
                best, best_epoch, stale = score['wape'], epoch+1, 0
            else:
                stale += 1
            stopped = stale >= config['patience']
        record = dict(epoch=epoch+1, train_loss=total/count, validation=score, seconds=time.perf_counter()-tick)
        records.append(record)
        atomic_torch(checkpoint, dict(model=cpu_weights(model), optimizer=optimizer.state_dict(), epoch=epoch+1,
            seed=seed, params=params, stale=stale, best=best, best_epoch=best_epoch, stopped=stopped,
            denominator=denominator, records=records, python_rng=random.getstate(), numpy_rng=np.random.get_state(),
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.startswith('cuda') else []))
        print(json.dumps(dict(event='epoch', checkpoint=str(checkpoint), **record)), flush=True)
    return dict(epochs_completed=len(records), best_epoch=best_epoch if tune is not None else epochs,
                best_wape=best if tune is not None else None, records=records, training_windows=len(windows))


def selection(panel, kind, folder, device, config):
    path = folder/'selection.json'
    if path.exists():
        verify_file(path)
        return json.loads(path.read_text(encoding='utf-8'))
    train_windows = Windows(panel, panel.schedule['train'])
    tune_windows = Windows(panel, panel.schedule['tune'])
    trials = []
    for index, params in enumerate(candidates(kind, config)):
        directory = folder/'search'/f'trial_{index:02d}'
        result_path = directory/'result.json'
        if result_path.exists():
            verify_file(result_path)
            result = json.loads(result_path.read_text(encoding='utf-8'))
        else:
            seed_all(config['seeds'][0])
            model = make_model(kind, params, config)
            result = train(model, train_windows, params, device, config, config['max_epochs'],
                           config['seeds'][0], directory/'resume.pt', tune=tune_windows)
            result.update(index=index, params=params)
            write_json(result_path, result)
            seal_file(result_path)
            del model
        trials.append(result)
    best = min(trials, key=lambda r: (r['best_wape'], r['best_epoch'], r['index']))
    result = dict(best=best, trials=trials, train=panel.schedule['train'], tune=panel.schedule['tune'],
                  selection_seed=config['seeds'][0], protocol=config['selection'])
    write_json(path, result)
    seal_file(path)
    return result


def predict_origin(model, panel, origin, device, config):
    rows = panel.rows(origin)
    output = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), config['prediction_batch_size']):
            batch_rows = rows[start:start+config['prediction_batch_size']]
            x = panel.context(origin, batch_rows)
            scale = np.maximum(x.mean(1, dtype=np.float64), 1).astype(np.float32)
            values = torch.from_numpy((x/scale[:, None])[..., None]).to(device)
            pred = model(values).squeeze(-1).cpu().numpy().astype(np.float64)*scale[:, None]
            if not np.isfinite(pred).all():
                raise RuntimeError('Nonfinite backbone prediction')
            keys = panel.keys(origin, batch_rows)
            keys['forecast'] = np.maximum(pred, 0).ravel()
            output.append(keys)
    import pandas as pd
    return pd.concat(output, ignore_index=True)


def run_backbone(panel, kind, folder, device, config):
    configure(config, device)
    folder.mkdir(parents=True, exist_ok=True)
    selected = selection(panel, kind, folder, device, config)['best']
    fit_windows = Windows(panel, panel.schedule['fit'])
    for seed in config['seeds']:
        directory = folder/str(seed)
        weights_path = directory/'model.pt'
        seed_all(seed)
        model = make_model(kind, selected['params'], config)
        if weights_path.exists():
            verify_file(weights_path)
            verify_file(directory/'training.json')
            model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))
            model.to(device)
        else:
            result = train(model, fit_windows, selected['params'], device, config, selected['best_epoch'],
                           seed, directory/'resume.pt')
            write_json(directory/'training.json', dict(**result, seed=seed, params=selected['params'],
                parameter_count=sum(p.numel() for p in model.parameters()), fit=panel.schedule['fit']))
            seal_file(directory/'training.json')
            atomic_torch(weights_path, cpu_weights(model))
        for origin in panel.schedule['dev'] + panel.schedule['test']:
            path = directory/'predictions'/f'{origin}.parquet'
            if path.exists():
                verify_file(path)
                continue
            prediction = predict_origin(model, panel, origin, device, config)
            merge(panel.keys(origin), prediction)
            save_parquet(path, prediction)
            print(kind, seed, origin, len(prediction), 'backbone rows saved', flush=True)
        del model
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
