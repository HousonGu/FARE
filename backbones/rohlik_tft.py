"""PyTorch Forecasting TFT with lazy fixed weekly windows and a plain Torch loop.

No replacement/simplified TFT architecture. Only the data adapter and optimizer
loop are local. Decoder sales are zero placeholders, excluded from decoder inputs.
"""
import gc
import itertools
import json
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from data_provider.supervised_features import CONFIG, parquet_write, json_write, metric, check_prediction


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


class Windows(Dataset):
    def __init__(self, data, origins, labels, sample=False):
        self.packs = []
        self.ends = []
        self.keys = []
        total = 0
        for origin in origins:
            rows = data.eligible(origin)
            if sample and CONFIG["tft_fit_series_per_origin"] is not None and len(rows) > CONFIG["tft_fit_series_per_origin"]:
                # Date-based seed is stable when a temporal split is refit.
                rng = np.random.default_rng(CONFIG["seed"]+pd.Timestamp(origin).toordinal())
                rows = np.sort(rng.choice(rows, CONFIG["tft_fit_series_per_origin"], replace=False))
            rows, h, scale, cov, y, keys = data.window(origin, rows, labels=labels)
            self.packs.append(((h/scale[:, None]).astype(np.float32), scale, cov, data.static[rows],
                               None if y is None else y.astype(np.float32)))
            self.keys.append(keys if not labels else None)
            total += len(rows)
            self.ends.append(total)

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        p = int(np.searchsorted(self.ends, index, side="right"))
        r = index-(self.ends[p-1] if p else 0)
        h, scale, cov, static, y = self.packs[p]
        target = np.concatenate([h[r], np.zeros(7, np.float32)])
        cont = np.column_stack([target, cov[r], np.full(63, np.log1p(scale[r]), np.float32)]).astype(np.float32)
        cats = np.broadcast_to(static[r], (63, static.shape[1])).copy()
        x = dict(encoder_cont=torch.from_numpy(cont[:56]), decoder_cont=torch.from_numpy(cont[56:]),
                 encoder_cat=torch.from_numpy(cats[:56]), decoder_cat=torch.from_numpy(cats[56:]),
                 encoder_lengths=torch.tensor(56), decoder_lengths=torch.tensor(7),
                 target_scale=torch.tensor([0., 1.]))
        return x, torch.tensor(scale[r]), torch.from_numpy(y[r].copy() if y is not None else np.zeros(7, np.float32))


def make_model(data, params):
    import importlib.metadata
    if importlib.metadata.version("pytorch-forecasting") != "1.4.0":
        raise RuntimeError("This adapter requires pytorch-forecasting==1.4.0")
    from pytorch_forecasting import TemporalFusionTransformer
    from pytorch_forecasting.metrics import MAE
    from pytorch_forecasting.data.encoders import TorchNormalizer
    reals = ["sales"]+[f"weekday_{i}" for i in range(7)]+["factor1", "factor2", "log_scale"]
    known = reals[1:-1]
    normalizer = TorchNormalizer(method="identity")
    normalizer.fit(np.array([0., 1.]))
    return TemporalFusionTransformer(
        hidden_size=params["hidden_size"], dropout=params["dropout"], learning_rate=params["learning_rate"],
        lstm_layers=1, attention_head_size=4, hidden_continuous_size=16,
        max_encoder_length=56, output_size=1, loss=MAE(), output_transformer=normalizer,
        x_reals=reals, x_categoricals=data.static_names, static_categoricals=data.static_names,
        static_reals=["log_scale"], time_varying_reals_encoder=["sales"]+known,
        time_varying_reals_decoder=known,
        embedding_sizes={k: (v, min(16, max(2, int(np.ceil(np.sqrt(v)))))) for k, v in data.cardinalities.items()},
        log_interval=-1, logging_metrics=torch.nn.ModuleList())


def loader(windows, shuffle=False):
    return DataLoader(windows, batch_size=CONFIG["tft_batch_size"], shuffle=shuffle,
                      num_workers=0, pin_memory=torch.cuda.is_available())


def forward(model, x, scale, device):
    x = {k: v.to(device, non_blocking=True) for k, v in x.items()}
    # output_transformer is identity: model output remains history-normalized.
    pred = model(x).prediction.squeeze(-1)
    return pred*scale.to(device)[:, None]


def evaluate(model, windows, device):
    model.eval()
    errors = denom = 0.
    with torch.inference_mode():
        for x, scale, y in loader(windows):
            p = forward(model, x, scale, device).clamp_min(0)
            y = y.to(device)
            errors += (p-y).abs().sum().item()
            denom += y.sum().item()
    if denom <= 0 or not np.isfinite(errors):
        raise ValueError("Invalid TFT validation")
    return errors/denom


def train(model, train_set, device, epochs, valid=None, checkpoint=None):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=model.hparams.learning_rate)
    best = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    start_epoch = 1
    if checkpoint is not None and checkpoint.exists():
        from data_provider.supervised_features import verify
        verify(checkpoint)
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        best,best_epoch,stale,history=state['best'],state['best_epoch'],state['stale'],state['history']
        random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng']); torch.set_rng_state(state['torch_rng'])
        if torch.cuda.is_available(): torch.cuda.set_rng_state_all(state['cuda_rng'])
        start_epoch=state['epoch']+1
    # Constant dataset denominator gives raw MAE-equivalent scaling across batches.
    target_mean = sum(p[4].sum(dtype=np.float64) for p in train_set.packs)/(len(train_set)*7)
    if target_mean <= 0:
        raise ValueError("Zero training targets")
    for epoch in range(start_epoch, epochs+1):
        if valid is not None and stale >= CONFIG['tft_patience']: break
        model.train()
        total = 0.
        for x, scale, y in loader(train_set, shuffle=True):
            optimizer.zero_grad(set_to_none=True)
            pred = forward(model, x, scale, device)
            loss = (pred-y.to(device)).abs().mean()/target_mean
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite TFT loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += float(loss.detach())
        val = evaluate(model, valid, device) if valid is not None else None
        history.append(dict(epoch=epoch, loss_sum=total, validation_wape=val))
        print("TFT epoch", epoch, "validation", val, flush=True)
        if valid is not None:
            if val < best:
                best, best_epoch, stale = val, epoch, 0
            else:
                stale += 1
        if checkpoint is not None:
            import os
            from data_provider.supervised_features import seal
            state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),best=best,best_epoch=best_epoch,
                stale=stale,history=history,epoch=epoch,python_rng=random.getstate(),numpy_rng=np.random.get_state(),
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
            torch.save(state,str(checkpoint)+'.tmp'); os.replace(str(checkpoint)+'.tmp',checkpoint); seal(checkpoint)
    return dict(history=history, best_epoch=best_epoch if valid is not None else epochs,
                best_validation_wape=best if valid is not None else None)


def run(data, schedule, folder, device):
    folder.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(CONFIG["cpu_threads"])
    selection = folder/"selection.json"
    if not selection.exists():
        train_set = Windows(data, schedule["train"], labels=True, sample=True)
        valid = Windows(data, schedule["tune"], labels=True)
        combinations = itertools.product(CONFIG["tft_hidden_sizes"], CONFIG["tft_dropouts"], CONFIG["tft_learning_rates"])
        trials = []
        for i, (hidden, dropout, lr) in enumerate(combinations):
            path = folder/f"trial_{i:02d}.json"
            if path.exists():
                trials.append(json.loads(path.read_text()))
                continue
            params = dict(hidden_size=hidden, dropout=dropout, learning_rate=lr)
            seed_all(CONFIG["seed"])
            model = make_model(data, params)
            result = train(model, train_set, device, CONFIG["tft_epochs"], valid, folder/f'trial_{i:02d}_resume.pt')
            trial = dict(index=i, params=params, **result)
            json_write(path, trial)
            trials.append(trial)
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        best = min(trials, key=lambda t: (t["best_validation_wape"], t["index"]))
        json_write(selection, dict(best=best, trials=trials, train=schedule["train"], tune=schedule["tune"],
            training_windows=len(train_set), validation_windows=len(valid)))
        del train_set, valid
        gc.collect()
    selected = json.loads(selection.read_text())["best"]
    fit_set = None
    for seed in CONFIG["tft_seeds"]:
        seed_all(seed)
        model = make_model(data, selected["params"]).to(device)
        checkpoint = folder/f"seed_{seed}.pt"
        if checkpoint.exists():
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        else:
            if fit_set is None:
                fit_set = Windows(data, schedule["fit"], labels=True, sample=True)
            log = train(model, fit_set, device, selected["best_epoch"], checkpoint=folder/f'seed_{seed}_resume.pt')
            torch.save(model.state_dict(), str(checkpoint)+".tmp")
            import os
            os.replace(str(checkpoint)+".tmp", checkpoint)
            from data_provider.supervised_features import seal
            seal(checkpoint)
            json_write(folder/f"seed_{seed}_fit.json", dict(seed=seed, windows=len(fit_set), **log))
        model.eval()
        for origin in schedule["dev"] + schedule["test"]:
            path = folder/f"{origin}_seed_{seed}.parquet"
            if path.exists():
                check_prediction(pd.read_parquet(path))
                continue
            windows = Windows(data, [origin], labels=False)
            chunks = []
            with torch.inference_mode():
                for x, scale, _ in loader(windows):
                    chunks.append(forward(model, x, scale, device).clamp_min(0).cpu().numpy())
            keys = windows.keys[0].copy()
            keys["forecast"] = np.concatenate(chunks).ravel()
            check_prediction(keys)
            parquet_write(path, keys)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
