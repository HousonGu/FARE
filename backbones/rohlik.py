"""Backbone stages: pinned Chronos-2 and validation-tuned LightGBM."""

import gc

import importlib.metadata

import itertools

import numpy as np

import pandas as pd

from data_provider.supervised_features import CONFIG, ROOT, KEYS, metric, parquet_write, json_write, digest, check_prediction

def tree_data(data, origins):
    xs, ys, ws = [], [], []
    for origin in origins:
        x, y, w, _ = data.tree(origin, labels=True)
        xs.append(x); ys.append(y); ws.append(w)
    return pd.concat(xs, ignore_index=True), np.concatenate(ys), np.concatenate(ws)

def lightgbm(data, schedule, folder):
    import lightgbm as lgb
    if lgb.__version__ != "4.6.0":
        raise RuntimeError("This protocol requires lightgbm==4.6.0")
    rng = np.random.default_rng(CONFIG["seed"])
    space = list(itertools.product([31, 63, 127], [.03, .06, .1], [50, 150, 400], [0., 1., 5.]))
    order = rng.choice(len(space), size=CONFIG["lightgbm_trials"], replace=False)
    params = [dict(num_leaves=space[i][0], learning_rate=space[i][1], min_child_samples=space[i][2], reg_lambda=space[i][3]) for i in order]
    base = dict(objective="regression_l1", verbosity=-1, n_jobs=CONFIG["cpu_threads"],
                subsample=.9, subsample_freq=1, colsample_bytree=.9,
                random_state=CONFIG["seed"], deterministic=True, force_col_wise=True)
    folder.mkdir(parents=True, exist_ok=True)
    selection = folder/"selection.json"
    if not selection.exists():
        x, y, w = tree_data(data, schedule["train"])
        vx, vy, vw = tree_data(data, schedule["tune"])
        trials = []
        for i, p in enumerate(params):
            trial_path = folder/f"trial_{i:02d}.json"
            if trial_path.exists():
                import json
                trials.append(json.loads(trial_path.read_text()))
                continue
            model = lgb.LGBMRegressor(**base, **p, n_estimators=CONFIG["lightgbm_max_rounds"])
            model.fit(x, y, sample_weight=w, categorical_feature=data.static_names+["horizon"],
                eval_set=[(vx, vy)], eval_sample_weight=[vw], eval_metric="l1",
                callbacks=[lgb.early_stopping(CONFIG["lightgbm_patience"], verbose=False)])
            pred = np.maximum(model.predict(vx), 0)*vw
            trial = dict(index=i, params=p, iterations=int(model.best_iteration_), validation=metric(vy*vw, pred))
            json_write(trial_path, trial)
            trials.append(trial)
            print("lightgbm trial", i, trial["validation"]["wape"], flush=True)
        best = min(trials, key=lambda r: (r["validation"]["wape"], r["index"]))
        json_write(selection, dict(best=best, trials=trials, train=schedule["train"], tune=schedule["tune"]))
        del x, y, w, vx, vy, vw
        gc.collect()
    import json
    selected = json.loads(selection.read_text())["best"]
    model_path = folder/"model.txt"
    if model_path.exists():
        booster = lgb.Booster(model_file=str(model_path))
    else:
        x, y, w = tree_data(data, schedule["fit"])
        model = lgb.LGBMRegressor(**base, **selected["params"], n_estimators=selected["iterations"])
        model.fit(x, y, sample_weight=w, categorical_feature=data.static_names+["horizon"])
        booster = model.booster_
        booster.save_model(str(model_path)+".tmp")
        import os
        os.replace(str(model_path)+".tmp", model_path)
        from data_provider.supervised_features import seal
        seal(model_path)
        del x, y, w
        gc.collect()
    for origin in schedule["dev"]+schedule["test"]:
        path = folder/f"{origin}.parquet"
        if path.exists():
            check_prediction(pd.read_parquet(path))
            continue
        x, _, scale, keys = data.tree(origin, labels=False)
        keys["forecast"] = np.maximum(booster.predict(x, num_threads=CONFIG["cpu_threads"]), 0)*scale
        check_prediction(keys)
        parquet_write(path, keys)
        print("lightgbm predict", origin, len(keys), flush=True)
