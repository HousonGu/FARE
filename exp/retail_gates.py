"""Equal-input, equal-parameter two-layer versus flat softmax comparison."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax
from threadpoolctl import threadpool_limits
from data_provider.retail import CONFIG, KEYS, checked_merge, parquet_write, json_write, check_prediction
from fare.fusion import base_features, historical_features, fit_gate, predict_gate


def flat_objective(theta, x, y, parts, penalty=0.002):
    k = x.shape[1]
    logits = np.column_stack([np.zeros(len(x)), x @ theta[:k], x @ theta[k:]])
    w = softmax(logits, axis=1)
    pred = (parts*w).sum(1)
    residual = pred-y
    smooth = np.sqrt(residual**2+1)
    g = residual/smooth/y.sum()
    gradient = np.concatenate([x.T @ (g*w[:, j]*(parts[:, j]-pred)) for j in (1, 2)])
    penalized = np.ones(2*k, bool)
    penalized[[0, k]] = False
    gradient[penalized] += 2*penalty*theta[penalized]
    value = smooth.sum()/y.sum()+penalty*np.square(theta[penalized]).sum()
    return float(value), gradient


def flat_fit(x, y, parts):
    theta = np.zeros(2*x.shape[1])
    theta[[0, x.shape[1]]] = -np.log(2)  # same initial .5/.25/.25 as two-layer
    with threadpool_limits(limits=4):
        result = minimize(flat_objective, theta, args=(x, y, parts, CONFIG["gate_penalty"]),
            jac=True, method="L-BFGS-B", options={"maxiter": CONFIG["gate_maxiter"],
            "ftol": CONFIG["gate_ftol"], "gtol": CONFIG["gate_gtol"]})
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"Softmax did not converge: {result.message}")
    return result.x, dict(iterations=int(result.nit), objective=float(result.fun),
                          parameters=result.x.tolist(), initial_weights=[.5, .25, .25])


def flat_predict(theta, x, parts):
    k = x.shape[1]
    w = softmax(np.column_stack([np.zeros(len(x)), x@theta[:k], x@theta[k:]]), axis=1)
    return (w*parts).sum(1), w


def design(frame, train, dataset):
    # Held-out labels must be absent even when the source file already contains them.
    safe = frame.copy()
    safe.loc[~train, "actual"] = np.nan
    parts = np.maximum(safe[["backbone_forecast", "shape_retrieval", "factor_retrieval"]].to_numpy(float), 0)
    history, _, info = historical_features(safe, "series_id", parts, train)
    base, names, _ = base_features(safe, "Favorita" if dataset == "favorita" else "M5")
    names += ["hist_backbone_error", "hist_shape_error", "hist_factor_error", "hist_log_support"]
    raw = np.column_stack([base, history])
    continuous = [i for i, n in enumerate(names) if n not in {"holiday", "snap", "event"}]
    mean = raw[train][:, continuous].mean(0)
    std = np.maximum(raw[train][:, continuous].std(0), 1e-8)
    raw[:, continuous] = (raw[:, continuous]-mean)/std
    return np.column_stack([np.ones(len(raw)), raw]), parts, dict(names=names, mean=mean.tolist(), std=std.tolist(), history=info)


def run(data, schedule, folder, backbone):
    dev, test = [], []
    for origin in schedule["dev"] + schedule["test"]:
        ref = pd.read_parquet(folder / "references" / f"{origin}.parquet")
        path = (folder / "tft" / f"{origin}_seed_{backbone.removeprefix('tft_seed_')}.parquet"
                if backbone.startswith("tft_seed_") else folder / backbone / f"{origin}.parquet")
        b = pd.read_parquet(path).rename(columns={"forecast": "backbone_forecast"})
        frame = checked_merge(ref, b)
        if origin in schedule["dev"]:
            dev.append(frame)
        else:
            test.append(frame)
    dev = checked_merge(pd.concat(dev, ignore_index=True), data.truth(schedule["dev"]))
    test = pd.concat(test, ignore_index=True).assign(actual=np.nan)
    dev["actual"] = dev.actual.astype(float)
    test["actual"] = test.actual.astype(float)
    frame = pd.concat([dev, test], ignore_index=True).sort_values(KEYS).reset_index(drop=True)
    train = frame.origin.isin(schedule["dev"]).to_numpy()
    x, parts, meta = design(frame, train, data.dataset)
    y = frame.actual.to_numpy(float)
    output = frame.loc[~train, KEYS].copy()
    output["backbone"] = parts[~train, 0]
    with threadpool_limits(limits=CONFIG["cpu_threads"]):
        theta, fit = fit_gate(x[train], y[train], *parts[train].T)
        prediction, beta, alpha = predict_gate(x[~train], theta, *parts[~train].T)
    output["soft_history"] = prediction
    fits = {"soft_history": fit}
    if backbone == "chronos":
        from exp.static_controls import select_global_weights, select_by_history
        theta, fit = flat_fit(x[train], y[train], parts[train])
        output["flat_softmax"], _ = flat_predict(theta, x[~train], parts[~train])
        fits["flat_softmax"] = fit
        _, _, factor_names = base_features(frame, "Favorita" if data.dataset == "favorita" else "M5")
        names = meta["names"]
        variants = {
            "soft_no_history": (set(names[-4:]), False),
            "soft_history_no_factor": (factor_names | {"hist_factor_error"}, True),
            "soft_history_no_disagreement": ({"backbone_shape_disagreement", "factor_shape_disagreement"}, False),
        }
        for name, (drop, single) in variants.items():
            keep = [0]+[i+1 for i,n in enumerate(names) if n not in drop]
            xx = np.ascontiguousarray(x[:, keep])
            with threadpool_limits(limits=CONFIG["cpu_threads"]):
                theta, fit = fit_gate(xx[train], y[train], *parts[train].T, single=single)
                output[name] = predict_gate(xx[~train], theta, parts[~train,0], parts[~train,1],
                    None if single else parts[~train,2])[0]
            fits[name] = fit
        weights, dev_wape = select_global_weights(y[train], parts[train])
        static = np.maximum(parts @ np.asarray(weights)/20, 0)
        output["shape_half"] = parts[~train][:, [0,1]].mean(1)
        output["factor_half"] = parts[~train][:, [0,2]].mean(1)
        output["global_static"] = static[~train]
        candidates = np.column_stack([parts[:,0], static, parts[:,[0,1]].mean(1), parts[:,[0,2]].mean(1)])
        scale_cohort = np.where(frame.history_mean.to_numpy()<1, 0, np.where(frame.history_mean.to_numpy()<5, 1, 2))
        cohort = 2*scale_cohort+(frame.history_zero_fraction.to_numpy()>=.5).astype(int)
        codes, _ = pd.factorize(frame.series_id, sort=True)
        _, hard, _, _, info = select_by_history(y, candidates, cohort, codes, train, (0,1,2,3))
        output["history_hard"] = np.maximum(hard[~train], 0)
        fits["global_static"] = dict(weights=np.asarray(weights).tolist(), development_wape=dev_wape)
        fits["history_hard"] = info
    check_prediction(output)
    dest = folder / ("gates_" + backbone)
    parquet_write(dest / "predictions.parquet", output)
    json_write(dest / "fit.json", dict(design=meta, fits=fits, full_parameter_count=2*x.shape[1],
        alpha_quantiles=np.quantile(alpha, [0, .1, .5, .9, 1]).tolist(),
        beta_quantiles=np.quantile(beta, [0, .1, .5, .9, 1]).tolist()))
