"""Rebuild unchanged Top-8 shape and alternating factor references."""
import numpy as np
import pandas as pd
from data_provider.retail import CONFIG, parquet_write, check_prediction, json_write


def build(data, schedule, folder):
    from data_provider.retail_factors import select
    origins = schedule["dev"] + schedule["test"]
    banks = schedule["fit"] + origins
    # A completed week can become a reference; current/future target slices never do.
    for origin in origins:
        path = folder / f"{origin}.parquet"
        if path.exists():
            check_prediction(pd.read_parquet(path))
            continue
        valid = [o for o in banks if pd.Timestamp(o) + pd.Timedelta(days=6) < pd.Timestamp(origin)]
        eligible = set(data.eligible(origin))
        pieces = []
        fallback_series = 0
        for members in data.group_rows:
            # Stable order: origin ascending, then lexical series ID.
            members = members[np.argsort(data.ids[members], kind="stable")]
            query_rows = np.array([r for r in members if r in eligible], dtype=int)
            if not len(query_rows):
                continue
            hs, fs, ps, es = [], [], [], []
            for past in valid:
                active = members[data.first[members] <= np.datetime64(past)-np.timedelta64(56, "D")]
                if not len(active):
                    continue
                cols = data.indices(past)
                h = data.values[np.ix_(active, cols[:56])]
                y = data.values[np.ix_(active, cols[56:])]
                cp = data.factor1[np.ix_(active, cols[56:])]
                ce = data.factor2[np.ix_(active, cols[56:])]
                keep = np.ones(len(active), dtype=bool)
                if data.dataset == "favorita":
                    keep = ~((cp == 2) & (ce == 1)).any(1)
                hs.append(h[keep]); fs.append(y[keep])
                ps.append(cp[keep]); es.append(ce[keep])
            if not hs or sum(len(v) for v in hs) < 8:
                # Newly listed items may have 56 history days but no completed legal
                # same-item reference week. Keep their evaluation rows explicitly.
                _, q, _, cov, _, keys = data.window(origin, query_rows, labels=False)
                keys["shape_retrieval"] = q[:, -7:].ravel()
                keys["factor_retrieval"] = q[:, -7:].ravel()
                keys["history_mean"] = np.repeat(q.mean(1), 7)
                keys["history_std"] = np.repeat(q.std(1), 7)
                keys["history_zero_fraction"] = np.repeat((q == 0).mean(1), 7)
                if data.dataset != "m5":
                    raise ValueError("Unexpected insufficient Favorita reference bank")
                keys["bank_size"] = sum(len(v) for v in hs)
                keys["snap"] = cov[:,56:,7].ravel()
                keys["event"] = cov[:,56:,8].ravel()
                pieces.append(keys)
                fallback_series += len(query_rows)
                continue
            h, f, p, e = map(np.concatenate, (hs, fs, ps, es))
            all_query_rows = query_rows
            for offset in range(0, len(all_query_rows), CONFIG["reference_query_batch"]):
                query_rows = all_query_rows[offset:offset+CONFIG["reference_query_batch"]]
                _, q, _, cov, _, keys = data.window(origin, query_rows, labels=False)
                # Float64 preserves the original implementation's ranking and tie behavior.
                h, q, f = h.astype(float), q.astype(float), f.astype(float)
                hscl, qscl = np.maximum(h.mean(1), 1), np.maximum(q.mean(1), 1)
                shape = ((q[:, None, :]/qscl[:, None, None]-h[None, :, :]/hscl[None, :, None])**2).mean(2)
                pdist = (cov[:, None, 56:, 7] != p[None, :, :]).mean(2)
                edist = (cov[:, None, 56:, 8] != e[None, :, :]).mean(2)
                chosen, distances = select(shape, pdist, edist)
                norm = f/hscl[:, None]
                keys["shape_retrieval"] = (norm[chosen["shape"]].mean(1)*qscl[:, None]).ravel()
                keys["factor_retrieval"] = (norm[chosen["factor"]].mean(1)*qscl[:, None]).ravel()
                keys["history_mean"] = np.repeat(q.mean(1), 7)
                keys["history_std"] = np.repeat(q.std(1), 7)
                keys["history_zero_fraction"] = np.repeat((q == 0).mean(1), 7)
                if data.dataset == "m5":
                    keys["bank_size"] = len(h)
                    keys["snap"] = cov[:, 56:, 7].ravel()
                    keys["event"] = cov[:, 56:, 8].ravel()
                else:
                    cf = chosen["factor"]
                    keys["promo_distance"] = np.repeat(np.take_along_axis(distances["promo_branch"], cf[:, ::2], 1).mean(1), 7)
                    keys["holiday_distance"] = np.repeat(np.take_along_axis(distances["holiday_branch"], cf[:, 1::2], 1).mean(1), 7)
                    keys["promo_bank_size"] = len(h)
                    keys["holiday_bank_size"] = len(h)
                    keys["holiday_day"] = cov[:, 56:, 8].ravel()
                    keys["onpromotion"] = data.promo[np.ix_(query_rows, data.indices(origin)[56:])].ravel()
                pieces.append(keys)
        frame = pd.concat(pieces, ignore_index=True)
        check_prediction(frame)
        json_write(folder/f"{origin}_audit.json", dict(seasonal_fallback_series=fallback_series, total_series=len(frame)//7))
        parquet_write(path, frame)
        print("references", origin, len(frame), flush=True)
