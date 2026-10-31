"""Frozen RAF mechanics and the declared same-series weekly-fit retail bank.

The forecasting backbone and feature extractor share the original frozen
Chronos-2 weights.  This is an explicitly documented retail
adaptation, not an exact reproduction of the original cross-series benchmark.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np

CONTEXT = 56
HORIZON = 7
AUGMENTED_CONTEXT = 119
EPS = 1e-7
PLAN_VERSION = "raf-chronos2-same-series-weekly-fit-anchors64-v3"


def _finite_matrix(value, width, name):
    out = np.asarray(value, dtype=np.float32)
    if out.ndim != 2 or out.shape[1] != width or not np.isfinite(out).all():
        raise ValueError(f"{name} must be a finite N x {width} matrix")
    return out


def instance_normalize(context):
    """Author convention: population std + 1e-7, with query-only statistics."""
    values = _finite_matrix(context, CONTEXT, "query context")
    mean = values.mean(axis=1, dtype=np.float32)
    scale = values.std(axis=1, ddof=0, dtype=np.float32) + EPS
    return (values - mean[:, None]) / scale[:, None], mean, scale


@dataclass
class AugmentedBatch:
    context: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    past_covariates: np.ndarray | None = None
    matched_origins: np.ndarray | None = None
    source_rows: np.ndarray | None = None

    @property
    def context119(self):
        return self.context

    @property
    def past_covariates119(self):
        return self.past_covariates

    def denormalize(self, forecast):
        values = np.asarray(forecast, dtype=np.float32)
        if values.ndim != 2 or values.shape != (len(self.mean), HORIZON):
            raise ValueError("Expected an N x 7 point forecast")
        result = values * self.scale[:, None] + self.mean[:, None]
        if not np.isfinite(result).all():
            raise ValueError("Non-finite RAF prediction; no silent imputation")
        return result


def augment_context(query_context, retrieved_segment):
    """Normalize retrieved 56+7 and query 56 separately, align, concatenate.

    The seven retrieved continuation values are historical, observed values.
    The caller is responsible for the origin cutoff; augment_from_plan enforces
    that cutoff and the frozen fit-only library contract.
    """
    query = _finite_matrix(query_context, CONTEXT, "query context")
    segment = _finite_matrix(retrieved_segment, CONTEXT + HORIZON, "retrieved segment")
    if len(query) != len(segment):
        raise ValueError("Query/retrieved row counts differ")
    normalized, mean, scale = instance_normalize(query)
    segment_mean = segment.mean(axis=1, dtype=np.float32)
    segment_scale = segment.std(axis=1, ddof=0, dtype=np.float32) + EPS
    segment = (segment - segment_mean[:, None]) / segment_scale[:, None]
    segment += (normalized[:, 0] - segment[:, -1])[:, None]
    augmented = np.ascontiguousarray(np.concatenate([segment, normalized], axis=1))
    if augmented.shape != (len(query), AUGMENTED_CONTEXT) or not np.isfinite(augmented).all():
        raise ValueError("Invalid augmented RAF context")
    return AugmentedBatch(augmented, mean, scale)


def token_l2_distances(query_embeddings, bank_embeddings, device="cpu"):
    """Return sum_t ||E_query[t] - E_bank[t]||_2, exactly the author metric.

    No pooling, flattened-vector distance, cosine metric, or ANN shortlist is
    substituted.  Batched cdist over token positions avoids the enormous
    Q x K x token x hidden broadcasting allocation in the author code.
    """
    import torch

    query = np.asarray(query_embeddings, dtype=np.float32)
    bank = np.asarray(bank_embeddings, dtype=np.float32)
    if query.ndim != 3 or bank.ndim != 3 or query.shape[1:] != bank.shape[1:]:
        raise ValueError("Embeddings must have matching token and hidden dimensions")
    if not len(query) or not len(bank) or not np.isfinite(query).all() or not np.isfinite(bank).all():
        raise ValueError("Empty or non-finite RAF embeddings")
    with torch.inference_mode():
        q = torch.as_tensor(query, device=device).permute(1, 0, 2).contiguous()
        k = torch.as_tensor(bank, device=device).permute(1, 0, 2).contiguous()
        # Direct accumulation also avoids the dot-product cancellation in the
        # matrix-multiplication implementation of small Euclidean distances.
        distances = torch.cdist(q, k, p=2, compute_mode="donot_use_mm_for_euclid_dist").sum(dim=0)
        result = distances.float().cpu().numpy()
    if not np.isfinite(result).all():
        raise ValueError("Non-finite RAF distances")
    return result


class Chronos2Embedder:
    """Use the existing frozen Chronos-2 official embed API, without training.

    Supplying pipeline reuses the exact object used for forecasting.  Supplying
    model_path loads that same snapshot, with no second model download.
    Retrieval intentionally uses target history alone; the original known
    factors are still preserved in the final forecast input.
    """
    def __init__(self, model_path=None, device="cuda", batch_size=64, dtype="float32", pipeline=None):
        import torch
        from chronos import Chronos2Pipeline

        if batch_size < 1 or dtype not in ("float32", "bfloat16", "float16"):
            raise ValueError("Invalid embedding batch size or dtype")
        self.device = device
        self.batch_size = int(batch_size)
        self.model_path = str(model_path) if model_path is not None else "shared-frozen-chronos2-pipeline"
        self.dtype = dtype
        if pipeline is None and model_path is None:
            raise ValueError("Provide the original Chronos-2 snapshot or its loaded pipeline")
        self.pipeline = pipeline if pipeline is not None else Chronos2Pipeline.from_pretrained(
            self.model_path, device_map=device, torch_dtype=getattr(torch, dtype), local_files_only=True)
        if not isinstance(self.pipeline, Chronos2Pipeline):
            raise ValueError("RAF adaptation requires the original Chronos-2 pipeline")
        inner = self.pipeline.model
        inner.eval()
        inner.requires_grad_(False)

    def __call__(self, context):
        import torch

        values = _finite_matrix(context, CONTEXT, "embedding input")
        if not len(values):
            raise ValueError("Cannot embed an empty context batch")
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(values), self.batch_size):
                embedded, _ = self.pipeline.embed(
                    values[start:start + self.batch_size, None, :],
                    batch_size=self.batch_size, context_length=CONTEXT)
                # Each list element is (1, patches+2, hidden).  Independent
                # univariate groups prevent cross-series information sharing.
                if any(item.ndim != 3 or item.shape[0] != 1 for item in embedded):
                    raise ValueError("Unexpected multivariate Chronos-2 embeddings")
                chunks.append(torch.cat(embedded, dim=0).detach().float().cpu().numpy())
        result = np.concatenate(chunks, axis=0)
        if result.ndim != 3 or result.shape[0] != len(values) or result.shape[1] < 1:
            raise ValueError("Unexpected Chronos-2 encoder output shape")
        if not np.isfinite(result).all():
            raise ValueError("Non-finite encoder embeddings")
        return result


def _panel_identity(panel):
    payload = dict(dataset=panel.dataset, series=len(panel.ids), first_day=panel.days[0],
                   last_day=panel.days[-1], schedule=panel.schedule)
    for name in ("panel.json", "values.npy", "covariates.json", "covariates.npy"):
        path = Path(panel.folder) / name
        if not path.is_file():
            raise FileNotFoundError(path)
    payload["folder"] = str(panel.folder)
    return payload


def _encoder_identity(embedder):
    signature = dict(model_path=getattr(embedder, "model_path", "synthetic-test-encoder"),
                     dtype=getattr(embedder, "dtype", "float32"),
                     kind="original_frozen_chronos2_official_embed")
    folder = Path(signature["model_path"])
    if isinstance(embedder, Chronos2Embedder) and not folder.is_dir():
        raise ValueError("A local model_path is required for a persisted RAF plan")
    if folder.is_dir():
        for name in ("config.json", "model.safetensors"):
            if not (folder / name).is_file():
                raise FileNotFoundError(folder / name)
    signature["model_path"] = str(signature["model_path"])
    return signature


def _write_json(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


class RetrievalPlan:
    """A compact immutable plan: query series -> matched fit origin index."""
    def __init__(self, folder, require_complete=True):
        self.folder = Path(folder)
        self.meta = json.loads((self.folder / "plan.json").read_text(encoding="utf-8"))
        if self.meta["version"] != PLAN_VERSION:
            raise ValueError("Unsupported RAF retrieval plan")
        if require_complete and self.meta["status"] != "complete":
            raise ValueError("RAF retrieval plan has not completed")
        self.fit_origins = list(self.meta["fit_origins"])
        self.origins = list(self.meta["origins"])
        self._matches = {}

    def matches(self, origin, rows):
        if origin not in self.origins:
            raise ValueError("Query origin is absent from the frozen RAF plan")
        if origin not in self._matches:
            # These compact int32 vectors are small (one integer per series).
            # Loading them normally also avoids holding Windows file locks.
            self._matches[origin] = np.load(self.folder / f"matches_{origin}.npy")
        index = np.asarray(self._matches[origin][np.asarray(rows, dtype=np.int64)])
        if (index < 0).any() or (index >= len(self.fit_origins)).any():
            raise ValueError("Query series lacks a legal fit-bank RAF match")
        return np.asarray(self.fit_origins, dtype=str)[index]

    def sources(self, origin, rows):
        if origin not in self.origins:
            raise ValueError("Query origin is absent from the frozen RAF plan")
        key = "source_" + origin
        if key not in self._matches:
            self._matches[key] = np.load(self.folder / f"source_rows_{origin}.npy")
        source = np.asarray(self._matches[key][np.asarray(rows, dtype=np.int64)])
        if (source < 0).any() or (source >= self.meta["series"]).any():
            raise ValueError("RAF match lacks a legal source series")
        return source.astype(np.int64)

    def fallbacks(self, origin, rows):
        return self.sources(origin, rows) != np.asarray(rows, dtype=np.int64)


def build_retrieval_plan(panel, embedder, out_dir, origins=None, rows_per_chunk=16,
                         distance_device=None, resume=True):
    """Embed each series' fit bank and eval queries once per series chunk.

    Only compact selected fit indices and distances are written, not hundreds
    of GB of token embeddings.  No target labels from dev/test enter the bank.
    Checkpoints advance only after every requested origin has been flushed.
    A partially flushed chunk is safely overwritten when resumed.
    """
    origins = list(origins if origins is not None else panel.schedule["dev"] + panel.schedule["test"])
    if not origins or origins != sorted(set(origins)):
        raise ValueError("RAF query origins must be nonempty, sorted and unique")
    if not set(origins).issubset(set(panel.schedule["dev"] + panel.schedule["test"])):
        raise ValueError("RAF plan accepts only frozen dev/test query origins")
    if rows_per_chunk < 1:
        raise ValueError("rows_per_chunk must be positive")
    fit = list(panel.schedule["fit"])
    if not fit or any(np.datetime64(fit[-1]) + np.timedelta64(HORIZON, "D") > np.datetime64(o)
                      for o in origins):
        raise ValueError("Fit-bank continuation overlaps a query forecast window")
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    identity = _panel_identity(panel)
    encoder = _encoder_identity(embedder)
    anchor_origin = fit[-1]
    anchor_eligible = sorted(map(int, panel.rows(anchor_origin)), key=lambda row: str(panel.ids[row]))
    if not anchor_eligible:
        raise ValueError("No legal complete fit examples for the deterministic global anchor bank")
    anchor_indices = np.linspace(0, len(anchor_eligible) - 1, min(64, len(anchor_eligible)), dtype=np.int64)
    anchor_rows = np.asarray(anchor_eligible, dtype=np.int64)[anchor_indices]
    contract = dict(version=PLAN_VERSION, dataset=panel.dataset, panel=identity,
                    fit_origins=fit, origins=origins, series=len(panel.ids), encoder=encoder,
                    metric="sum_token_l2_all_chronos2_tokens", bank="same_series_all_legal_fit_origins",
                    cold_start=dict(kind="64_deterministic_global_anchors_from_last_fit",
                                    origin=anchor_origin, rows=anchor_rows.tolist(),
                                    ids=panel.ids[anchor_rows].tolist()),
                    context=CONTEXT, horizon=HORIZON)
    meta_path = output / "plan.json"
    start = 0
    if meta_path.exists():
        if not resume:
            raise FileExistsError("RAF plan already exists and resume is disabled")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for key, value in contract.items():
            if meta.get(key) != value:
                raise ValueError("RAF resume contract changed: " + key)
        start = int(meta["next_row"])
        if not 0 <= start <= len(panel.ids):
            raise ValueError("Invalid RAF checkpoint row")
    else:
        meta = dict(**contract, status="running", next_row=0, completed_chunks=0,
                    seconds=0.0, rows_per_chunk=rows_per_chunk, no_bank_queries=0, checkpoints=[])
        for origin in origins:
            match = np.lib.format.open_memmap(output / f"matches_{origin}.npy", mode="w+",
                                               dtype=np.int32, shape=(len(panel.ids),))
            match[:] = -1
            match.flush()
            distance = np.lib.format.open_memmap(output / f"distances_{origin}.npy", mode="w+",
                                                  dtype=np.float32, shape=(len(panel.ids),))
            distance[:] = np.nan
            distance.flush()
            source = np.lib.format.open_memmap(output / f"source_rows_{origin}.npy", mode="w+",
                                               dtype=np.int32, shape=(len(panel.ids),))
            source[:] = -1
            source.flush()
        _write_json(meta_path, meta)
    match_arrays = {o: np.load(output / f"matches_{o}.npy", mmap_mode="r+") for o in origins}
    distances = {o: np.load(output / f"distances_{o}.npy", mmap_mode="r+") for o in origins}
    source_arrays = {o: np.load(output / f"source_rows_{o}.npy", mmap_mode="r+") for o in origins}
    for arrays in (match_arrays, distances, source_arrays):
        if any(array.shape != (len(panel.ids),) for array in arrays.values()):
            raise ValueError("RAF checkpoint array shape changed")
    checkpoint_end = 0
    for checkpoint in meta["checkpoints"]:
        if checkpoint["start"] != checkpoint_end or not checkpoint_end < checkpoint["end"] <= start:
            raise ValueError("RAF checkpoint row boundaries changed")
        checkpoint_end = checkpoint["end"]
    if checkpoint_end != start:
        raise ValueError("RAF checkpoint does not cover the last committed row")
    if meta["status"] == "complete":
        if start != len(panel.ids):
            raise ValueError("Completed RAF plan has incomplete row coverage")
        return RetrievalPlan(output)
    all_origins = fit + origins
    eligibility = {o: set(map(int, panel.rows(o))) for o in all_origins}
    device = distance_device or getattr(embedder, "device", "cpu")
    started = time.perf_counter()
    previous_seconds = float(meta["seconds"])
    anchor_embeddings = np.asarray(embedder(panel.context(anchor_origin, rows=anchor_rows)), dtype=np.float32)
    for first_row in range(start, len(panel.ids), rows_per_chunk):
        rows = np.arange(first_row, min(first_row + rows_per_chunk, len(panel.ids)), dtype=np.int64)
        contexts, coordinates = [], []
        # Origin-major batches preserve efficient panel reads.  Every context
        # receives a (series row, origin) coordinate before concatenation.
        for origin in all_origins:
            eligible = np.asarray([r for r in rows if int(r) in eligibility[origin]], dtype=np.int64)
            if not len(eligible):
                continue
            contexts.append(panel.context(origin, rows=eligible))
            coordinates.extend((int(r), origin) for r in eligible)
        if contexts:
            embeddings = np.asarray(embedder(np.concatenate(contexts)), dtype=np.float32)
            if len(embeddings) != len(coordinates):
                raise ValueError("Encoder changed query/bank ordering")
            lookup = {coordinate: i for i, coordinate in enumerate(coordinates)}
            for row in rows:
                query_origins = [o for o in origins if (int(row), o) in lookup]
                if not query_origins:
                    continue
                bank_indices = [i for i, o in enumerate(fit) if (int(row), o) in lookup]
                query = embeddings[[lookup[(int(row), o)] for o in query_origins]]
                if bank_indices:
                    bank = embeddings[[lookup[(int(row), fit[i])] for i in bank_indices]]
                    source_candidates = np.full(len(bank_indices), row, dtype=np.int64)
                else:
                    bank = anchor_embeddings
                    bank_indices = [len(fit) - 1] * len(anchor_rows)
                    source_candidates = anchor_rows
                distance = token_l2_distances(query, bank, device=device)
                # np.argmin resolves exact ties to the earliest frozen fit origin.
                selected = distance.argmin(axis=1)
                for q, origin in enumerate(query_origins):
                    match_arrays[origin][row] = bank_indices[int(selected[q])]
                    distances[origin][row] = distance[q, selected[q]]
                    source_arrays[origin][row] = source_candidates[int(selected[q])]
        for array in list(match_arrays.values()) + list(distances.values()) + list(source_arrays.values()):
            array.flush()
        chunk_end = int(rows[-1]) + 1
        meta["checkpoints"].append(dict(start=first_row, end=chunk_end))
        meta.update(next_row=int(rows[-1]) + 1, completed_chunks=int(meta["completed_chunks"]) + 1,
                    seconds=previous_seconds + time.perf_counter() - started)
        _write_json(meta_path, meta)
        print(f"RAF retrieval {panel.dataset}: {meta['next_row']}/{len(panel.ids)} series", flush=True)
    coverage, cold_start_coverage = {}, {}
    for origin in origins:
        rows = panel.rows(origin)
        if ((match_arrays[origin][rows] < 0).any() or (source_arrays[origin][rows] < 0).any()
                or not np.isfinite(distances[origin][rows]).all()):
            raise ValueError("RAF retrieval plan has incomplete evaluation coverage")
        coverage[origin] = int(len(rows))
        cold_start_coverage[origin] = int(np.sum(source_arrays[origin][rows] != rows))
    meta.update(status="complete", coverage=coverage, cold_start_coverage=cold_start_coverage,
                no_bank_queries=int(sum(cold_start_coverage.values())),
                seconds=previous_seconds + time.perf_counter() - started)
    _write_json(meta_path, meta)
    return RetrievalPlan(output)


def augment_from_plan(panel, origin, rows, plan):
    from backbone import covariates_for

    rows = np.asarray(rows, dtype=np.int64)
    query = panel.context(origin, rows=rows)
    matched = plan.matches(origin, rows)
    matched_sources = plan.sources(origin, rows)
    segment = np.empty((len(rows), CONTEXT + HORIZON), dtype=np.float32)
    query_covariates = covariates_for(panel, origin, rows)
    covariates = np.empty((len(rows), AUGMENTED_CONTEXT, query_covariates.shape[-1]), dtype=np.float32)
    for fit_origin in np.unique(matched):
        fit_origin = str(fit_origin)
        if fit_origin not in panel.schedule["fit"]:
            raise ValueError("RAF retrieval escaped the frozen fit bank")
        if np.datetime64(fit_origin) + np.timedelta64(HORIZON, "D") > np.datetime64(origin):
            raise ValueError("RAF retrieved continuation is unavailable at query origin")
        selected = np.flatnonzero(matched == fit_origin)
        source_rows = matched_sources[selected]
        if not np.isin(source_rows, panel.rows(fit_origin)).all():
            raise ValueError("RAF retrieved ineligible fit examples")
        indices = panel.indices(fit_origin)
        segment[selected] = panel.values[np.ix_(source_rows, indices)]
        covariates[selected, :CONTEXT + HORIZON] = covariates_for(panel, fit_origin, source_rows)
    covariates[:, CONTEXT + HORIZON:] = query_covariates[:, :CONTEXT]
    if not np.isfinite(covariates).all():
        raise ValueError("Missing RAF historical covariates; no implicit imputation")
    result = augment_context(query, segment)
    result.past_covariates = covariates
    result.matched_origins = matched
    result.source_rows = matched_sources
    return result
