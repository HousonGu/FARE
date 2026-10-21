"""FARE's fixed-budget shape and factor-conditioned reference construction.

Inputs represent one already selected dataset-specific reference group. Rows must
be ordered by reference origin, then series ID, to reproduce stable tie breaking.
"""
from dataclasses import dataclass

import numpy as np

CONTEXT = 56
HORIZON = 7
REFERENCE_COUNT = 8


def _matrix(value, name, width=None):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or (width is not None and value.shape[1] != width):
        raise ValueError(f"{name} must be a two-dimensional array" +
                         (f" with {width} columns" if width is not None else ""))
    if not np.isfinite(value).all() or (value < 0).any():
        raise ValueError(f"{name} must contain finite nonnegative values")
    return value


def select(shape, factor1, factor2):
    """Select eight unique windows, with stable ties and four per factor branch.

    All arguments have shape ``(n_queries, n_references)``. Shape is normalized
    history MSE; factors are seven-day mismatch fractions in [0, 1]. The factor
    reference alternates branches, starting with factor 1, and skips duplicates.
    """
    shape = _matrix(shape, "shape")
    factor1, factor2 = _matrix(factor1, "factor1"), _matrix(factor2, "factor2")
    if shape.shape != factor1.shape or shape.shape != factor2.shape:
        raise ValueError("Distance arrays must have the same shape")
    if shape.shape[0] == 0 or shape.shape[1] < REFERENCE_COUNT:
        raise ValueError("Need at least one query and eight shared references")
    if (factor1 > 1).any() or (factor2 > 1).any():
        raise ValueError("Factor mismatch fractions must lie in [0, 1]")
    bounded = shape / (1.0 + shape)
    joint = (bounded + factor1 + factor2) / 3
    branch1, branch2 = (bounded + factor1) / 2, (bounded + factor2) / 2
    selections = {
        "shape": np.argsort(shape, axis=1, kind="stable")[:, :REFERENCE_COUNT],
        "joint": np.argsort(joint, axis=1, kind="stable")[:, :REFERENCE_COUNT],
    }
    order1, order2 = (np.argsort(branch, axis=1, kind="stable")
                      for branch in (branch1, branch2))
    picked = []
    for first, second in zip(order1, order2):
        chosen, pointers = [], [0, 0]
        for turn in range(REFERENCE_COUNT):
            branch = turn % 2
            order = first if branch == 0 else second
            while int(order[pointers[branch]]) in chosen:
                pointers[branch] += 1
            chosen.append(int(order[pointers[branch]]))
            pointers[branch] += 1
        picked.append(chosen)
    selections["factor"] = np.asarray(picked, dtype=np.int64)
    scores = {"shape": shape, "joint": joint, "factor1_branch": branch1,
              "factor2_branch": branch2, "promo_branch": branch1,
              "holiday_branch": branch2}
    return selections, scores


@dataclass(frozen=True)
class RetrievalResult:
    """Forecast candidates and selection diagnostics for a query batch."""

    shape: np.ndarray
    factor: np.ndarray
    shape_indices: np.ndarray
    factor_indices: np.ndarray
    factor1_score: np.ndarray
    factor2_score: np.ndarray
    bank_size: int
    fallback: bool


def retrieve_references(history, future, query_history, factor1, factor2,
                        query_factor1, query_factor2, *, reference_origins=None,
                        query_origin=None, query_batch_size=8):
    """Build normalized Top-8 candidates for one legal same-group bank.

    Reference history and future have shapes ``(n_references, 56)`` and
    ``(n_references, 7)``. Query arrays use ``n_queries`` rows. Factor arrays
    contain nonnegative category codes and have seven columns. Optional dates
    filter the bank to ``reference_origin + 6 days < query_origin``; supply both
    arguments together. Without dates the caller must enforce that boundary.

    Returned indices address the original input bank. If fewer than eight legal
    windows remain, both candidates use the query's previous-week sales. Dataset
    grouping and Favorita's controlled-combination exclusion are caller duties.
    Queries are processed in batches of eight by default, matching the original
    reference drivers and bounding the distance temporary to one query batch.
    """
    if (isinstance(query_batch_size, (bool, np.bool_)) or
            not isinstance(query_batch_size, (int, np.integer)) or query_batch_size <= 0):
        raise ValueError("query_batch_size must be a positive integer")
    history = _matrix(history, "history", CONTEXT)
    future = _matrix(future, "future", HORIZON)
    query_history = _matrix(query_history, "query_history", CONTEXT)
    factor1, factor2 = (_matrix(value, name, HORIZON) for value, name in
                        ((factor1, "factor1"), (factor2, "factor2")))
    query_factor1, query_factor2 = (
        _matrix(value, name, HORIZON) for value, name in
        ((query_factor1, "query_factor1"), (query_factor2, "query_factor2")))
    n, nq = len(history), len(query_history)
    if nq == 0:
        raise ValueError("Need at least one query")
    if any(len(v) != n for v in (future, factor1, factor2)):
        raise ValueError("Reference arrays must have equal row counts")
    if any(len(v) != nq for v in (query_factor1, query_factor2)):
        raise ValueError("Query arrays must have equal row counts")
    if (reference_origins is None) != (query_origin is None):
        raise ValueError("Supply reference_origins and query_origin together")
    eligible = np.arange(n)
    if reference_origins is not None:
        dates = np.asarray(reference_origins, dtype="datetime64[ns]")
        origin = np.asarray(query_origin, dtype="datetime64[ns]")
        if dates.shape != (n,) or origin.ndim != 0:
            raise ValueError("Reference dates must be a row vector and query origin a scalar")
        if np.isnat(dates).any() or np.isnat(origin):
            raise ValueError("Reference dates and query origin cannot be missing")
        eligible = eligible[dates + np.timedelta64(HORIZON - 1, "D") < origin]
    size = len(eligible)
    if size < REFERENCE_COUNT:
        previous_week = query_history[:, -HORIZON:].copy()
        empty = np.empty((nq, 0), dtype=np.int64)
        zeros = np.zeros(nq, dtype=np.float64)
        return RetrievalResult(previous_week, previous_week.copy(), empty, empty.copy(),
                               zeros, zeros.copy(), size, True)
    history, future, factor1, factor2 = (v[eligible] for v in
                                        (history, future, factor1, factor2))
    reference_scale = np.maximum(history.mean(axis=1), 1)
    query_scale = np.maximum(query_history.mean(axis=1), 1)
    normalized = history / reference_scale[:, None]
    normalized_future = future / reference_scale[:, None]
    shape, factor = np.empty((nq, HORIZON)), np.empty((nq, HORIZON))
    shape_indices = np.empty((nq, REFERENCE_COUNT), dtype=np.int64)
    factor_indices = np.empty((nq, REFERENCE_COUNT), dtype=np.int64)
    score1, score2 = np.empty(nq), np.empty(nq)
    for start in range(0, nq, int(query_batch_size)):
        rows = slice(start, start + int(query_batch_size))
        scale = query_scale[rows]
        # Preserve the original direct-MSE arithmetic for each query and tie.
        shape_distance = ((query_history[rows, None, :] / scale[:, None, None]
                           - normalized[None, :, :]) ** 2).mean(axis=2)
        mismatch1 = (query_factor1[rows, None, :] != factor1[None, :, :]).mean(axis=2)
        mismatch2 = (query_factor2[rows, None, :] != factor2[None, :, :]).mean(axis=2)
        chosen, scores = select(shape_distance, mismatch1, mismatch2)
        shape[rows] = normalized_future[chosen["shape"]].mean(axis=1) * scale[:, None]
        factor[rows] = normalized_future[chosen["factor"]].mean(axis=1) * scale[:, None]
        shape_indices[rows], factor_indices[rows] = eligible[chosen["shape"]], eligible[chosen["factor"]]
        score1[rows] = np.take_along_axis(scores["factor1_branch"], chosen["factor"][:, ::2], 1).mean(1)
        score2[rows] = np.take_along_axis(scores["factor2_branch"], chosen["factor"][:, 1::2], 1).mean(1)
    return RetrievalResult(shape, factor, shape_indices, factor_indices, score1, score2, size, False)
