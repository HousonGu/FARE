"""Dataset-specific gate features in the order used by the experiments."""
import numpy as np

from .fusion import HISTORY_NAMES, base_features


def build_features(frame, dataset, *, history=None):
    """Return ``(raw_features, feature_names, binary_feature_names)``.

    Pass the four columns from ``HistoricalErrorState.features`` through
    ``history`` to obtain the complete 15/12/14/14-column design for
    Favorita/M5/Rossmann/Rohlik. Otherwise only the origin and disagreement
    features are returned. Intercept and development standardization are added
    by ``fit_fusion``. ``frame`` may be a DataFrame or a mapping of arrays.
    """
    raw, names, _ = base_features(frame, dataset)
    binary_by_dataset = {"favorita": {"holiday"}, "m5": {"snap", "event"},
                         "rossmann": {"promo", "school_holiday"},
                         "rohlik": {"discount", "holiday"}}
    if history is not None:
        history = np.asarray(history, dtype=np.float64)
        if history.shape != (len(raw), 4) or not np.isfinite(history).all():
            raise ValueError("Historical features must be a finite (n_rows, 4) array")
        raw = np.column_stack([raw, history])
        names = names + HISTORY_NAMES
    return raw, names, binary_by_dataset[dataset.lower()].copy()


__all__ = ["base_features", "build_features"]
