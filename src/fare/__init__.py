"""Factor-Aware Retrieval with Error-Guided Fusion for retail forecasting."""
from .retrieval import RetrievalResult, retrieve_references, select
from .fusion import (HISTORY_NAMES, FusionModel, HistoricalErrorState, base_features,
                     fit_fusion, fit_gate, historical_features, predict_gate)
from .features import build_features

__version__ = "0.1.0"
__all__ = ["RetrievalResult", "retrieve_references", "select", "HISTORY_NAMES",
           "FusionModel", "HistoricalErrorState", "base_features", "fit_fusion",
           "fit_gate", "historical_features", "predict_gate", "build_features"]
