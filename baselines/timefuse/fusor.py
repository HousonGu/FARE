"""TimeFuse meta features and fusor applied to the stored B/S/F candidates.

The meta-feature extractor is the upstream TimeFuse implementation
(third_party/TimeFuse/meta_feature.py). The fusor is the same linear layer with a
softmax over the three candidates as upstream timefuse.py.
"""
from __future__ import annotations

import math
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "third_party" / "TimeFuse"))


FEATURE_NAMES = (
    "mean", "std", "min", "max", "skewness", "kurtosis", "autocorrelation_mean",
    "stationarity", "rate_of_change_mean", "rate_of_change_std", "autoreg_coef_mean",
    "residual_std_mean", "frequency_mean", "frequency_peak", "spectral_entropy",
    "spectral_variation", "spectral_skewness", "spectral_kurtosis", "covariance_mean",
    "covariance_max", "covariance_min", "covariance_std",
)
SEED = 2021
EPOCHS = 5
BATCH = 64

class ModelFusor(nn.Module):
    """Same linear-plus-softmax fusor as upstream TimeFuse timefuse.py."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        init.kaiming_uniform_(self.fc.weight, a=math.sqrt(5))

    def forward(self, meta_features: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.fc(meta_features), dim=-1)


def meta_row(history: np.ndarray, counts: dict[str, int]) -> np.ndarray:
    try:
        from meta_feature import extract_meta_feature
    except ImportError as error:
        raise ImportError("TimeFuse source missing: run bash scripts/fetch_third_party.sh TimeFuse first") from error
    if history.shape != (56,) or not np.isfinite(history).all() or (history < 0).any():
        raise ValueError("History must be 56 finite non-negative days")
    if float(np.std(history)) == 0:
        counts["constant"] += 1
        constant = float(history[0])
        row = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
        row[FEATURE_NAMES.index("mean")] = constant
        row[FEATURE_NAMES.index("min")] = constant
        row[FEATURE_NAMES.index("max")] = constant
        return row
    data = history.reshape(56, 1)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            features = extract_meta_feature(data)
    except Exception:
        counts["exception"] += 1
        return np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    row = np.array([features[name] for name in FEATURE_NAMES], dtype=np.float64)
    bad = ~np.isfinite(row)
    counts["nonfinite"] += int(bad.sum())
    row[bad] = 0
    return row


def train(meta: np.ndarray, forecasts: np.ndarray, actual: np.ndarray, device: torch.device):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    scaler_mean = meta.mean(axis=0)
    scaler_std = np.maximum(meta.std(axis=0), 1e-8)
    scaled = (meta - scaler_mean) / scaler_std
    fusor = ModelFusor(len(FEATURE_NAMES), 3).to(device)
    optimizer = torch.optim.Adam(fusor.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = nn.SmoothL1Loss(beta=0.01)
    loader = DataLoader(
        TensorDataset(torch.tensor(scaled, dtype=torch.float32),
                      torch.tensor(forecasts, dtype=torch.float32),
                      torch.tensor(actual, dtype=torch.float32)),
        batch_size=BATCH, shuffle=True)
    fusor.train()
    for epoch in range(EPOCHS):
        losses = []
        for x_meta, y_pred, y_true in loader:
            x_meta, y_pred, y_true = x_meta.to(device), y_pred.to(device), y_true.to(device)
            weights = fusor(x_meta).unsqueeze(-1)
            fused = torch.sum(weights * y_pred, dim=1)
            loss = criterion(fused, y_true)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if not torch.isfinite(weights).all():
                raise ValueError("Nonfinite fusion weight")
            losses.append(float(loss.detach()))
        scheduler.step()
        print(f"epoch {epoch + 1} train_loss {np.mean(losses):.6f}", flush=True)
    fusor.eval()
    return fusor, scaler_mean, scaler_std


def predict(fusor: ModelFusor, meta: np.ndarray, forecasts: np.ndarray,
            scaler_mean: np.ndarray, scaler_std: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    scaled = (meta - scaler_mean) / scaler_std
    loader = DataLoader(
        TensorDataset(torch.tensor(scaled, dtype=torch.float32),
                      torch.tensor(forecasts, dtype=torch.float32)),
        batch_size=BATCH, shuffle=False)
    weights, fused = [], []
    with torch.no_grad():
        for x_meta, y_pred in loader:
            w = fusor(x_meta.to(device))
            y = torch.sum(w.unsqueeze(-1) * y_pred.to(device), dim=1)
            weights.append(w.cpu().numpy())
            fused.append(y.cpu().numpy())
    return np.vstack(weights), np.vstack(fused)
