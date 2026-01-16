# src/data.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class DatasetSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def make_linear(n: int = 200, seed: int = 0, noise: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    # linear boundary: x + y > 0
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    X = X + rng.normal(scale=noise, size=X.shape)
    return X, y


def make_xor(n: int = 200, seed: int = 0, noise: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 2))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    X = X + rng.normal(scale=noise, size=X.shape)
    return X, y


def train_test_split(X: np.ndarray, y: np.ndarray, test_ratio: float = 0.3, seed: int = 0) -> DatasetSplit:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    cut = int(len(X) * (1.0 - test_ratio))
    tr, te = idx[:cut], idx[cut:]
    return DatasetSplit(X[tr], y[tr], X[te], y[te])


def standardize_fit_transform(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Global preprocessing shared across encoders (kept minimal)."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    return (X_train - mean) / std, (X_test - mean) / std
