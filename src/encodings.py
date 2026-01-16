# src/encodings.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Tuple
import numpy as np
from qiskit import QuantumCircuit
from typing import Optional


class Encoder(Protocol):
    name: str

    def n_qubits(self, n_features: int) -> int: ...
    def preprocess(self, x: np.ndarray) -> np.ndarray: ...
    def circuit(self, x: np.ndarray) -> QuantumCircuit: ...


@dataclass(frozen=True)
class BasisEncoding:
    name: str = "basis"
    threshold: float = 0.0  # binarize: x > threshold -> 1

    def n_qubits(self, n_features: int) -> int:
        return n_features

    def preprocess(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).ravel()
        return (x > self.threshold).astype(int)

    def circuit(self, x: np.ndarray) -> QuantumCircuit:
        bits = self.preprocess(x)
        qc = QuantumCircuit(len(bits))
        for i, b in enumerate(bits):
            if b == 1:
                qc.x(i)
        return qc


@dataclass(frozen=True)
class AngleEncoding:
    name: str = "angle"
    angle_range: Tuple[float, float] = (-np.pi, np.pi)

    def n_qubits(self, n_features: int) -> int:
        return n_features

    def preprocess(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).ravel()
        # Minimal, explicit scaling rule:
        # normalize feature-wise to [-1, 1] then map to angle_range
        denom = np.max(np.abs(x)) if np.max(np.abs(x)) != 0 else 1.0
        x_norm = x / denom  # now in [-1,1] per sample
        lo, hi = self.angle_range
        return lo + (x_norm + 1.0) * (hi - lo) / 2.0

    def circuit(self, x: np.ndarray) -> QuantumCircuit:
        angles = self.preprocess(x)
        qc = QuantumCircuit(len(angles))
        for i, theta in enumerate(angles):
            qc.ry(float(theta), i)
        return qc


@dataclass(frozen=True)
class AmplitudeEncoding:
    name: str = "amplitude"

    def n_qubits(self, n_features: int) -> int:
        # smallest n such that 2^n >= n_features
        return int(np.ceil(np.log2(max(1, n_features))))

    def preprocess(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).ravel()
        n = self.n_qubits(len(x))
        dim = 2**n
        if len(x) < dim:
            x = np.pad(x, (0, dim - len(x)))    # pad with zeros if less than required dimension
        elif len(x) > dim:
            x = x[:dim]  # explicit truncation rule if more than required dimension
        norm = np.linalg.norm(x)
        if norm == 0:
            # deterministic fallback: encode |0...0>
            x = np.zeros(dim, dtype=float)
            x[0] = 1.0
            return x
        return x / norm

    def circuit(self, x: np.ndarray) -> QuantumCircuit:
        amps = self.preprocess(x)
        n = self.n_qubits(len(x))
        qc = QuantumCircuit(n)
        qc.initialize(amps.astype(complex).tolist(), list(range(n)))
        return qc
    
@dataclass(frozen=True)
class ReuploadingAngleEncoding:
    """
    Data re-uploading variant of angle encoding.

    Idea:
      Repeat the data-encoding rotations multiple times (reps),
      optionally separated by a light entangling pattern.
    This increases expressivity at the cost of circuit depth.

    Notes:
      - This is still a *pure encoding* circuit (no trainable ansatz here).
      - We reuse the same preprocessing as AngleEncoding.
    """
    name: str = "angle_reupload"
    reps: int = 3
    angle_range: Tuple[float, float] = (-np.pi, np.pi)
    entangle: bool = True

    def n_qubits(self, n_features: int) -> int:
        return n_features

    def preprocess(self, x: np.ndarray) -> np.ndarray:
        # Same scaling rule as AngleEncoding (keep comparisons fair)
        x = np.asarray(x, dtype=float).ravel()
        denom = np.max(np.abs(x)) if np.max(np.abs(x)) != 0 else 1.0
        x_norm = x / denom  # [-1,1] per sample
        lo, hi = self.angle_range
        return lo + (x_norm + 1.0) * (hi - lo) / 2.0

    def circuit(self, x: np.ndarray) -> QuantumCircuit:
        angles = self.preprocess(x)
        n = len(angles)
        qc = QuantumCircuit(n)

        reps = max(1, int(self.reps))

        for r in range(reps):
            # Re-upload the same data
            for i, theta in enumerate(angles):
                qc.ry(float(theta), i)

            # Optional lightweight entangling between uploads
            if self.entangle and n > 1 and r != reps - 1:
                for i in range(n - 1):
                    qc.cx(i, i + 1)

        return qc


def circuit_stats(qc: QuantumCircuit) -> dict:
    return {
        "n_qubits": qc.num_qubits,
        "depth": qc.depth(),
        "size": qc.size(),
        "ops": qc.count_ops(),
    }
