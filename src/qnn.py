from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.optimize import minimize

from src.encodings import Encoder


@dataclass(frozen=True)
class AnsatzConfig:
    n_qubits: int = 2
    layers: int = 2
    entangle: bool = True

    @property
    def n_params(self) -> int:
        return self.n_qubits * self.layers


DEFAULT_ANSATZ = AnsatzConfig()
Z0 = SparsePauliOp.from_list([("ZI", 1.0)])


@dataclass(frozen=True)
class TrainResult:
    theta_star: np.ndarray
    loss_history: list[float]
    success: bool
    message: str | None


def build_ansatz(theta: Sequence[float], cfg: AnsatzConfig) -> QuantumCircuit:
    n_qubits = cfg.n_qubits
    layers = cfg.layers
    if n_qubits < 1:
        raise ValueError("n_qubits must be >= 1")
    if layers < 1:
        raise ValueError("layers must be >= 1")
    if len(theta) != cfg.n_params:
        raise ValueError("theta length must be n_qubits * layers")

    qc = QuantumCircuit(n_qubits)
    idx = 0
    for layer in range(layers):
        for q in range(n_qubits):
            qc.ry(float(theta[idx]), q)
            idx += 1
        if cfg.entangle and n_qubits > 1 and layer < layers - 1:
            for q in range(n_qubits - 1):
                qc.cx(q, q + 1)
    return qc


def build_qnn_circuit(
    encoder: Encoder, x: np.ndarray, theta: Sequence[float], ansatz_cfg: AnsatzConfig = DEFAULT_ANSATZ
) -> QuantumCircuit:
    enc = encoder.circuit(x)
    ansatz = build_ansatz(theta, ansatz_cfg)
    if enc.num_qubits != ansatz.num_qubits:
        raise ValueError(
            f"Encoding qubits={enc.num_qubits} do not match ansatz qubits={ansatz.num_qubits}"
        )
    return enc.compose(ansatz, inplace=False)


def predict_value(
    encoder: Encoder,
    x: np.ndarray,
    theta: Sequence[float],
    observable: SparsePauliOp = Z0,
    ansatz_cfg: AnsatzConfig = DEFAULT_ANSATZ,
) -> float:
    qc = build_qnn_circuit(encoder, x, theta, ansatz_cfg)
    sv = Statevector.from_instruction(qc)
    return float(np.real(sv.expectation_value(observable)))


def mse_loss(
    encoder: Encoder,
    theta: Sequence[float],
    X: np.ndarray,
    y: np.ndarray,
    ansatz_cfg: AnsatzConfig = DEFAULT_ANSATZ,
    observable: SparsePauliOp = Z0,
) -> float:
    preds = np.array(
        [predict_value(encoder, x, theta, observable=observable, ansatz_cfg=ansatz_cfg) for x in X],
        dtype=float,
    )
    return float(np.mean((preds - y) ** 2))


def train_qnn(
    encoder: Encoder,
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 0,
    maxiter: int = 100,
    ansatz_cfg: AnsatzConfig = DEFAULT_ANSATZ,
    observable: SparsePauliOp = Z0,
    log_every: int = 5,
) -> TrainResult:
    rng = np.random.default_rng(seed)
    theta0 = rng.uniform(-0.1, 0.1, size=ansatz_cfg.n_params)

    loss_history: list[float] = []
    if log_every < 1:
        raise ValueError("log_every must be >= 1")

    def loss_fn(theta: np.ndarray) -> float:
        return mse_loss(encoder, theta, X, y, ansatz_cfg=ansatz_cfg, observable=observable)

    loss_history.append(loss_fn(theta0))
    step = 0

    def callback(theta_k: np.ndarray) -> None:
        nonlocal step
        step += 1
        if step % log_every == 0:
            loss_history.append(loss_fn(theta_k))

    res = minimize(
        fun=loss_fn,
        x0=theta0,
        method="COBYLA",
        callback=callback,
        options={"maxiter": int(maxiter)},
    )

    return TrainResult(
        theta_star=np.array(res.x, dtype=float),
        loss_history=loss_history,
        success=bool(res.success),
        message=str(res.message) if res.message is not None else None,
    )
