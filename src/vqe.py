from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.optimize import minimize

from src.encodings import Encoder

DEFAULT_HAMILTONIAN_TERMS: list[tuple[str, float]] = [
    ("ZI", 0.5),
    ("IZ", 0.5),
    ("ZZ", 0.2),
    ("XI", 0.3),
    ("IX", 0.3),
]


@dataclass(frozen=True)
class VQERunResult:
    encoding: str
    seed: int
    x: np.ndarray
    energies: np.ndarray
    best_energies: np.ndarray
    final_energy: float
    final_error: float
    params: np.ndarray


def build_hamiltonian(terms: Iterable[tuple[str, float]], n_qubits: int) -> SparsePauliOp:
    term_list = list(terms)
    for pauli, _ in term_list:
        if len(pauli) != n_qubits:
            raise ValueError(f"Pauli string {pauli!r} does not match n_qubits={n_qubits}")
    return SparsePauliOp.from_list(term_list)


def exact_ground_energy(hamiltonian: SparsePauliOp) -> float:
    mat = hamiltonian.to_matrix()
    evals = np.linalg.eigvalsh(mat)
    return float(np.min(np.real(evals)))


def hardware_efficient_ansatz(n_qubits: int, layers: int = 2) -> tuple[QuantumCircuit, list[Parameter]]:
    if n_qubits < 1:
        raise ValueError("n_qubits must be >= 1")
    if layers < 1:
        raise ValueError("layers must be >= 1")

    theta = ParameterVector("theta", length=n_qubits * layers)
    qc = QuantumCircuit(n_qubits)

    idx = 0
    for _ in range(layers):
        for q in range(n_qubits):
            qc.ry(theta[idx], q)
            idx += 1
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)

    return qc, list(theta)


def compose_encoding_ansatz(encoder: Encoder, x: np.ndarray, ansatz: QuantumCircuit) -> QuantumCircuit:
    enc = encoder.circuit(x)
    if enc.num_qubits != ansatz.num_qubits:
        raise ValueError(
            f"Encoding qubits={enc.num_qubits} do not match ansatz qubits={ansatz.num_qubits}"
        )
    return enc.compose(ansatz, inplace=False)


def expectation_value(
    qc: QuantumCircuit, params: list[Parameter], values: np.ndarray, hamiltonian: SparsePauliOp
) -> float:
    if len(params) != len(values):
        raise ValueError("Parameter length does not match values length")
    bind = {p: float(v) for p, v in zip(params, values)}
    bound = qc.assign_parameters(bind, inplace=False)
    state = Statevector.from_instruction(bound)
    return float(np.real(state.expectation_value(hamiltonian)))


def cobyla_minimize(
    objective: Callable[[np.ndarray], float],
    x0: np.ndarray,
    maxiter: int,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    energies: list[float] = []
    best_energies: list[float] = []
    best = float("inf")

    def record(val: float) -> None:
        nonlocal best
        energies.append(float(val))
        if val < best:
            best = float(val)
        best_energies.append(best)

    def obj(x: np.ndarray) -> float:
        return float(objective(x))

    def cb(xk: np.ndarray) -> None:
        record(objective(xk))

    record(objective(x0))

    res = minimize(
        obj,
        np.array(x0, dtype=float),
        method="COBYLA",
        callback=cb,
        options={"maxiter": int(maxiter)},
    )

    return res.x, float(res.fun), np.asarray(energies), np.asarray(best_energies)


def run_vqe(
    encoder: Encoder,
    x: np.ndarray,
    hamiltonian: SparsePauliOp,
    n_qubits: int = 2,
    layers: int = 2,
    maxiter: int = 200,
    seed: int = 0,
    ground_energy: float | None = None,
) -> VQERunResult:
    ansatz, params = hardware_efficient_ansatz(n_qubits=n_qubits, layers=layers)
    full_circuit = compose_encoding_ansatz(encoder, x, ansatz)

    def objective(theta: np.ndarray) -> float:
        return expectation_value(full_circuit, params, theta, hamiltonian)

    rng = np.random.default_rng(seed)
    theta0 = rng.uniform(-np.pi, np.pi, size=len(params))

    opt_params, best_energy, energies, best_energies = cobyla_minimize(
        objective, theta0, maxiter=maxiter
    )

    if ground_energy is None:
        ground_energy = exact_ground_energy(hamiltonian)
    final_error = abs(best_energy - ground_energy)

    return VQERunResult(
        encoding=getattr(encoder, "name", "unknown"),
        seed=seed,
        x=np.asarray(x, dtype=float).ravel(),
        energies=energies,
        best_energies=best_energies,
        final_energy=float(best_energy),
        final_error=float(final_error),
        params=opt_params,
    )
