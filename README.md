# Information Encoding in Variational Quantum Algorithms (VQE + QNN)

This project compares quantum data encoding methods and their impact on optimization behavior and
performance in variational algorithms. The experiments use a controlled VQE setup and a minimal
QNN classifier, keeping the ansatz and optimizer fixed while varying only the encoding.

## Encodings Implemented

- Basis encoding
- Angle encoding
- Angle encoding with data re-uploading
- Amplitude encoding

All encoders expose a common interface (`encoder.circuit(x)`) so they can be swapped consistently.

## Project Structure

- `src/encodings.py`: encoding implementations and circuit stats
- `src/vqe.py`: VQE core (Hamiltonian builder, ansatz, expectation, run loop)
- `src/qnn.py`: QNN core (circuit builder, prediction, loss, training)
- `src/data.py`: toy datasets + standardization helpers
- `notebooks/00_encoding_smoke_test.ipynb`: sanity checks for all encodings
- `notebooks/01_vqe_encoding_compare.ipynb`: VQE encoding comparison + plots
- `notebooks/02_qnn_minimal_angle.ipynb`: minimal QNN with angle encoding
- `notebooks/03_qnn_encoding_compare.ipynb`: QNN encoding comparison
- `notes/vqe_experiment_spec.md`: fixed VQE experiment design

## How to Run

1) Create a virtual environment and install dependencies:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

2) Run the notebooks in order (recommended):

- `00_encoding_smoke_test.ipynb`
- `01_vqe_encoding_compare.ipynb`
- `02_qnn_minimal_angle.ipynb`
- `03_qnn_encoding_compare.ipynb`

## Outputs

The notebooks save artifacts automatically:

- `results/vqe/`: VQE run logs and plots
- `results/vqe/circuits/`: encoding circuit snapshots
- `results/figures/circuits/`: full VQE circuit example
- `results/qnn/`: QNN plots (loss curves, decision boundaries)
