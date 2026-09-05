# Adaptive Signal Acquisition

This repository implements a staged simulation study of adaptive signal acquisition.

## Stage 1: deterministic toy simulator

Stage 1 intentionally contains **no reconstruction, adaptive sensing policy, LLM, or reinforcement learning**. It makes the model

\[
y = A x + \epsilon
\]

concrete and reproducible.

The simulator:

1. generates a directly sparse one-dimensional signal `x`;
2. creates a finite dictionary of allowed, normalized measurement vectors;
3. selects a fixed subset of those vectors before seeing any observations;
4. simulates clean measurements `A @ x`, Gaussian noise, and noisy observations `y`;
5. saves every relevant array and plots the signal, measurements, and noise.

Here, `A` is the matrix whose rows are the selected measurement vectors. A fixed selection means that its rows are chosen before any measurement values are observed; it is the non-adaptive starting point.

## Setup

Create and activate a virtual environment, then install the dependencies:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### PyCharm note

The installed PyCharm Community 2022.1 package-management window is too old to install packages into a Python 3.12 environment: its bundled `pip` imports the removed `distutils` module. This does not affect the project interpreter or the code itself.

In PyCharm, select `.venv\\Scripts\\python.exe` as the project interpreter, open the built-in **Terminal** tool window, and use the commands above there. The terminal uses the current virtual environment's modern `pip`, not PyCharm's broken package-management helper. Upgrading PyCharm is recommended, but not required to run this Stage 1 project.

## Run the toy simulator

```powershell
python scripts/run_toy_simulator.py --seed 20260905
```

This writes a reproducible run to `artifacts/toy_run/`:

- `arrays.npz` contains `x`, the sparse support, the full candidate dictionary, selected action indices, `A`, clean measurements, noise, and `y`;
- `metadata.json` records the configuration and array shapes;
- `toy_run.png` is the visual sanity-check plot.

Run the command twice with the same seed to obtain byte-identical numerical arrays.

## Tests

```powershell
python -m pytest
```
