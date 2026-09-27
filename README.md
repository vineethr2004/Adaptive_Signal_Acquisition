# Adaptive Signal Acquisition

This repository implements a staged simulation study of adaptive signal acquisition.

## Stage 1: deterministic foundations

Stage 1 intentionally contains **no reconstruction, adaptive sensing policy, LLM, or reinforcement learning**. It makes the model

\[
y = A x + \epsilon
\]

concrete and reproducible.

The foundations code:

1. generates a directly sparse one-dimensional signal `x`;
2. creates a finite dictionary of allowed, normalized measurement vectors;
3. selects a fixed subset of those vectors before seeing any observations;
4. simulates clean measurements `A @ x`, Gaussian noise, and noisy observations `y`;
5. provides reproducible arrays and visual checks for the signal, measurements, and noise.

Here, `A` is the matrix whose rows are the selected measurement vectors. A fixed selection means that its rows are chosen before any measurement values are observed; it is the non-adaptive starting point.

## Setup

Create and activate a virtual environment, then install the dependencies:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### PyCharm note

The installed PyCharm Community 2022.1 package-management window is too old to install packages into a Python 3.12 environment: its bundled `pip` imports the removed `distutils` module. This does not affect the project interpreter or the code itself.

In PyCharm, select `.venv\\Scripts\\python.exe` as the project interpreter, open the built-in **Terminal** tool window, and use the commands above there. The terminal uses the current virtual environment's modern `pip`, not PyCharm's broken package-management helper. Upgrading PyCharm is recommended, but not required to run this project.

## Stage 2: reconstruction and non-adaptive baselines

Stage 2 adds a single transparent LASSO reconstructor and compares two non-adaptive sensing designs:

- **fixed**: use the same preselected action rows in every trial;
- **random**: preselect action rows at random before each trial's observations.

Both methods receive the same hidden signal and the same pre-drawn noise values in each paired trial. They differ only in their selected rows of `A`. This establishes the fair reference point needed before adding adaptive sensing.

## Run the Stage 2 baseline study

```powershell
python scripts/run_baseline_study.py
```

This writes a reproducible study to `artifacts/stage2_baselines/`:

- `trial_records.csv` contains every fixed/random trial result;
- `baseline_summary.csv` contains mean and median NMSE, support F1, runtime, and convergence rate by budget;
- `nmse_vs_budget.png` compares mean NMSE across measurement budgets;
- `metadata.json` records the frozen configuration.

Run the command twice with the same seed to obtain identical signal, action, noise, and reconstruction-quality results. Runtime values naturally vary slightly between runs.

## Stage 3: sequential information-guided sensing

Stage 3 keeps the Stage 2 LASSO reconstruction and paired comparison protocol, then adds a third method:

- **information-guided**: select one unused candidate measurement row at a time. Before choosing it, use only the earlier action/observation history to score each available row by its expected information.

The policy begins with a Gaussian uncertainty approximation and uses the score

\[
\log\!\left(1 + \frac{a^T \Sigma a}{\sigma^2}\right).
\]

After each observation it updates that uncertainty. It also forms a provisional LASSO estimate from the *past* measurements and uses a documented support-aware uncertainty proxy. This proxy is intentionally simple rather than claiming to be an exact sparse Bayesian posterior; its purpose is to make later choices depend on what has actually been observed. The policy function has no `x` argument, so the hidden signal cannot leak into a choice.

## Run the Stage 3 comparison

```powershell
python scripts/run_adaptive_study.py
```

This writes reproducible artifacts to `artifacts/stage3_adaptive/`:

- `trial_records.csv` — reconstruction results for fixed, random, and information-guided sensing;
- `adaptive_summary.csv` — NMSE, support F1, runtime, and convergence summaries;
- `action_traces.csv` — every adaptive action, observation, and information score;
- `nmse_vs_budget.png` — the fair three-method comparison;
- `metadata.json` — the exact configuration.

## Stage 4A: diagnose the first adaptive policy

Stage 4A freezes the Stage 3 policy as `adaptive_v1` and varies only the LASSO-proxy weight
`gamma`. For every selected action it separately records:

- the Gaussian uncertainty contribution `a.T @ Sigma @ a`;
- the history-dependent LASSO contribution `gamma * (a.T @ x_hat)**2`;
- row correlation, smallest singular value, and sensing-matrix condition number;
- the provisional LASSO NMSE and support F1 after the resulting observation.

It also reports paired NMSE differences against random sensing with deterministic bootstrap
95% confidence intervals.

```powershell
python scripts/run_stage4a_diagnostics.py
```

Artifacts are written to `artifacts/stage4a_diagnostics/`. The purpose is diagnosis: the
current adaptive method remains unchanged, and a later `adaptive_v2` will be introduced only
after these ablations identify what is failing.

## Stage 4B: bootstrap reconstruction uncertainty

Stage 4B keeps `adaptive_v1` frozen and introduces `adaptive_v2` as a separate policy. It
uses eight Gaussian-diverse burn-in measurements, then perturbs the observed measurement
history, reconstructs an ensemble of plausible sparse signals, and measures their empirical
covariance. Candidate rows are scored where these plausible reconstructions disagree, with
Gaussian covariance shrinkage retained to preserve sensing-matrix diversity.

```powershell
python scripts/run_stage4b_study.py
```

Artifacts are written to `artifacts/stage4b_adaptive_v2/`: paired trial and summary CSVs,
decision-level diagnostics, exact metadata, and three figures. The comparison includes fixed,
random, Gaussian-only, frozen `adaptive_v1`, and `adaptive_v2`, all under the same hidden
signals, noise draws, budgets, and final LASSO reconstructor.

## Tests

```powershell
python -m pytest
```
