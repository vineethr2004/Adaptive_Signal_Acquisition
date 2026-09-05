"""Run the Stage 1 toy simulator and save reproducible artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from adaptive_signal_acquisition import ToyConfig, ToyRun, run_toy_simulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the deterministic Stage 1 toy simulator.")
    parser.add_argument("--seed", type=int, default=ToyConfig.seed)
    parser.add_argument("--output-dir", type=Path, default=REPOSITORY_ROOT / "artifacts" / "toy_run")
    return parser.parse_args()


def save_plot(run: ToyRun, output_path: Path) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(11, 9), constrained_layout=True)

    axes[0].stem(np.arange(run.x.size), run.x, basefmt=" ")
    axes[0].set(title="Hidden sparse signal x", xlabel="Signal index", ylabel="Amplitude")
    axes[0].axhline(0.0, color="black", linewidth=0.8)

    measurement_index = np.arange(run.observations.size)
    axes[1].plot(measurement_index, run.clean_measurements, "o-", label="A @ x")
    axes[1].plot(measurement_index, run.observations, "s--", label="y = A @ x + noise")
    axes[1].set(title="Fixed noisy measurements", xlabel="Measurement index", ylabel="Value")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    axes[2].stem(measurement_index, run.noise, basefmt=" ")
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set(title="Gaussian noise epsilon", xlabel="Measurement index", ylabel="Noise")

    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    run = run_toy_simulation(ToyConfig(seed=args.seed))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_dir / "arrays.npz",
        x=run.x,
        support=run.support,
        candidate_dictionary=run.candidate_dictionary,
        selected_action_indices=run.selected_action_indices,
        A=run.sensing_matrix,
        clean_measurements=run.clean_measurements,
        noise=run.noise,
        y=run.observations,
    )
    metadata = {
        "config": asdict(run.config),
        "array_shapes": {
            "x": list(run.x.shape),
            "support": list(run.support.shape),
            "candidate_dictionary": list(run.candidate_dictionary.shape),
            "A": list(run.sensing_matrix.shape),
            "y": list(run.observations.shape),
        },
        "interpretation": {
            "x": "Hidden directly sparse signal.",
            "candidate_dictionary": "All allowed normalized measurement vectors.",
            "A": "Fixed sensing matrix; selected rows from the candidate dictionary.",
            "y": "Noisy observations y = A @ x + noise.",
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    save_plot(run, output_dir / "toy_run.png")

    print(f"Saved deterministic artifacts to: {output_dir}")
    print(f"support={run.support.tolist()}")
    print(f"x shape={run.x.shape}; A shape={run.sensing_matrix.shape}; y shape={run.observations.shape}")


if __name__ == "__main__":
    main()
