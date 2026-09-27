"""Run Stage 4B and save paired results, diagnostics, and figures."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPOSITORY_ROOT / "artifacts" / ".matplotlib"))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from adaptive_signal_acquisition.adaptive_v2 import BootstrapPolicyConfig
from adaptive_signal_acquisition.reconstruction import LassoConfig
from adaptive_signal_acquisition.signals import SignalConfig
from adaptive_signal_acquisition.stage4b import (
    METHOD_ORDER,
    Stage4BConfig,
    Stage4BResult,
    run_stage4b_study,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Stage 4B adaptive-v2 study.")
    parser.add_argument("--seed", type=int, default=20_260_927)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--budgets", type=int, nargs="+", default=[8, 12, 16, 20, 24, 32])
    parser.add_argument("--burn-in", type=int, default=8)
    parser.add_argument("--ensemble-size", type=int, default=12)
    parser.add_argument("--gaussian-weight", type=float, default=0.25)
    parser.add_argument("--perturbation-scale", type=float, default=1.0)
    parser.add_argument("--lambda-value", type=float, default=0.08)
    parser.add_argument("--ensemble-iterations", type=int, default=500)
    parser.add_argument("--bootstrap-resamples", type=int, default=2_000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts" / "stage4b_adaptive_v2",
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_nmse_plot(result: Stage4BResult, path: Path) -> None:
    labels = {
        "fixed": "fixed",
        "random": "random",
        "gaussian_only": "Gaussian-only",
        "adaptive_v1_gamma_1": "adaptive_v1",
        "adaptive_v2_bootstrap": "adaptive_v2",
    }
    figure, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    for method in METHOD_ORDER:
        rows = [row for row in result.summaries if row.method == method]
        axis.plot(
            [row.budget for row in rows],
            [row.mean_nmse for row in rows],
            marker="o",
            label=labels[method],
        )
    axis.set(
        title="Stage 4B: bootstrap uncertainty versus reference policies",
        xlabel="Measurement budget B",
        ylabel="Mean NMSE (lower is better)",
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_paired_plot(result: Stage4BResult, path: Path) -> None:
    methods = ("gaussian_only", "adaptive_v1_gamma_1", "adaptive_v2_bootstrap")
    figure, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    for method in methods:
        rows = [row for row in result.summaries if row.method == method]
        means = np.asarray([row.mean_nmse_difference_vs_random for row in rows])
        lower = np.asarray([row.difference_ci_lower_95 for row in rows])
        upper = np.asarray([row.difference_ci_upper_95 for row in rows])
        axis.errorbar(
            [row.budget for row in rows],
            means,
            yerr=np.vstack((means - lower, upper - means)),
            marker="o",
            capsize=3,
            label=method,
        )
    axis.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    axis.set(
        title="Paired NMSE difference from random sensing (95% bootstrap CI)",
        xlabel="Measurement budget B",
        ylabel="Method NMSE - random NMSE (negative is better)",
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_diagnostics(result: Stage4BResult, path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    fields = (
        ("bootstrap_component_fraction", "Bootstrap fraction of score variance"),
        ("bootstrap_covariance_rank", "Bootstrap covariance rank"),
        ("max_previous_action_correlation", "Max row correlation"),
        ("provisional_nmse", "Provisional estimate NMSE"),
    )
    steps = sorted({row.step for row in result.diagnostic_traces})
    for axis, (field, label) in zip(axes.flat, fields, strict=True):
        values = [
            float(np.mean([getattr(row, field) for row in result.diagnostic_traces if row.step == step]))
            for step in steps
        ]
        axis.plot(steps, values, color="black")
        axis.axvline(result.config.adaptive_v2.burn_in_steps + 0.5, color="0.5", linestyle="--")
        axis.set(xlabel="Sequential step", ylabel=label)
        axis.grid(alpha=0.25)
    figure.suptitle("Stage 4B adaptive_v2 decision diagnostics")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    final_lasso = LassoConfig(lambda_value=args.lambda_value)
    ensemble_lasso = LassoConfig(
        lambda_value=args.lambda_value,
        max_iterations=args.ensemble_iterations,
        tolerance=1e-5,
    )
    config = Stage4BConfig(
        signal=SignalConfig(),
        budgets=tuple(args.budgets),
        trials_per_budget=args.trials,
        final_lasso=final_lasso,
        adaptive_v1_lasso=final_lasso,
        adaptive_v2=BootstrapPolicyConfig(
            burn_in_steps=args.burn_in,
            ensemble_size=args.ensemble_size,
            perturbation_scale=args.perturbation_scale,
            gaussian_weight=args.gaussian_weight,
            ensemble_lasso=ensemble_lasso,
        ),
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    result = run_stage4b_study(config)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "trial_records.csv", [asdict(row) for row in result.records])
    write_csv(output_dir / "paired_summary.csv", [asdict(row) for row in result.summaries])
    write_csv(output_dir / "diagnostic_traces.csv", [asdict(row) for row in result.diagnostic_traces])
    (output_dir / "metadata.json").write_text(
        json.dumps(asdict(config), indent=2, default=str), encoding="utf-8"
    )
    save_nmse_plot(result, output_dir / "nmse_comparison.png")
    save_paired_plot(result, output_dir / "paired_difference_vs_random.png")
    save_diagnostics(result, output_dir / "adaptive_v2_diagnostics.png")

    print(f"Saved Stage 4B artifacts to: {output_dir}")
    for row in result.summaries:
        print(
            f"{row.method:23} B={row.budget:2d} NMSE={row.mean_nmse:.4f} "
            f"delta={row.mean_nmse_difference_vs_random:+.4f} "
            f"CI=[{row.difference_ci_lower_95:+.4f}, {row.difference_ci_upper_95:+.4f}]"
        )


if __name__ == "__main__":
    main()
