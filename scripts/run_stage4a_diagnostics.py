"""Run Stage 4A gamma ablations and save diagnostic artifacts."""

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

from adaptive_signal_acquisition.reconstruction import LassoConfig
from adaptive_signal_acquisition.signals import SignalConfig
from adaptive_signal_acquisition.stage4 import (
    Stage4AConfig,
    Stage4AResult,
    gamma_method_name,
    run_stage4a_diagnostics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 4A adaptive-v1 diagnostics.")
    parser.add_argument("--seed", type=int, default=20_260_919)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--budgets", type=int, nargs="+", default=[8, 12, 16, 20, 24, 32])
    parser.add_argument("--gamma-values", type=float, nargs="+", default=[0.0, 0.1, 0.5, 1.0])
    parser.add_argument("--lambda-value", type=float, default=0.08)
    parser.add_argument("--bootstrap-resamples", type=int, default=2_000)
    parser.add_argument(
        "--output-dir", type=Path, default=REPOSITORY_ROOT / "artifacts" / "stage4a_diagnostics"
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_nmse_plot(result: Stage4AResult, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    methods = ("fixed", "random", *(gamma_method_name(g) for g in result.config.gamma_values))
    for method in methods:
        rows = [summary for summary in result.summaries if summary.method == method]
        axis.plot(
            [row.budget for row in rows],
            [row.mean_nmse for row in rows],
            marker="o",
            label=method,
        )
    axis.set(
        title="Stage 4A: adaptive-v1 gamma ablation",
        xlabel="Measurement budget B",
        ylabel="Mean NMSE (lower is better)",
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_paired_difference_plot(result: Stage4AResult, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    for gamma in result.config.gamma_values:
        method = gamma_method_name(gamma)
        rows = [summary for summary in result.summaries if summary.method == method]
        budgets = np.asarray([row.budget for row in rows])
        means = np.asarray([row.mean_nmse_difference_vs_random for row in rows])
        lower = np.asarray([row.difference_ci_lower_95 for row in rows])
        upper = np.asarray([row.difference_ci_upper_95 for row in rows])
        axis.errorbar(
            budgets,
            means,
            yerr=np.vstack((means - lower, upper - means)),
            marker="o",
            capsize=3,
            label=f"gamma={gamma:g}",
        )
    axis.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    axis.set(
        title="Paired NMSE difference versus random sensing (95% bootstrap CI)",
        xlabel="Measurement budget B",
        ylabel="Adaptive NMSE - random NMSE (negative is better)",
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_trace_diagnostics(result: Stage4AResult, output_path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    fields = (
        ("lasso_component_fraction", "LASSO fraction of raw score component"),
        ("max_previous_action_correlation", "Max correlation with earlier action"),
        ("sensing_matrix_condition_number", "Sensing-matrix condition number"),
        ("provisional_nmse", "Provisional LASSO NMSE"),
    )
    for gamma in result.config.gamma_values:
        method = gamma_method_name(gamma)
        method_rows = [trace for trace in result.diagnostic_traces if trace.method == method]
        steps = sorted({trace.step for trace in method_rows})
        for axis, (field, ylabel) in zip(axes.flat, fields, strict=True):
            means = [
                float(np.mean([getattr(trace, field) for trace in method_rows if trace.step == step]))
                for step in steps
            ]
            axis.plot(steps, means, label=f"gamma={gamma:g}")
            axis.set(xlabel="Sequential measurement step", ylabel=ylabel)
            axis.grid(alpha=0.25)
    for axis in axes.flat:
        axis.legend(fontsize=8)
    figure.suptitle("Why adaptive-v1 selects its actions", fontsize=14)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    final_lasso = LassoConfig(lambda_value=args.lambda_value)
    provisional_lasso = LassoConfig(lambda_value=args.lambda_value, max_iterations=1_000)
    config = Stage4AConfig(
        signal=SignalConfig(),
        budgets=tuple(args.budgets),
        trials_per_budget=args.trials,
        lasso=final_lasso,
        provisional_lasso=provisional_lasso,
        gamma_values=tuple(args.gamma_values),
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    result = run_stage4a_diagnostics(config)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(output_dir / "trial_records.csv", [asdict(row) for row in result.records])
    write_csv(output_dir / "paired_summary.csv", [asdict(row) for row in result.summaries])
    write_csv(
        output_dir / "diagnostic_traces.csv",
        [asdict(row) for row in result.diagnostic_traces],
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(asdict(result.config), indent=2, default=str), encoding="utf-8"
    )
    save_nmse_plot(result, output_dir / "nmse_gamma_ablation.png")
    save_paired_difference_plot(result, output_dir / "paired_difference_vs_random.png")
    save_trace_diagnostics(result, output_dir / "trace_diagnostics.png")

    print(f"Saved Stage 4A artifacts to: {output_dir}")
    for summary in result.summaries:
        print(
            f"{summary.method:23} B={summary.budget:2d} "
            f"NMSE={summary.mean_nmse:.4f} "
            f"delta-vs-random={summary.mean_nmse_difference_vs_random:+.4f} "
            f"CI=[{summary.difference_ci_lower_95:+.4f}, "
            f"{summary.difference_ci_upper_95:+.4f}]"
        )


if __name__ == "__main__":
    main()
