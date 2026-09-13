"""Run the Stage 3 information-guided sensing comparison and save artifacts."""

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

from adaptive_signal_acquisition import run_adaptive_study
from adaptive_signal_acquisition.experiments import AdaptiveStudyConfig, AdaptiveStudyResult
from adaptive_signal_acquisition.reconstruction import LassoConfig
from adaptive_signal_acquisition.sequential import InformationGuidedPolicyConfig
from adaptive_signal_acquisition.signals import SignalConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Stage 3 adaptive sensing study.")
    parser.add_argument("--seed", type=int, default=20_260_913)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--budgets", type=int, nargs="+", default=[8, 12, 16, 20, 24, 32])
    parser.add_argument("--lambda-value", type=float, default=0.08)
    parser.add_argument("--support-proxy-weight", type=float, default=1.0)
    parser.add_argument(
        "--output-dir", type=Path, default=REPOSITORY_ROOT / "artifacts" / "stage3_adaptive"
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_nmse_plot(result: AdaptiveStudyResult, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    styles = (
        ("fixed", "fixed non-adaptive", "tab:blue"),
        ("random", "random non-adaptive", "tab:orange"),
        ("information_guided", "information-guided adaptive", "tab:green"),
    )
    for method, label, color in styles:
        rows = [summary for summary in result.summaries if summary.method == method]
        axis.plot(
            [row.budget for row in rows],
            [row.mean_nmse for row in rows],
            marker="o",
            label=label,
            color=color,
        )
    axis.set(
        title="Stage 3: adaptive versus non-adaptive reconstruction",
        xlabel="Measurement budget B (rows of A)",
        ylabel="Mean NMSE (lower is better)",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    lasso = LassoConfig(lambda_value=args.lambda_value)
    config = AdaptiveStudyConfig(
        signal=SignalConfig(),
        budgets=tuple(args.budgets),
        trials_per_budget=args.trials,
        lasso=lasso,
        policy=InformationGuidedPolicyConfig(
            support_proxy_weight=args.support_proxy_weight,
            provisional_lasso=lasso,
        ),
        seed=args.seed,
    )
    result = run_adaptive_study(config)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(output_dir / "trial_records.csv", [asdict(record) for record in result.records])
    write_csv(output_dir / "adaptive_summary.csv", [asdict(summary) for summary in result.summaries])
    write_csv(output_dir / "action_traces.csv", [asdict(trace) for trace in result.action_traces])
    (output_dir / "metadata.json").write_text(
        json.dumps(asdict(result.config), indent=2, default=str), encoding="utf-8"
    )
    save_nmse_plot(result, output_dir / "nmse_vs_budget.png")

    print(f"Saved Stage 3 artifacts to: {output_dir}")
    for summary in result.summaries:
        print(
            f"{summary.method:19} B={summary.budget:2d} "
            f"mean NMSE={summary.mean_nmse:.4f} "
            f"mean F1={summary.mean_support_f1:.3f} "
            f"converged={summary.convergence_rate:.0%}"
        )


if __name__ == "__main__":
    main()
