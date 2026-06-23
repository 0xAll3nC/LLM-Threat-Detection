"""Evaluate end-to-end SOC triage repeatability across multiple runs."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None

import evaluate_results


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = PROJECT_ROOT / "src" / "main.py"
OUTPUTS_PATH = PROJECT_ROOT / "outputs"
PREDICTIONS_PATH = OUTPUTS_PATH / "prioritized_alerts.csv"
EVALUATION_PATH = OUTPUTS_PATH / "evaluation_results.csv"
FAILURE_SUMMARY_PATH = OUTPUTS_PATH / "failure_analysis_summary.csv"
REPEATABILITY_PATH = OUTPUTS_PATH / "repeatability_summary.csv"
RUNS_PATH = OUTPUTS_PATH / "repeatability_runs"
DEFAULT_RUNS = 3

SUMMARY_COLUMNS = [
    "record_type",
    "run",
    "accuracy_percent",
    "correct_count",
    "llm_selection_errors",
    "retrieval_misses",
    "parse_output_errors",
]


def main() -> int:
    args = parse_args()

    if pd is None:
        print("Missing dependency: pandas. Install it with `python3 -m pip install pandas`.")
        return 1

    try:
        alerts = evaluate_results.read_csv(
            evaluate_results.ALERTS_PATH,
            evaluate_results.REQUIRED_ALERT_COLUMNS,
        )
        prepare_runs_directory(RUNS_PATH)
        run_metrics = run_evaluations(alerts, args.runs)
        summary = build_summary(run_metrics)
        save_summary(summary, REPEATABILITY_PATH)
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        print(f"Repeatability evaluation failed: {exc}")
        return 1

    print_summary(run_metrics)
    print(f"\nSaved repeatability summary to {REPEATABILITY_PATH}")
    print(f"Saved per-run artifacts to {RUNS_PATH}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and evaluate the SOC triage pipeline multiple times."
    )
    parser.add_argument(
        "--runs",
        type=positive_int,
        default=DEFAULT_RUNS,
        help=f"Number of full pipeline runs (default: {DEFAULT_RUNS}).",
    )
    return parser.parse_args()


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--runs must be a positive integer.") from exc

    if parsed < 1:
        raise argparse.ArgumentTypeError("--runs must be a positive integer.")

    return parsed


def prepare_runs_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

    for child in path.iterdir():
        if child.is_dir() and child.name.startswith("run_"):
            shutil.rmtree(child)


def run_evaluations(alerts, run_count: int) -> List[Dict[str, Any]]:
    metrics = []

    for run_number in range(1, run_count + 1):
        print(f"Starting run {run_number}/{run_count}...")
        run_path = RUNS_PATH / f"run_{run_number:02d}"
        run_path.mkdir(parents=True, exist_ok=True)

        process = run_pipeline()
        save_process_log(process, run_path)

        if process.returncode != 0:
            error = process.stderr.strip() or process.stdout.strip()
            raise RuntimeError(
                f"Pipeline run {run_number} exited with code "
                f"{process.returncode}: {error}"
            )

        evaluation, failure_summary = evaluate_current_predictions(alerts)
        run_metric = calculate_run_metrics(run_number, evaluation)
        metrics.append(run_metric)
        save_run_artifacts(run_path, evaluation, failure_summary)

        print(
            f"Run {run_number}: accuracy={run_metric['accuracy_percent']:.2f}%, "
            f"correct={run_metric['correct_count']}, "
            f"llm_selection_errors={run_metric['llm_selection_errors']}, "
            f"retrieval_misses={run_metric['retrieval_misses']}, "
            f"parse_output_errors={run_metric['parse_output_errors']}"
        )

    return metrics


def run_pipeline() -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(MAIN_PATH)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def evaluate_current_predictions(alerts):
    predictions = evaluate_results.read_csv(
        PREDICTIONS_PATH,
        evaluate_results.REQUIRED_PREDICTION_COLUMNS,
    )
    evaluation = evaluate_results.evaluate(alerts, predictions)
    failure_summary = evaluate_results.summarize_failures(evaluation)

    evaluate_results.save_evaluation(evaluation, EVALUATION_PATH)
    evaluate_results.save_evaluation(failure_summary, FAILURE_SUMMARY_PATH)
    return evaluation, failure_summary


def calculate_run_metrics(run_number: int, evaluation) -> Dict[str, Any]:
    total = len(evaluation)
    correct = int(evaluation["is_correct"].sum()) if total else 0

    return {
        "record_type": "run",
        "run": run_number,
        "accuracy_percent": (correct / total * 100) if total else 0.0,
        "correct_count": correct,
        "llm_selection_errors": evaluate_results.failure_count(
            evaluation,
            "llm_selection_error",
        ),
        "retrieval_misses": evaluate_results.failure_count(
            evaluation,
            "retrieval_miss",
        ),
        "parse_output_errors": evaluate_results.failure_count(
            evaluation,
            "parse_or_output_error",
        ),
    }


def save_process_log(
    process: subprocess.CompletedProcess[str],
    run_path: Path,
) -> None:
    (run_path / "pipeline_stdout.txt").write_text(
        process.stdout,
        encoding="utf-8",
    )
    (run_path / "pipeline_stderr.txt").write_text(
        process.stderr,
        encoding="utf-8",
    )


def save_run_artifacts(run_path: Path, evaluation, failure_summary) -> None:
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Pipeline output was not created: {PREDICTIONS_PATH}"
        )

    shutil.copy2(PREDICTIONS_PATH, run_path / "prioritized_alerts.csv")
    evaluation.to_csv(run_path / "evaluation_results.csv", index=False)
    failure_summary.to_csv(run_path / "failure_analysis_summary.csv", index=False)


def build_summary(run_metrics: List[Dict[str, Any]]):
    runs = pd.DataFrame(run_metrics, columns=SUMMARY_COLUMNS)
    metric_columns = SUMMARY_COLUMNS[2:]
    aggregate_rows = []

    for label, operation in (
        ("average", "mean"),
        ("minimum", "min"),
        ("maximum", "max"),
    ):
        aggregate = {
            "record_type": "aggregate",
            "run": label,
        }
        values = getattr(runs[metric_columns], operation)()
        aggregate.update(values.to_dict())
        aggregate_rows.append(aggregate)

    return pd.concat(
        [runs, pd.DataFrame(aggregate_rows, columns=SUMMARY_COLUMNS)],
        ignore_index=True,
    )


def save_summary(summary, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)


def print_summary(run_metrics: List[Dict[str, Any]]) -> None:
    runs = pd.DataFrame(run_metrics)
    print("\nRepeatability summary")
    print(
        "Run,Accuracy,Correct,LLM Selection Errors,Retrieval Misses,"
        "Parse/Output Errors"
    )

    for metric in run_metrics:
        print(
            f"{metric['run']},"
            f"{metric['accuracy_percent']:.2f}%,"
            f"{metric['correct_count']},"
            f"{metric['llm_selection_errors']},"
            f"{metric['retrieval_misses']},"
            f"{metric['parse_output_errors']}"
        )

    print(
        f"Average,{runs['accuracy_percent'].mean():.2f}%,"
        f"{runs['correct_count'].mean():.2f},"
        f"{runs['llm_selection_errors'].mean():.2f},"
        f"{runs['retrieval_misses'].mean():.2f},"
        f"{runs['parse_output_errors'].mean():.2f}"
    )
    print(f"Minimum accuracy: {runs['accuracy_percent'].min():.2f}%")
    print(f"Maximum accuracy: {runs['accuracy_percent'].max():.2f}%")


if __name__ == "__main__":
    raise SystemExit(main())
