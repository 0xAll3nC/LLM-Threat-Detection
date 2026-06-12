"""Evaluate predicted MITRE ATT&CK labels against expected alert labels."""

from __future__ import annotations

import re
from pathlib import Path

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALERTS_PATH = PROJECT_ROOT / "data" / "alerts.csv"
PREDICTIONS_PATH = PROJECT_ROOT / "outputs" / "prioritized_alerts.csv"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "evaluation_results.csv"
TECHNIQUE_ID_PATTERN = re.compile(r"\bT\d{4}(?:\.\d{3})?\b", re.IGNORECASE)

REQUIRED_ALERT_COLUMNS = {
    "alert_id",
    "alert_name",
    "expected_technique_id",
    "expected_technique_name",
}
REQUIRED_PREDICTION_COLUMNS = {
    "alert_id",
    "mitre_attack_technique",
    "priority",
    "confidence",
}
OUTPUT_COLUMNS = [
    "alert_id",
    "alert_name",
    "expected_technique_id",
    "expected_technique_name",
    "predicted_technique_id",
    "predicted_raw_value",
    "is_correct",
    "priority",
    "confidence",
]


def main() -> int:
    if pd is None:
        print("Missing dependency: pandas. Install it with `python3 -m pip install pandas`.")
        return 1

    try:
        alerts = read_csv(ALERTS_PATH, REQUIRED_ALERT_COLUMNS)
        predictions = read_csv(PREDICTIONS_PATH, REQUIRED_PREDICTION_COLUMNS)
        evaluation = evaluate(alerts, predictions)
        save_evaluation(evaluation, OUTPUT_PATH)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(exc)
        return 1

    total = len(evaluation)
    correct = int(evaluation["is_correct"].sum()) if total else 0
    accuracy = (correct / total * 100) if total else 0.0

    print(f"Total alerts evaluated: {total}")
    print(f"Correct count: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Saved evaluation results to {OUTPUT_PATH}")
    return 0


def read_csv(path: Path, required_columns: set[str]):
    try:
        data = pd.read_csv(path, dtype={"alert_id": str})
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Required file not found: {path}") from exc
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"CSV file is empty: {path}") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"Malformed CSV file: {path}: {exc}") from exc

    missing_columns = sorted(required_columns - set(data.columns))
    if missing_columns:
        raise ValueError(f"{path} is missing required columns: {missing_columns}")

    return data


def evaluate(alerts, predictions):
    expected = alerts[
        [
            "alert_id",
            "alert_name",
            "expected_technique_id",
            "expected_technique_name",
        ]
    ].copy()
    predicted = predictions[
        [
            "alert_id",
            "mitre_attack_technique",
            "priority",
            "confidence",
        ]
    ].copy()

    merged = expected.merge(predicted, on="alert_id", how="inner")
    if merged.empty:
        raise ValueError("No matching alert_id values found between expected and predicted files.")

    merged["predicted_raw_value"] = merged["mitre_attack_technique"].fillna("")
    merged["predicted_technique_id"] = merged["predicted_raw_value"].apply(
        extract_technique_id
    )
    merged["is_correct"] = (
        merged["predicted_technique_id"].str.upper()
        == merged["expected_technique_id"].fillna("").str.upper()
    )

    return merged[OUTPUT_COLUMNS]


def extract_technique_id(raw_value: object) -> str:
    if raw_value is None:
        return ""

    value = str(raw_value).strip()
    if not value or value.lower() in {"nan", "none", "unknown"}:
        return ""

    match = TECHNIQUE_ID_PATTERN.search(value)
    return match.group(0).upper() if match else ""


def save_evaluation(evaluation, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation.to_csv(output_path, index=False)


if __name__ == "__main__":
    raise SystemExit(main())
