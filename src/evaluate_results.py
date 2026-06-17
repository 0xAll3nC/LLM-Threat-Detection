"""Evaluate predicted MITRE ATT&CK labels against expected alert labels."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALERTS_PATH = PROJECT_ROOT / "data" / "alerts.csv"
PREDICTIONS_PATH = PROJECT_ROOT / "outputs" / "prioritized_alerts.csv"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "evaluation_results.csv"
SUMMARY_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "failure_analysis_summary.csv"
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
    "retrieved_attack_candidates",
}
OUTPUT_COLUMNS = [
    "alert_id",
    "alert_name",
    "expected_technique_id",
    "expected_technique_name",
    "predicted_technique_id",
    "predicted_technique_name",
    "predicted_raw_value",
    "is_correct",
    "expected_in_retrieved_candidates",
    "expected_candidate_rank",
    "failure_type",
    "priority",
    "confidence",
]
FAILURE_TYPES = (
    "correct",
    "llm_selection_error",
    "retrieval_miss",
    "parse_or_output_error",
)
SUMMARY_COLUMNS = ["failure_type", "count", "percentage"]


def main() -> int:
    if pd is None:
        print("Missing dependency: pandas. Install it with `python3 -m pip install pandas`.")
        return 1

    try:
        alerts = read_csv(ALERTS_PATH, REQUIRED_ALERT_COLUMNS)
        predictions = read_csv(PREDICTIONS_PATH, REQUIRED_PREDICTION_COLUMNS)
        evaluation = evaluate(alerts, predictions)
        save_evaluation(evaluation, OUTPUT_PATH)
        summary = summarize_failures(evaluation)
        save_evaluation(summary, SUMMARY_OUTPUT_PATH)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(exc)
        return 1

    total = len(evaluation)
    correct = int(evaluation["is_correct"].sum()) if total else 0
    accuracy = (correct / total * 100) if total else 0.0

    print(f"Total alerts evaluated: {total}")
    print(f"Correct count: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print()
    print("Failure breakdown:")
    print(f"Correct: {failure_count(evaluation, 'correct')}")
    print(f"LLM selection errors: {failure_count(evaluation, 'llm_selection_error')}")
    print(f"Retrieval misses: {failure_count(evaluation, 'retrieval_miss')}")
    print(f"Parse/output errors: {failure_count(evaluation, 'parse_or_output_error')}")
    print(f"Saved evaluation results to {OUTPUT_PATH}")
    print(f"Saved failure analysis summary to {SUMMARY_OUTPUT_PATH}")
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
            "retrieved_attack_candidates",
        ]
    ].copy()

    merged = expected.merge(predicted, on="alert_id", how="inner")
    if merged.empty:
        raise ValueError("No matching alert_id values found between expected and predicted files.")

    merged["predicted_raw_value"] = merged["mitre_attack_technique"].fillna("")
    prediction_details = merged["predicted_raw_value"].apply(
        extract_prediction_details
    )
    merged["predicted_technique_id"] = prediction_details.apply(lambda item: item[0])
    merged["predicted_technique_name"] = prediction_details.apply(lambda item: item[1])
    candidate_details = merged.apply(
        lambda row: find_expected_candidate(
            row["expected_technique_id"],
            row["retrieved_attack_candidates"],
        ),
        axis=1,
    )
    merged["expected_in_retrieved_candidates"] = candidate_details.apply(
        lambda item: item[0]
    )
    merged["expected_candidate_rank"] = candidate_details.apply(
        lambda item: item[1] if item[1] is not None else ""
    )
    merged["is_correct"] = (
        merged["predicted_technique_id"].str.upper()
        == merged["expected_technique_id"].fillna("").str.upper()
    )
    merged["failure_type"] = merged.apply(classify_failure, axis=1)

    return merged[OUTPUT_COLUMNS]


def extract_technique_id(raw_value: object) -> str:
    technique_id, _ = extract_prediction_details(raw_value)
    return technique_id


def extract_prediction_details(raw_value: object) -> Tuple[str, str]:
    if raw_value is None:
        return "", ""

    value = str(raw_value).strip()
    if not value or value.lower() in {"nan", "none", "unknown"}:
        return "", ""

    literal = parse_literal(value)
    if literal is not None:
        technique_id, technique_name = extract_from_literal(literal)
        if technique_id:
            return technique_id, technique_name

    match = TECHNIQUE_ID_PATTERN.search(value)
    if not match:
        return "", ""

    technique_id = match.group(0).upper()
    technique_name = extract_name_from_text(value, match.end())
    return technique_id, technique_name


def parse_literal(value: str) -> Any:
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return None


def extract_from_literal(value: Any) -> Tuple[str, str]:
    if isinstance(value, dict):
        return extract_from_dict(value)
    if isinstance(value, list):
        for item in value:
            technique_id, technique_name = extract_from_literal(item)
            if technique_id:
                return technique_id, technique_name
    return "", ""


def extract_from_dict(value: Dict[Any, Any]) -> Tuple[str, str]:
    for key in ("technique_id", "id"):
        technique_id = normalize_technique_id(value.get(key))
        if technique_id:
            return technique_id, stringify(value.get("name"))

    for key, item in value.items():
        technique_id = normalize_technique_id(key)
        if technique_id:
            return technique_id, stringify(item)

    for item in value.values():
        technique_id, technique_name = extract_from_literal(item)
        if technique_id:
            return technique_id, technique_name

    return "", ""


def normalize_technique_id(value: object) -> str:
    if value is None:
        return ""

    match = TECHNIQUE_ID_PATTERN.search(str(value))
    return match.group(0).upper() if match else ""


def extract_name_from_text(value: str, start_index: int) -> str:
    remainder = value[start_index:].strip()
    if not remainder:
        return ""

    remainder = remainder.lstrip(" :-")
    return remainder.strip(" '\"{}") if remainder else ""


def find_expected_candidate(
    expected_technique_id: object,
    raw_candidates: object,
) -> Tuple[bool, Optional[int]]:
    expected_id = normalize_technique_id(expected_technique_id)
    if not expected_id:
        return False, None

    candidates = parse_retrieved_candidates(raw_candidates)
    for index, candidate in enumerate(candidates, start=1):
        candidate_id = candidate.get("technique_id", "")
        if normalize_technique_id(candidate_id) == expected_id:
            return True, index

    return False, None


def parse_retrieved_candidates(raw_candidates: object) -> List[Dict[str, Any]]:
    if raw_candidates is None:
        return []

    value = str(raw_candidates).strip()
    if not value or value.lower() in {"nan", "none", "unknown"}:
        return []

    parsed = parse_literal(value)
    if not isinstance(parsed, list):
        return []

    return [candidate for candidate in parsed if isinstance(candidate, dict)]


def classify_failure(row) -> str:
    if bool(row["is_correct"]):
        return "correct"
    if not row["predicted_technique_id"]:
        return "parse_or_output_error"
    if bool(row["expected_in_retrieved_candidates"]):
        return "llm_selection_error"
    return "retrieval_miss"


def summarize_failures(evaluation):
    total = len(evaluation)
    rows = []

    for failure_type in FAILURE_TYPES:
        count = failure_count(evaluation, failure_type)
        percentage = (count / total * 100) if total else 0.0
        rows.append(
            {
                "failure_type": failure_type,
                "count": count,
                "percentage": round(percentage, 2),
            }
        )

    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def failure_count(evaluation, failure_type: str) -> int:
    return int((evaluation["failure_type"] == failure_type).sum())


def save_evaluation(evaluation, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation.to_csv(output_path, index=False)


def stringify(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


if __name__ == "__main__":
    raise SystemExit(main())
