"""Generate a concise review of remaining ATT&CK mapping failures."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, List

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_PATH = PROJECT_ROOT / "outputs" / "evaluation_results.csv"
PREDICTIONS_PATH = PROJECT_ROOT / "outputs" / "prioritized_alerts.csv"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "remaining_failure_review.csv"

REQUIRED_EVALUATION_COLUMNS = {
    "alert_id",
    "alert_name",
    "expected_technique_id",
    "expected_technique_name",
    "predicted_technique_id",
    "predicted_technique_name",
    "is_correct",
    "failure_type",
    "expected_in_retrieved_candidates",
    "expected_candidate_rank",
}
REQUIRED_PREDICTION_COLUMNS = {
    "alert_id",
    "retrieved_attack_candidates",
}
OUTPUT_COLUMNS = [
    "alert_id",
    "alert_name",
    "expected_technique_id",
    "expected_technique_name",
    "predicted_technique_id",
    "predicted_technique_name",
    "failure_type",
    "expected_in_retrieved_candidates",
    "expected_candidate_rank",
    "top_retrieved_candidates",
]
FAILURE_GROUPS = (
    ("llm_selection_error", "LLM selection errors"),
    ("retrieval_miss", "Retrieval misses"),
    ("parse_or_output_error", "Parse/output errors"),
)


def main() -> int:
    if pd is None:
        print("Missing dependency: pandas. Install it with `python3 -m pip install pandas`.")
        return 1

    try:
        evaluation = read_csv(EVALUATION_PATH, REQUIRED_EVALUATION_COLUMNS)
        predictions = read_csv(PREDICTIONS_PATH, REQUIRED_PREDICTION_COLUMNS)
        review = build_failure_review(evaluation, predictions)
        save_review(review, OUTPUT_PATH)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(exc)
        return 1

    print_failure_summary(review)
    print(f"\nSaved remaining failure review to {OUTPUT_PATH}")
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


def build_failure_review(evaluation, predictions):
    failures = evaluation.loc[~evaluation["is_correct"].apply(_as_bool)].copy()
    candidates = predictions[["alert_id", "retrieved_attack_candidates"]].copy()
    merged = failures.merge(candidates, on="alert_id", how="left", validate="one_to_one")

    merged["top_retrieved_candidates"] = merged[
        "retrieved_attack_candidates"
    ].apply(format_candidates)

    merged["_alert_sort_key"] = pd.to_numeric(merged["alert_id"], errors="coerce")
    merged["_alert_sort_text"] = merged["alert_id"].fillna("")

    return merged.sort_values(
        by=["failure_type", "_alert_sort_key", "_alert_sort_text"],
        kind="stable",
        na_position="last",
    )[OUTPUT_COLUMNS]


def format_candidates(raw_candidates: object) -> str:
    candidates = parse_candidates(raw_candidates)
    formatted = []

    for rank, candidate in enumerate(candidates, start=1):
        technique_id = _stringify(candidate.get("technique_id"))
        name = _stringify(candidate.get("name"))

        if technique_id and name:
            label = f"{technique_id} - {name}"
        else:
            label = technique_id or name or "unparseable candidate"

        formatted.append(f"{rank}. {label}")

    return " | ".join(formatted)


def parse_candidates(raw_candidates: object) -> List[Dict[str, Any]]:
    if raw_candidates is None:
        return []

    value = str(raw_candidates).strip()
    if not value or value.lower() in {"nan", "none", "unknown"}:
        return []

    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return []

    if not isinstance(parsed, list):
        return []

    return [candidate for candidate in parsed if isinstance(candidate, dict)]


def print_failure_summary(review) -> None:
    print(f"Remaining incorrect alerts: {len(review)}")

    for failure_type, heading in FAILURE_GROUPS:
        group = review.loc[review["failure_type"] == failure_type]
        print(f"\n{heading} ({len(group)}):")

        if group.empty:
            print("  None")
            continue

        for _, row in group.iterrows():
            expected_rank = _format_rank(row["expected_candidate_rank"])
            print(
                f"  Alert {row['alert_id']}: {row['alert_name']}\n"
                f"    Expected: {row['expected_technique_id']} - "
                f"{row['expected_technique_name']}\n"
                f"    Predicted: {row['predicted_technique_id']} - "
                f"{row['predicted_technique_name']}\n"
                f"    Expected retrieved: "
                f"{_as_bool(row['expected_in_retrieved_candidates'])}; "
                f"rank: {expected_rank}\n"
                f"    Candidates: {row['top_retrieved_candidates']}"
            )


def save_review(review, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    review.to_csv(output_path, index=False)


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes"}


def _format_rank(value: object) -> str:
    if value is None or pd.isna(value):
        return "not retrieved"

    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value)


def _stringify(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


if __name__ == "__main__":
    raise SystemExit(main())
