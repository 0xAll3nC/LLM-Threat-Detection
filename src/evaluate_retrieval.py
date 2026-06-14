"""Evaluate ATT&CK retrieval quality without calling an LLM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None

from attack_retriever import retrieve_attack_techniques


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALERTS_PATH = PROJECT_ROOT / "data" / "alerts.csv"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "retrieval_evaluation_results.csv"

REQUIRED_COLUMNS = {
    "alert_id",
    "alert_name",
    "expected_technique_id",
    "expected_technique_name",
}
OUTPUT_COLUMNS = [
    "alert_id",
    "alert_name",
    "expected_technique_id",
    "expected_technique_name",
    "retrieved_rank",
    "top_1",
    "top_3",
    "top_5",
    "top_10",
    "retrieved_techniques",
]


def main() -> int:
    if pd is None:
        print("Missing dependency: pandas. Install it with `python3 -m pip install pandas`.")
        return 1

    try:
        alerts = read_alerts(ALERTS_PATH)
        evaluation_rows = evaluate_retrieval(alerts)
        save_results(evaluation_rows, OUTPUT_PATH)
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        print(exc)
        return 1

    metrics = calculate_metrics(evaluation_rows)
    print_summary(metrics)
    print(f"Saved retrieval evaluation results to {OUTPUT_PATH}")
    return 0


def read_alerts(path: Path):
    try:
        alerts = pd.read_csv(path, dtype={"alert_id": str})
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Alerts file not found: {path}") from exc
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Alerts file is empty: {path}") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"Malformed alerts CSV: {exc}") from exc

    missing_columns = sorted(REQUIRED_COLUMNS - set(alerts.columns))
    if missing_columns:
        raise ValueError(f"Alerts CSV is missing required columns: {missing_columns}")

    missing_labels = alerts[
        alerts["expected_technique_id"].isna()
        | alerts["expected_technique_name"].isna()
        | (alerts["expected_technique_id"].astype(str).str.strip() == "")
        | (alerts["expected_technique_name"].astype(str).str.strip() == "")
    ]
    if not missing_labels.empty:
        alert_ids = ", ".join(missing_labels["alert_id"].astype(str).tolist())
        raise ValueError(f"Missing expected MITRE labels for alert_id values: {alert_ids}")

    return alerts


def evaluate_retrieval(alerts) -> List[Dict[str, Any]]:
    evaluation_rows = []

    for _, row in alerts.iterrows():
        alert = row.where(pd.notna(row), None).to_dict()
        expected_id = str(alert["expected_technique_id"]).strip().upper()
        retrieved = retrieve_attack_techniques(alert, top_k=10)
        retrieved_ids = [
            str(candidate.get("technique_id", "")).strip().upper()
            for candidate in retrieved
        ]
        retrieved_rank = find_rank(expected_id, retrieved_ids)

        evaluation_rows.append(
            {
                "alert_id": alert.get("alert_id", ""),
                "alert_name": alert.get("alert_name", ""),
                "expected_technique_id": expected_id,
                "expected_technique_name": alert.get("expected_technique_name", ""),
                "retrieved_rank": retrieved_rank if retrieved_rank is not None else "",
                "top_1": is_hit_at_k(retrieved_rank, 1),
                "top_3": is_hit_at_k(retrieved_rank, 3),
                "top_5": is_hit_at_k(retrieved_rank, 5),
                "top_10": is_hit_at_k(retrieved_rank, 10),
                "retrieved_techniques": format_retrieved_techniques(retrieved),
            }
        )

    return evaluation_rows


def find_rank(expected_id: str, retrieved_ids: List[str]) -> Optional[int]:
    for index, retrieved_id in enumerate(retrieved_ids, start=1):
        if retrieved_id == expected_id:
            return index
    return None


def is_hit_at_k(rank: Optional[int], k: int) -> bool:
    return rank is not None and rank <= k


def format_retrieved_techniques(retrieved: List[Dict[str, Any]]) -> str:
    compact = [
        {
            "rank": index,
            "technique_id": candidate.get("technique_id", ""),
            "name": candidate.get("name", ""),
            "similarity_score": candidate.get("similarity_score", 0),
        }
        for index, candidate in enumerate(retrieved, start=1)
    ]
    return json.dumps(compact)


def calculate_metrics(evaluation_rows: List[Dict[str, Any]]) -> Dict[str, float]:
    total_alerts = len(evaluation_rows)
    if total_alerts == 0:
        return {
            "total_alerts": 0,
            "hit_rate_at_1": 0.0,
            "hit_rate_at_3": 0.0,
            "hit_rate_at_5": 0.0,
            "hit_rate_at_10": 0.0,
            "mean_reciprocal_rank": 0.0,
        }

    reciprocal_ranks = [
        1 / int(row["retrieved_rank"])
        for row in evaluation_rows
        if row["retrieved_rank"] != ""
    ]

    return {
        "total_alerts": total_alerts,
        "hit_rate_at_1": mean_bool(row["top_1"] for row in evaluation_rows),
        "hit_rate_at_3": mean_bool(row["top_3"] for row in evaluation_rows),
        "hit_rate_at_5": mean_bool(row["top_5"] for row in evaluation_rows),
        "hit_rate_at_10": mean_bool(row["top_10"] for row in evaluation_rows),
        "mean_reciprocal_rank": sum(reciprocal_ranks) / total_alerts,
    }


def mean_bool(values) -> float:
    values = list(values)
    return sum(1 for value in values if value) / len(values) if values else 0.0


def save_results(evaluation_rows: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(evaluation_rows, columns=OUTPUT_COLUMNS).to_csv(output_path, index=False)


def print_summary(metrics: Dict[str, float]) -> None:
    print("Retrieval evaluation summary")
    print(f"Total alerts: {int(metrics['total_alerts'])}")
    print(f"Hit rate @ 1: {metrics['hit_rate_at_1']:.2%}")
    print(f"Hit rate @ 3: {metrics['hit_rate_at_3']:.2%}")
    print(f"Hit rate @ 5: {metrics['hit_rate_at_5']:.2%}")
    print(f"Hit rate @ 10: {metrics['hit_rate_at_10']:.2%}")
    print(f"Mean reciprocal rank: {metrics['mean_reciprocal_rank']:.4f}")


if __name__ == "__main__":
    raise SystemExit(main())
