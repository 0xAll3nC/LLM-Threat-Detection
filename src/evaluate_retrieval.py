"""Compare BM25 and dense ATT&CK retrieval without calling an LLM."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None

from attack_retriever import (
    rerank_attack_candidates,
    retrieve_attack_techniques,
    retrieve_attack_techniques_dense,
    retrieve_attack_techniques_hybrid,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALERTS_PATH = PROJECT_ROOT / "data" / "alerts.csv"
COMPARISON_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "retrieval_comparison_with_rerank.csv"
EVALUATION_TOP_K = 10

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
    "bm25_rank",
    "dense_rank",
    "hybrid_rank",
    "hybrid_rerank_rank",
]


def main() -> int:
    if pd is None:
        print("Missing dependency: pandas. Install it with `python3 -m pip install pandas`.")
        return 1

    dense_error = None

    try:
        alerts = read_alerts(ALERTS_PATH)
        evaluation_rows, dense_error = evaluate_retrieval(alerts)
        save_results(evaluation_rows, COMPARISON_OUTPUT_PATH)
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        print(exc)
        return 1

    metrics_by_method = {
        "BM25": calculate_metrics(evaluation_rows, "bm25_rank"),
        "Dense": calculate_metrics(evaluation_rows, "dense_rank"),
        "Hybrid": calculate_metrics(evaluation_rows, "hybrid_rank"),
        "Hybrid+Rerank": calculate_metrics(evaluation_rows, "hybrid_rerank_rank"),
    }
    print_metrics_table(metrics_by_method)
    if dense_error:
        print(f"Dense retrieval warning: {dense_error}")
    print(f"Hybrid+Rerank reranks the top {EVALUATION_TOP_K} Hybrid candidates.")
    print(f"Saved retrieval comparison results to {COMPARISON_OUTPUT_PATH}")
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


def evaluate_retrieval(alerts) -> tuple[List[Dict[str, Any]], Optional[str]]:
    evaluation_rows = []
    dense_error = None

    for _, row in alerts.iterrows():
        alert = row.where(pd.notna(row), None).to_dict()
        expected_id = str(alert["expected_technique_id"]).strip().upper()

        bm25_results = retrieve_attack_techniques(alert, top_k=EVALUATION_TOP_K)
        bm25_rank = find_rank(expected_id, technique_ids(bm25_results))

        dense_rank = None
        hybrid_rank = None
        hybrid_rerank_rank = None
        if dense_error is None:
            try:
                dense_results = retrieve_attack_techniques_dense(
                    alert,
                    top_k=EVALUATION_TOP_K,
                )
                dense_rank = find_rank(expected_id, technique_ids(dense_results))

                hybrid_results = retrieve_attack_techniques_hybrid(
                    alert,
                    top_k=EVALUATION_TOP_K,
                )
                hybrid_rank = find_rank(expected_id, technique_ids(hybrid_results))

                hybrid_rerank_results = rerank_attack_candidates(
                    alert,
                    hybrid_results,
                    top_k=EVALUATION_TOP_K,
                )
                hybrid_rerank_rank = find_rank(
                    expected_id,
                    technique_ids(hybrid_rerank_results),
                )
            except Exception as exc:
                dense_error = str(exc)

        evaluation_rows.append(
            {
                "alert_id": alert.get("alert_id", ""),
                "alert_name": alert.get("alert_name", ""),
                "expected_technique_id": expected_id,
                "bm25_rank": bm25_rank if bm25_rank is not None else "",
                "dense_rank": dense_rank if dense_rank is not None else "",
                "hybrid_rank": hybrid_rank if hybrid_rank is not None else "",
                "hybrid_rerank_rank": (
                    hybrid_rerank_rank if hybrid_rerank_rank is not None else ""
                ),
            }
        )

    return evaluation_rows, dense_error


def technique_ids(results: List[Dict[str, Any]]) -> List[str]:
    return [
        str(candidate.get("technique_id", "")).strip().upper()
        for candidate in results
    ]


def find_rank(expected_id: str, retrieved_ids: List[str]) -> Optional[int]:
    for index, retrieved_id in enumerate(retrieved_ids, start=1):
        if retrieved_id == expected_id:
            return index
    return None


def is_hit_at_k(rank: Optional[int], k: int) -> bool:
    return rank is not None and rank <= k


def calculate_metrics(
    evaluation_rows: List[Dict[str, Any]],
    rank_column: str,
) -> Dict[str, float]:
    total_alerts = len(evaluation_rows)
    if total_alerts == 0:
        return empty_metrics()

    ranks = [row[rank_column] for row in evaluation_rows]
    numeric_ranks = [int(rank) for rank in ranks if rank != ""]

    return {
        "total_alerts": total_alerts,
        "hit_rate_at_1": mean_bool(is_hit_at_k(rank_or_none(rank), 1) for rank in ranks),
        "hit_rate_at_3": mean_bool(is_hit_at_k(rank_or_none(rank), 3) for rank in ranks),
        "hit_rate_at_5": mean_bool(is_hit_at_k(rank_or_none(rank), 5) for rank in ranks),
        "hit_rate_at_10": mean_bool(is_hit_at_k(rank_or_none(rank), 10) for rank in ranks),
        "mean_reciprocal_rank": sum(1 / rank for rank in numeric_ranks) / total_alerts,
    }


def empty_metrics() -> Dict[str, float]:
    return {
        "total_alerts": 0,
        "hit_rate_at_1": 0.0,
        "hit_rate_at_3": 0.0,
        "hit_rate_at_5": 0.0,
        "hit_rate_at_10": 0.0,
        "mean_reciprocal_rank": 0.0,
    }


def rank_or_none(rank: object) -> Optional[int]:
    if rank == "":
        return None
    return int(rank)


def mean_bool(values) -> float:
    values = list(values)
    return sum(1 for value in values if value) / len(values) if values else 0.0


def save_results(evaluation_rows: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(evaluation_rows, columns=OUTPUT_COLUMNS).to_csv(output_path, index=False)


def print_metrics_table(metrics_by_method: Dict[str, Dict[str, float]]) -> None:
    print("Retriever,Hit@1,Hit@3,Hit@5,Hit@10,MRR")
    for retriever, metrics in metrics_by_method.items():
        print(
            f"{retriever},"
            f"{metrics['hit_rate_at_1']:.2f},"
            f"{metrics['hit_rate_at_3']:.2f},"
            f"{metrics['hit_rate_at_5']:.2f},"
            f"{metrics['hit_rate_at_10']:.2f},"
            f"{metrics['mean_reciprocal_rank']:.4f}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
