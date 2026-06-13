"""Retrieve relevant MITRE ATT&CK techniques for an alert."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List

try:
    from rank_bm25 import BM25Okapi
except ModuleNotFoundError as exc:
    BM25Okapi = None
    BM25_IMPORT_ERROR = exc
else:
    BM25_IMPORT_ERROR = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ATTACK_CACHE_PATH = PROJECT_ROOT / "data" / "attack_techniques.json"
ALERTS_PATH = PROJECT_ROOT / "data" / "alerts.csv"
ALERT_FIELD_WEIGHTS = {
    "alert_name": 3,
    "description": 3,
    "command": 2,
    "process": 2,
    "detection_logic": 1,
}
TECHNIQUE_SEARCH_FIELDS = (
    "technique_id",
    "name",
    "description",
    "tactics",
    "platforms",
)
FIELD_SCORE_WEIGHTS = {
    "name": 3.0,
    "tactics": 2.0,
    "description": 1.0,
    "platforms": 0.5,
}
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:\.[a-z0-9]+)?")


def retrieve_attack_techniques(
    alert: Dict[str, Any],
    top_k: int = 5,
    cache_path: Path = ATTACK_CACHE_PATH,
) -> List[Dict[str, Any]]:
    """Return the top matching ATT&CK techniques for an alert dictionary."""
    if not isinstance(alert, dict):
        raise ValueError("Alert must be a dictionary.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if BM25Okapi is None:
        missing = getattr(BM25_IMPORT_ERROR, "name", "rank-bm25")
        raise RuntimeError(
            f"Missing dependency: {missing}. "
            "Install it with `python3 -m pip install rank-bm25`."
        )

    techniques = load_attack_techniques(cache_path)
    if not techniques:
        return []

    alert_text = alert_to_search_text(alert)
    query_tokens = tokenize_text(alert_text)

    scores = calculate_field_weighted_scores(techniques, query_tokens)

    ranked_indexes = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )[:top_k]
    results = []

    for index in ranked_indexes:
        technique = techniques[int(index)]
        results.append(
            {
                "technique_id": technique.get("technique_id", ""),
                "name": technique.get("name", ""),
                "description": technique.get("description", ""),
                "tactics": _as_list(technique.get("tactics", [])),
                "platforms": _as_list(technique.get("platforms", [])),
                "similarity_score": round(float(scores[index]), 6),
                "field_weighted_score": round(float(scores[index]), 6),
            }
        )

    return results


def load_attack_techniques(cache_path: Path = ATTACK_CACHE_PATH) -> List[Dict[str, Any]]:
    """Load the local ATT&CK technique cache."""
    try:
        with cache_path.open("r", encoding="utf-8") as file:
            techniques = json.load(file)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"ATT&CK cache file not found: {cache_path}. "
            "Run `python3 src/download_attack_data.py` first."
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"ATT&CK cache file contains invalid JSON: {exc}") from exc

    if not isinstance(techniques, list):
        raise ValueError("ATT&CK cache must contain a JSON list of techniques.")

    return [technique for technique in techniques if isinstance(technique, dict)]


def alert_to_search_text(alert: Dict[str, Any]) -> str:
    """Convert an alert dictionary into searchable text."""
    text_blocks = []

    for field, weight in ALERT_FIELD_WEIGHTS.items():
        block = _stringify(alert.get(field, ""))
        if block:
            text_blocks.extend([block] * weight)

    return " ".join(text_blocks)


def expand_security_terms(text: str) -> str:
    """Compatibility helper: return text unchanged for baseline retrieval."""
    return text


def technique_to_search_text(technique: Dict[str, Any]) -> str:
    """Convert a technique dictionary into searchable text."""
    return " ".join(
        _stringify(technique.get(field, "")) for field in TECHNIQUE_SEARCH_FIELDS
    )


def calculate_hybrid_score(
    alert_text: str,
    technique: dict,
    tfidf_score: float,
) -> float:
    """Compatibility helper: return the original TF-IDF score unchanged."""
    return tfidf_score


def calculate_field_weighted_scores(
    techniques: List[Dict[str, Any]],
    query_tokens: List[str],
) -> List[float]:
    """Calculate weighted BM25 scores across individual ATT&CK fields."""
    scores = [0.0] * len(techniques)

    for field, weight in FIELD_SCORE_WEIGHTS.items():
        field_texts = [technique_field_text(technique, field) for technique in techniques]
        field_scores = bm25_field_scores(field_texts, query_tokens)

        for index, score in enumerate(field_scores):
            scores[index] += weight * float(score)

    return scores


def bm25_field_scores(field_texts: List[str], query_tokens: List[str]):
    """Score one ATT&CK field across all techniques with BM25."""
    tokenized_corpus = [tokenize_text(text) for text in field_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25.get_scores(query_tokens)


def technique_field_text(technique: Dict[str, Any], field: str) -> str:
    """Convert one technique field into searchable text."""
    return _stringify(technique.get(field, ""))


def tokenize_text(text: str) -> List[str]:
    """Tokenize text with a simple lowercase regex tokenizer."""
    return TOKEN_PATTERN.findall(_stringify(text).lower())


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return " ".join(_stringify(item) for item in value)
    if isinstance(value, dict):
        return " ".join(_stringify(item) for item in value.values())
    return str(value)


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def main() -> int:
    try:
        alerts = load_alerts(ALERTS_PATH, limit=3)
        for alert in alerts:
            results = retrieve_attack_techniques(alert, top_k=5)
            print_retrieval_report(alert, results)
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        print(exc)
        return 1

    return 0


def load_alerts(alerts_path: Path = ALERTS_PATH, limit: int = 3) -> List[Dict[str, Any]]:
    """Load alert rows from the local alerts CSV."""
    try:
        with alerts_path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            alerts = [dict(row) for _, row in zip(range(limit), reader)]
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Alerts file not found: {alerts_path}") from exc

    if not alerts:
        raise ValueError(f"Alerts file is empty: {alerts_path}")

    return alerts


def print_retrieval_report(
    alert: Dict[str, Any],
    results: List[Dict[str, Any]],
) -> None:
    """Print a compact retrieval report for one alert."""
    print(f"alert_id: {alert.get('alert_id', '')}")
    print(f"alert_name: {alert.get('alert_name', '')}")
    print(f"expected_technique_id: {alert.get('expected_technique_id', '')}")
    print(f"expected_technique_name: {alert.get('expected_technique_name', '')}")
    print("top_5_candidates:")

    for candidate in results:
        print(
            "  - "
            f"{candidate.get('technique_id', '')} | "
            f"{candidate.get('name', '')} | "
            f"similarity_score={candidate.get('similarity_score', 0)} | "
            f"field_weighted_score={candidate.get('field_weighted_score', 0)}"
        )

    print()


if __name__ == "__main__":
    raise SystemExit(main())
