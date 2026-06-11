"""Retrieve relevant MITRE ATT&CK techniques for an alert."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ModuleNotFoundError as exc:
    TfidfVectorizer = None
    cosine_similarity = None
    SKLEARN_IMPORT_ERROR = exc
else:
    SKLEARN_IMPORT_ERROR = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ATTACK_CACHE_PATH = PROJECT_ROOT / "data" / "attack_techniques.json"
ALERTS_PATH = PROJECT_ROOT / "data" / "alerts.csv"
ALERT_SEARCH_FIELDS = (
    "alert_id",
    "alert_name",
    "event_id",
    "source",
    "host",
    "user",
    "process",
    "command",
    "severity",
    "description",
    "detection_logic",
)
TECHNIQUE_SEARCH_FIELDS = (
    "technique_id",
    "name",
    "description",
    "tactics",
    "platforms",
)


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
    if TfidfVectorizer is None or cosine_similarity is None:
        missing = getattr(SKLEARN_IMPORT_ERROR, "name", "scikit-learn")
        raise RuntimeError(
            f"Missing dependency: {missing}. "
            "Install it with `python3 -m pip install scikit-learn`."
        )

    techniques = load_attack_techniques(cache_path)
    if not techniques:
        return []

    alert_text = alert_to_search_text(alert)
    technique_texts = [technique_to_search_text(technique) for technique in techniques]

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform([alert_text, *technique_texts])
    scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten()

    ranked_indexes = scores.argsort()[::-1][:top_k]
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
    return " ".join(_stringify(alert.get(field, "")) for field in ALERT_SEARCH_FIELDS)


def technique_to_search_text(technique: Dict[str, Any]) -> str:
    """Convert a technique dictionary into searchable text."""
    return " ".join(
        _stringify(technique.get(field, "")) for field in TECHNIQUE_SEARCH_FIELDS
    )


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
        alert = load_first_alert(ALERTS_PATH)
        results = retrieve_attack_techniques(alert, top_k=5)
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        print(exc)
        return 1

    print(json.dumps(results, indent=2))
    return 0


def load_first_alert(alerts_path: Path = ALERTS_PATH) -> Dict[str, Any]:
    """Load the first alert row from the local alerts CSV."""
    try:
        with alerts_path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            first_alert = next(reader, None)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Alerts file not found: {alerts_path}") from exc

    if first_alert is None:
        raise ValueError(f"Alerts file is empty: {alerts_path}")

    return dict(first_alert)


if __name__ == "__main__":
    raise SystemExit(main())
