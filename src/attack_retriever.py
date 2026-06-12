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
    text_blocks = []

    for field, weight in ALERT_FIELD_WEIGHTS.items():
        block = _stringify(alert.get(field, ""))
        if block:
            text_blocks.extend([block] * weight)

    return expand_security_terms(" ".join(text_blocks))


def expand_security_terms(text: str) -> str:
    """Append generic security terms that improve lexical retrieval."""
    expanded_terms = []
    normalized_text = text.lower()

    authentication_failure_phrases = (
        "failed logon",
        "failed login",
        "failed authentication",
        "failed interactive logon",
        "excessive failed logon",
    )
    if any(phrase in normalized_text for phrase in authentication_failure_phrases):
        expanded_terms.append(
            "authentication failure password guessing brute force credential access "
            "password attack"
        )

    password_spray_phrases = ("password spray", "password spraying")
    if any(phrase in normalized_text for phrase in password_spray_phrases):
        expanded_terms.append(
            "password spraying credential access authentication attempts"
        )

    powershell_phrases = ("encodedcommand", "encoded command", "powershell")
    if any(phrase in normalized_text for phrase in powershell_phrases):
        expanded_terms.append(
            "powershell command execution scripting interpreter execution"
        )

    if not expanded_terms:
        return text

    return " ".join([text, *expanded_terms])


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
            f"similarity_score={candidate.get('similarity_score', 0)}"
        )

    print()


if __name__ == "__main__":
    raise SystemExit(main())
