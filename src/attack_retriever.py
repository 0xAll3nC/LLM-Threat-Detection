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

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception as exc:
    SentenceTransformer = None
    cosine_similarity = None
    DENSE_IMPORT_ERROR = exc
else:
    DENSE_IMPORT_ERROR = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ATTACK_CACHE_PATH = PROJECT_ROOT / "data" / "attack_techniques.json"
ALERTS_PATH = PROJECT_ROOT / "data" / "alerts.csv"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
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
_EMBEDDING_MODEL = None
_DENSE_TECHNIQUES = None
_DENSE_TECHNIQUE_EMBEDDINGS = None


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


def retrieve_attack_techniques_dense(alert: dict, top_k: int = 10) -> list[dict]:
    """Return the top matching ATT&CK techniques using dense embeddings."""
    if not isinstance(alert, dict):
        raise ValueError("Alert must be a dictionary.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if SentenceTransformer is None or cosine_similarity is None:
        raise RuntimeError(
            "Dense retrieval dependencies are unavailable. "
            f"Install/fix sentence-transformers and scikit-learn. Error: {DENSE_IMPORT_ERROR}"
        )

    techniques, technique_embeddings = get_dense_technique_index()
    if not techniques:
        return []

    alert_text = alert_to_search_text(alert)
    model = get_embedding_model()
    alert_embedding = model.encode(
        [alert_text],
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    scores = cosine_similarity(alert_embedding, technique_embeddings).flatten()

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
                "dense_score": round(float(scores[index]), 6),
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


def dense_technique_to_search_text(technique: Dict[str, Any]) -> str:
    """Convert a technique dictionary into dense retrieval text."""
    return " ".join(
        [
            _stringify(technique.get("technique_id", "")),
            _stringify(technique.get("name", "")),
            _stringify(technique.get("tactics", [])),
            _stringify(technique.get("platforms", [])),
            _stringify(technique.get("description", "")),
        ]
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


def get_embedding_model():
    """Load the dense embedding model once per process."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        _EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _EMBEDDING_MODEL


def get_dense_technique_index():
    """Load ATT&CK techniques and dense embeddings once per process."""
    global _DENSE_TECHNIQUES, _DENSE_TECHNIQUE_EMBEDDINGS
    if _DENSE_TECHNIQUES is None or _DENSE_TECHNIQUE_EMBEDDINGS is None:
        _DENSE_TECHNIQUES = load_attack_techniques(ATTACK_CACHE_PATH)
        technique_texts = [
            dense_technique_to_search_text(technique)
            for technique in _DENSE_TECHNIQUES
        ]
        _DENSE_TECHNIQUE_EMBEDDINGS = get_embedding_model().encode(
            technique_texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    return _DENSE_TECHNIQUES, _DENSE_TECHNIQUE_EMBEDDINGS


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
            print_dense_retrieval_report(alert)
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


def print_dense_retrieval_report(alert: Dict[str, Any]) -> None:
    """Print dense retrieval results for one alert, if available."""
    print("dense_top_5_candidates:")

    try:
        results = retrieve_attack_techniques_dense(alert, top_k=5)
    except Exception as exc:
        print(f"  dense retrieval unavailable: {exc}")
        print()
        return

    for candidate in results:
        print(
            "  - "
            f"{candidate.get('technique_id', '')} | "
            f"{candidate.get('name', '')} | "
            f"dense_score={candidate.get('dense_score', 0)}"
        )

    print()


if __name__ == "__main__":
    raise SystemExit(main())
