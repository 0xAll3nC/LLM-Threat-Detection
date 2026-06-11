"""LLM-assisted SOC alert triage using a local Ollama model."""

from __future__ import annotations

import json
from typing import Any, Dict

import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3:8b"
TIMEOUT_SECONDS = 60

EXPECTED_FIELDS = {
    "priority": "unknown",
    "mitre_attack_technique": "unknown",
    "confidence": "low",
    "explanation": "",
    "recommended_action": "",
}


def triage_alert(alert: Dict[str, Any]) -> Dict[str, Any]:
    """Send an alert dictionary to Ollama and return normalized triage JSON."""
    if not isinstance(alert, dict):
        return _failure_response("Alert must be a dictionary.")

    payload = {
        "model": MODEL_NAME,
        "prompt": _build_prompt(alert),
        "stream": False,
        "format": "json",
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        ollama_data = response.json()
    except requests.RequestException as exc:
        return _failure_response(f"Failed to contact Ollama: {exc}")
    except ValueError as exc:
        return _failure_response(f"Ollama returned invalid JSON: {exc}")

    raw_model_response = ollama_data.get("response", "")
    if not raw_model_response:
        return _failure_response("Ollama response did not include model output.")

    try:
        triage = _parse_model_json(raw_model_response)
    except ValueError as exc:
        return _failure_response(f"Model returned invalid triage JSON: {exc}")

    return _normalize_triage(triage)


def _build_prompt(alert: Dict[str, Any]) -> str:
    alert_json = json.dumps(alert, indent=2, sort_keys=True, default=str)
    return (
        "You are an experienced Security Operations Center (SOC) analyst. "
        "Triage the following security alert.\n\n"
        "Return only valid JSON with exactly these keys:\n"
        '- "priority": one of "critical", "high", "medium", "low"\n'
        '- "mitre_attack_technique": the most relevant MITRE ATT&CK technique ID '
        'and name, or "unknown"\n'
        '- "confidence": one of "high", "medium", "low"\n'
        '- "explanation": a concise analyst explanation\n'
        '- "recommended_action": the next action a SOC analyst should take\n\n'
        f"Alert:\n{alert_json}"
    )


def _parse_model_json(raw_text: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in model response.")
        parsed = json.loads(raw_text[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("Triage response must be a JSON object.")

    return parsed


def _normalize_triage(triage: Dict[str, Any]) -> Dict[str, Any]:
    normalized = EXPECTED_FIELDS.copy()

    for key in normalized:
        value = triage.get(key)
        if value is not None:
            normalized[key] = value

    return normalized


def _failure_response(error: str) -> Dict[str, Any]:
    result = EXPECTED_FIELDS.copy()
    result.update(
        {
            "priority": "unknown",
            "confidence": "low",
            "explanation": error,
            "recommended_action": "Review the alert manually and verify Ollama is running.",
            "error": error,
        }
    )
    return result


if __name__ == "__main__":
    sample_alert = {
        "source_ip": "10.0.4.23",
        "destination_ip": "198.51.100.12",
        "event_type": "multiple_failed_logins",
        "username": "admin",
        "failed_attempts": 25,
    }
    print(json.dumps(triage_alert(sample_alert), indent=2))
