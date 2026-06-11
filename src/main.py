"""Batch SOC alert triage entrypoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None

try:
    from llm_triage import triage_alert
except ModuleNotFoundError as exc:
    triage_alert = None
    TRIAGE_IMPORT_ERROR = exc
else:
    TRIAGE_IMPORT_ERROR = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "alerts.csv"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "prioritized_alerts.csv"


def main() -> int:
    """Read alerts, enrich them with LLM triage, and write prioritized results."""
    if pd is None:
        print("Missing dependency: pandas. Install it with `pip install pandas`.")
        return 1
    if triage_alert is None:
        missing = getattr(TRIAGE_IMPORT_ERROR, "name", "llm_triage")
        print(f"Missing dependency: {missing}. Install it before running triage.")
        return 1

    try:
        alerts = pd.read_csv(INPUT_PATH)
    except FileNotFoundError:
        print(f"Input file not found: {INPUT_PATH}")
        return 1
    except pd.errors.EmptyDataError:
        print(f"Input file is empty: {INPUT_PATH}")
        return 1
    except pd.errors.ParserError as exc:
        print(f"Could not parse input CSV: {exc}")
        return 1
    except OSError as exc:
        print(f"Could not read input CSV: {exc}")
        return 1

    prioritized_alerts = []

    for _, row in alerts.iterrows():
        alert = row.where(pd.notna(row), None).to_dict()
        triage = _triage_with_error_handling(alert)
        combined = {**alert, **triage}
        prioritized_alerts.append(combined)
        _print_alert_summary(combined)

    try:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(prioritized_alerts).to_csv(OUTPUT_PATH, index=False)
    except OSError as exc:
        print(f"Could not write output CSV: {exc}")
        return 1

    print(f"Saved prioritized alerts to {OUTPUT_PATH}")
    return 0


def _triage_with_error_handling(alert: Dict[str, Any]) -> Dict[str, Any]:
    try:
        triage = triage_alert(alert)
    except Exception as exc:  # Defensive: keep batch processing alive.
        return {
            "priority": "unknown",
            "mitre_attack_technique": "unknown",
            "confidence": "low",
            "explanation": f"Triage failed: {exc}",
            "recommended_action": "Review the alert manually.",
            "error": str(exc),
        }

    if not isinstance(triage, dict):
        return {
            "priority": "unknown",
            "mitre_attack_technique": "unknown",
            "confidence": "low",
            "explanation": "Triage function returned a non-dictionary result.",
            "recommended_action": "Review the alert manually.",
            "error": "Invalid triage result type.",
        }

    return triage


def _print_alert_summary(alert: Dict[str, Any]) -> None:
    alert_id = alert.get("alert_id", "unknown")
    priority = alert.get("priority", "unknown")
    technique = alert.get("mitre_attack_technique", "unknown")

    print(f"Alert ID: {alert_id}")
    print(f"Priority: {priority}")
    print(f"MITRE Technique: {technique}")
    print()


if __name__ == "__main__":
    raise SystemExit(main())
