"""Download and cache simplified MITRE ATT&CK Enterprise techniques."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from requests.exceptions import RequestException
except ModuleNotFoundError:
    RequestException = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "attack_techniques.json"


def main() -> int:
    """Fetch Enterprise ATT&CK techniques and save a simplified JSON cache."""
    try:
        raw_techniques = fetch_enterprise_techniques()
        techniques = simplify_techniques(raw_techniques)
        save_techniques(techniques, OUTPUT_PATH)
    except ModuleNotFoundError as exc:
        missing_package = exc.name or "attackcti"
        print(
            f"Missing dependency: {missing_package}. "
            "Install it with `python3 -m pip install attackcti`."
        )
        return 1
    except (ConnectionError, TimeoutError) as exc:
        print(f"Could not connect to MITRE ATT&CK TAXII server: {exc}")
        return 1
    except OSError as exc:
        if _looks_like_taxii_network_error(exc):
            print(f"Could not connect to MITRE ATT&CK TAXII server: {exc}")
            return 1
        print(f"Could not write ATT&CK cache: {exc}")
        return 1
    except Exception as exc:
        if RequestException is not None and isinstance(exc, RequestException):
            print(f"Could not connect to MITRE ATT&CK TAXII server: {exc}")
            return 1
        if _looks_like_taxii_network_error(exc):
            print(f"Could not connect to MITRE ATT&CK TAXII server: {exc}")
            return 1
        print(f"Failed to download ATT&CK data: {exc}")
        return 1

    print(f"Saved {len(techniques)} techniques to {OUTPUT_PATH}")
    return 0


def fetch_enterprise_techniques() -> Iterable[Any]:
    """Connect to MITRE ATT&CK TAXII through attackcti and fetch techniques."""
    from attackcti import attack_client

    client = attack_client()

    try:
        return client.get_enterprise_techniques(stix_format=True)
    except TypeError:
        return client.get_enterprise_techniques()
    except Exception as exc:
        raise ConnectionError(exc) from exc


def simplify_techniques(raw_techniques: Iterable[Any]) -> List[Dict[str, Any]]:
    """Reduce STIX technique objects to fields used by this project."""
    techniques = []

    for technique in raw_techniques:
        if _is_revoked_or_deprecated(technique):
            continue

        technique_id = _extract_technique_id(technique)
        if not technique_id:
            continue

        techniques.append(
            {
                "technique_id": technique_id,
                "name": _get(technique, "name", ""),
                "description": _get(technique, "description", ""),
                "tactics": _extract_tactics(technique),
                "platforms": _get_list(technique, "x_mitre_platforms"),
            }
        )

    return sorted(techniques, key=lambda item: item["technique_id"])


def save_techniques(techniques: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(techniques, file, indent=2, sort_keys=True)
        file.write("\n")


def _is_revoked_or_deprecated(technique: Any) -> bool:
    return bool(_get(technique, "revoked", False)) or bool(
        _get(technique, "x_mitre_deprecated", False)
    )


def _extract_technique_id(technique: Any) -> Optional[str]:
    for reference in _get_list(technique, "external_references"):
        source_name = _get(reference, "source_name", "")
        external_id = _get(reference, "external_id", "")
        if source_name == "mitre-attack" and external_id:
            return external_id
    return None


def _extract_tactics(technique: Any) -> List[str]:
    tactics = []

    for phase in _get_list(technique, "kill_chain_phases"):
        if _get(phase, "kill_chain_name") != "mitre-attack":
            continue

        tactic = _get(phase, "phase_name")
        if tactic:
            tactics.append(tactic)

    return sorted(set(tactics))


def _get_list(item: Any, key: str) -> List[Any]:
    value = _get(item, key, [])
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return list(value) if isinstance(value, tuple) else [value]


def _get(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)

    if hasattr(item, "get"):
        try:
            return item.get(key, default)
        except TypeError:
            pass

    return getattr(item, key, default)


def _looks_like_taxii_network_error(exc: Exception) -> bool:
    message = str(exc).lower()
    network_markers = (
        "attack-taxii.mitre.org",
        "failed to resolve",
        "name resolution",
        "max retries exceeded",
        "connection refused",
        "connection aborted",
        "timed out",
    )
    return any(marker in message for marker in network_markers)


if __name__ == "__main__":
    raise SystemExit(main())
