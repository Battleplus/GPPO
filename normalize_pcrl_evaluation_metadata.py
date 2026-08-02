from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


NORMALIZATION_VERSION = "pcrl-evaluation-metadata-normalization-v1"
PROVENANCE_FIELD = "metadata_normalization_provenance"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _load_object(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload, raw


def _load_rows(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("source evaluation must contain a non-empty JSON row list")
    if not all(isinstance(row, dict) for row in payload):
        raise ValueError("source evaluation contains a non-object row")
    return payload, raw


def _config_identity(
    path: Path, payload: Mapping[str, Any], raw: bytes
) -> dict[str, Any]:
    version = str(payload.get("version", "")).strip()
    if not version:
        raise ValueError(f"config {path} must declare a non-empty version")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_bytes(raw),
        "version": version,
    }


def normalize_evaluation_metadata(
    *,
    source_evaluation: Path,
    output: Path,
    source_config: Path,
    target_config: Path,
    target_protocol_version: str,
    expected_source_protocol: str | None = None,
) -> dict[str, Any]:
    """Create a provenance-rich derived copy after strict config validation."""

    source_evaluation = source_evaluation.resolve()
    output = output.resolve()
    source_config = source_config.resolve()
    target_config = target_config.resolve()
    if source_evaluation == output:
        raise ValueError("output must be a derived path, not the raw evaluation path")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite derived output: {output}")
    provenance_path = output.with_suffix(output.suffix + ".provenance.json")
    if provenance_path.exists():
        raise FileExistsError(
            f"refusing to overwrite provenance output: {provenance_path}"
        )
    target_protocol_version = str(target_protocol_version).strip()
    if not target_protocol_version:
        raise ValueError("target protocol version must be non-empty")
    if expected_source_protocol is not None:
        expected_source_protocol = str(expected_source_protocol).strip()
        if not expected_source_protocol:
            raise ValueError("expected source protocol must be non-empty when supplied")

    source_payload, source_config_raw = _load_object(
        source_config, label="source config"
    )
    target_payload, target_config_raw = _load_object(
        target_config, label="target config"
    )
    for label, payload in (
        ("source config", source_payload),
        ("target config", target_payload),
    ):
        if "scenario" not in payload or "scale_deadlines" not in payload:
            raise ValueError(f"{label} lacks scenario or scale_deadlines")
    scenario_equal = _canonical_json(source_payload["scenario"]) == _canonical_json(
        target_payload["scenario"]
    )
    deadlines_equal = _canonical_json(
        source_payload["scale_deadlines"]
    ) == _canonical_json(target_payload["scale_deadlines"])
    if not scenario_equal or not deadlines_equal:
        raise ValueError(
            "source and target configs are not physically identical: "
            f"scenario_equal={scenario_equal}, "
            f"scale_deadlines_equal={deadlines_equal}"
        )
    source_identity = _config_identity(
        source_config, source_payload, source_config_raw
    )
    target_identity = _config_identity(
        target_config, target_payload, target_config_raw
    )

    rows, source_raw = _load_rows(source_evaluation)
    source_version = source_identity["version"]
    for row_index, row in enumerate(rows):
        if str(row.get("scenario_version", "")).strip() != source_version:
            raise ValueError(
                f"source row {row_index} scenario_version does not match "
                f"source config version {source_version!r}"
            )
        if PROVENANCE_FIELD in row:
            raise ValueError(
                f"source row {row_index} already contains normalization provenance"
            )

    recorded_protocols = [
        str(row.get("protocol_version", "")).strip() for row in rows
    ]
    rows_with_protocol = [value for value in recorded_protocols if value]
    if rows_with_protocol:
        if len(rows_with_protocol) != len(rows):
            raise ValueError("source protocol_version is only partially recorded")
        if expected_source_protocol is None:
            raise ValueError(
                "source rows already record protocol_version; "
                "--expected-source-protocol is required"
            )
        if any(value != expected_source_protocol for value in rows_with_protocol):
            raise ValueError(
                "source protocol_version does not exactly match "
                "expected_source_protocol"
            )
        source_protocol: str | None = expected_source_protocol
    else:
        if expected_source_protocol is not None:
            raise ValueError(
                "expected_source_protocol was supplied but source rows do not "
                "record a protocol_version"
            )
        source_protocol = None

    normalized_at = datetime.now(timezone.utc).isoformat()
    row_provenance = {
        "normalization_version": NORMALIZATION_VERSION,
        "normalized_at_utc": normalized_at,
        "source_evaluation": {
            "path": str(source_evaluation),
            "sha256": sha256_bytes(source_raw),
            "row_count": len(rows),
        },
        "source_config": source_identity,
        "target_config": target_identity,
        "config_equality": {
            "scenario_equal": scenario_equal,
            "scale_deadlines_equal": deadlines_equal,
        },
        "protocol": {
            "source": source_protocol,
            "target": target_protocol_version,
        },
        "scenario_version": {
            "source": source_version,
            "target": target_identity["version"],
        },
    }
    derived_rows: list[dict[str, Any]] = []
    for row in rows:
        derived = dict(row)
        derived["scenario_version"] = target_identity["version"]
        derived["protocol_version"] = target_protocol_version
        derived[PROVENANCE_FIELD] = row_provenance
        derived_rows.append(derived)
    output_bytes = json.dumps(
        derived_rows, indent=2, ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(output_bytes)
    if source_evaluation.read_bytes() != source_raw:
        raise RuntimeError("raw evaluation changed during normalization")

    manifest = {
        **row_provenance,
        "derived_evaluation": {
            "path": str(output),
            "sha256": sha256_bytes(output_bytes),
            "row_count": len(derived_rows),
        },
    }
    provenance_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a derived PCRL evaluation with explicit, validated metadata"
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--target-config", type=Path, required=True)
    parser.add_argument("--protocol-version", required=True)
    parser.add_argument("--expected-source-protocol")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = normalize_evaluation_metadata(
        source_evaluation=args.input,
        output=args.output,
        source_config=args.source_config,
        target_config=args.target_config,
        target_protocol_version=args.protocol_version,
        expected_source_protocol=args.expected_source_protocol,
    )
    print(
        json.dumps(
            {
                "derived_evaluation": manifest["derived_evaluation"],
                "config_equality": manifest["config_equality"],
                "protocol": manifest["protocol"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
