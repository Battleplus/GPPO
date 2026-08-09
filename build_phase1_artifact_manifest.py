from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def hash_stable(path: Path) -> tuple[str, int]:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"File changed while hashing: {path}")
    return digest.hexdigest(), after.st_size


def collect(namespace: str, root: Path, excluded: set[Path]) -> list[dict[str, Any]]:
    if not root.is_dir():
        return []
    records = []
    resolved_root = root.resolve()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        resolved = path.resolve()
        if resolved in excluded:
            continue
        try:
            relative = resolved.relative_to(resolved_root).as_posix()
        except ValueError as error:
            raise RuntimeError(f"Artifact escapes declared root: {path}") from error
        digest, size = hash_stable(resolved)
        records.append({
            "logical_path": f"{namespace}/{relative}",
            "source_path": str(resolved),
            "size": size,
            "sha256": digest,
        })
    return records


def json_valid(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("valid") is True
    except (OSError, json.JSONDecodeError, AttributeError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the final hash-complete Phase-1 artifact manifest.")
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--formal-root", type=Path, default=Path("outputs/paper_faithful/formal"))
    parser.add_argument("--final-root", type=Path, default=Path("outputs/phase1_final"))
    parser.add_argument("--output", type=Path, default=Path("PHASE1_ARTIFACT_MANIFEST.json"))
    args = parser.parse_args()
    repo = args.repo.resolve()
    formal = args.formal_root.resolve()
    final = args.final_root.resolve()
    output = args.output.resolve()
    excluded = {output, (repo / "PHASE1_RESULTS.zip").resolve()}

    # Repo scope is intentionally selective: source/config/report evidence is
    # included, while .git, venv, caches and ignored scratch outputs are not.
    records: list[dict[str, Any]] = []
    for relative in ("src", "tests", "configs", "docs", "reports", "artifacts"):
        records += collect(f"repo/{relative}", repo / relative, excluded)
    for path in sorted(repo.glob("*.py")) + sorted(repo.glob("*.ps1")) + sorted(repo.glob("requirements*.txt")):
        digest, size = hash_stable(path.resolve())
        records.append({
            "logical_path": f"repo/{path.name}", "source_path": str(path.resolve()),
            "size": size, "sha256": digest,
        })
    records += collect("formal", formal, excluded)
    records += collect("final", final, excluded)
    for name in (
        "PHASE1_FINAL_REPORT_ZH.md", "PHASE1_REPRODUCTION_GUIDE.md", "PHASE1_CONFIG_AUDIT.json"
    ):
        path = repo / name
        if path.is_file():
            digest, size = hash_stable(path)
            records.append({"logical_path": f"repo/{name}", "source_path": str(path), "size": size, "sha256": digest})

    logical_paths = [record["logical_path"] for record in records]
    duplicate_paths = sorted({name for name in logical_paths if logical_paths.count(name) > 1})
    required = {
        "phase1_config_audit": repo / "PHASE1_CONFIG_AUDIT.json",
        "gate_screening": repo / "outputs/gate_screening/summary.json",
        "frozen_protocol": repo / "configs/PHASE1_FROZEN_PROTOCOL.json",
        "formal_matrix": formal / "PHASE1_FORMAL_MATRIX_RUN.json",
        "communication_audit": formal / "PHASE1_COMMUNICATION_CAUSAL_AUDIT.json",
        "disturbance_implementation": final / "DISTURBANCE_IMPLEMENTATION_AUDIT.json",
        "disturbance_calibration": final / "DISTURBANCE_CALIBRATION_AUDIT.json",
        "final_experiment": final / "PHASE1_FINAL_EXPERIMENT_AUDIT.json",
        "final_report": repo / "PHASE1_FINAL_REPORT_ZH.md",
        "reproduction_guide": repo / "PHASE1_REPRODUCTION_GUIDE.md",
    }
    evidence = {
        name: {
            "path": str(path), "exists": path.is_file(),
            "valid_json": json_valid(path) if path.suffix.lower() == ".json" else None,
        }
        for name, path in required.items()
    }
    required_valid = all(
        item["exists"] and (item["valid_json"] is not False)
        for item in evidence.values()
    )
    payload = {
        "version": "phase1-artifact-manifest-v1",
        "hash_algorithm": "SHA256",
        "records": records,
        "file_count": len(records),
        "total_bytes": sum(record["size"] for record in records),
        "duplicate_logical_paths": duplicate_paths,
        "required_evidence": evidence,
        "all_hashes_verified": not duplicate_paths,
        "valid": bool(records and not duplicate_paths and required_valid),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "files": len(records), "valid": payload["valid"]}, ensure_ascii=False))
    if not payload["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
