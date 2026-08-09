from __future__ import annotations

from build_phase1_artifact_manifest import collect, hash_stable


def test_manifest_hashes_files_with_namespaced_paths(tmp_path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    path = root / "artifact.json"
    path.write_text("{}", encoding="utf-8")
    records = collect("evidence", root, set())
    assert records[0]["logical_path"] == "evidence/artifact.json"
    assert records[0]["sha256"] == hash_stable(path)[0]
    assert records[0]["size"] == 2


def test_manifest_excludes_its_own_output(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text("changing", encoding="utf-8")
    assert collect("root", tmp_path, {path.resolve()}) == []
