from __future__ import annotations

import json
from pathlib import Path

import pytest

from normalize_pcrl_evaluation_metadata import (
    PROVENANCE_FIELD,
    normalize_evaluation_metadata,
    sha256_bytes,
)


def write_config(path: Path, version: str, *, deadline: float = 14.0) -> None:
    path.write_text(
        json.dumps(
            {
                "version": version,
                "scenario": {"max_uavs": 4, "workload_scale": 1.8},
                "scale_deadlines": {"2x12": deadline},
            }
        ),
        encoding="utf-8",
    )


def write_rows(
    path: Path,
    scenario_version: str,
    *,
    protocol_version: str | None = None,
) -> None:
    rows = [
        {
            "scenario_version": scenario_version,
            "preference_profile": profile,
            "scale": "2x12",
            "eval_seed": 73000,
            **(
                {"protocol_version": protocol_version}
                if protocol_version is not None
                else {}
            ),
        }
        for profile in ("balanced", "calibration_search_priority")
    ]
    path.write_text(json.dumps(rows), encoding="utf-8")


def test_normalization_writes_derived_copy_and_full_provenance(
    tmp_path: Path,
) -> None:
    source_config = tmp_path / "source.json"
    target_config = tmp_path / "target.json"
    source = tmp_path / "raw" / "evaluation.json"
    output = tmp_path / "derived" / "evaluation.json"
    source.parent.mkdir()
    write_config(source_config, "pcrl-v0-hard-3")
    write_config(target_config, "gppo-v2-hard-3")
    write_rows(source, "pcrl-v0-hard-3")
    raw_before = source.read_bytes()

    manifest = normalize_evaluation_metadata(
        source_evaluation=source,
        output=output,
        source_config=source_config,
        target_config=target_config,
        target_protocol_version="pcrl-v0-hard-3-anchor-grid-coarse",
    )

    assert source.read_bytes() == raw_before
    rows = json.loads(output.read_text(encoding="utf-8"))
    assert all(row["scenario_version"] == "gppo-v2-hard-3" for row in rows)
    assert all(
        row["protocol_version"] == "pcrl-v0-hard-3-anchor-grid-coarse"
        for row in rows
    )
    provenance = rows[0][PROVENANCE_FIELD]
    assert provenance["source_evaluation"]["path"] == str(source.resolve())
    assert provenance["source_evaluation"]["sha256"] == sha256_bytes(raw_before)
    assert provenance["source_config"]["version"] == "pcrl-v0-hard-3"
    assert provenance["target_config"]["version"] == "gppo-v2-hard-3"
    assert provenance["config_equality"] == {
        "scenario_equal": True,
        "scale_deadlines_equal": True,
    }
    assert provenance["protocol"] == {
        "source": None,
        "target": "pcrl-v0-hard-3-anchor-grid-coarse",
    }
    assert manifest["derived_evaluation"]["sha256"] == sha256_bytes(
        output.read_bytes()
    )
    assert output.with_suffix(".json.provenance.json").is_file()


@pytest.mark.parametrize(("change", "match"), (("scenario", "scenario_equal=False"), ("deadline", "scale_deadlines_equal=False")))
def test_normalization_rejects_nonidentical_configs(
    tmp_path: Path, change: str, match: str
) -> None:
    source_config = tmp_path / "source.json"
    target_config = tmp_path / "target.json"
    source = tmp_path / "evaluation.json"
    output = tmp_path / "derived.json"
    write_config(source_config, "source")
    write_config(target_config, "target", deadline=15.0 if change == "deadline" else 14.0)
    if change == "scenario":
        payload = json.loads(target_config.read_text(encoding="utf-8"))
        payload["scenario"]["max_uavs"] = 5
        target_config.write_text(json.dumps(payload), encoding="utf-8")
    write_rows(source, "source")
    with pytest.raises(ValueError, match=match):
        normalize_evaluation_metadata(
            source_evaluation=source,
            output=output,
            source_config=source_config,
            target_config=target_config,
            target_protocol_version="target-protocol",
        )
    assert not output.exists()


def test_normalization_rejects_wrong_source_version_and_raw_overwrite(
    tmp_path: Path,
) -> None:
    source_config = tmp_path / "source_config.json"
    target_config = tmp_path / "target_config.json"
    source = tmp_path / "evaluation.json"
    write_config(source_config, "source")
    write_config(target_config, "target")
    write_rows(source, "wrong")
    with pytest.raises(ValueError, match="scenario_version"):
        normalize_evaluation_metadata(
            source_evaluation=source,
            output=tmp_path / "derived.json",
            source_config=source_config,
            target_config=target_config,
            target_protocol_version="target-protocol",
        )
    with pytest.raises(ValueError, match="derived path"):
        normalize_evaluation_metadata(
            source_evaluation=source,
            output=source,
            source_config=source_config,
            target_config=target_config,
            target_protocol_version="target-protocol",
        )


def test_existing_source_protocol_requires_exact_explicit_expectation(
    tmp_path: Path,
) -> None:
    source_config = tmp_path / "config.json"
    source = tmp_path / "evaluation.json"
    write_config(source_config, "gppo-v2-hard-3")
    write_rows(source, "gppo-v2-hard-3", protocol_version="pcrl-v0-hard-3")
    common = {
        "source_evaluation": source,
        "source_config": source_config,
        "target_config": source_config,
        "target_protocol_version": "pcrl-v0-hard-3-anchor-grid-coarse",
    }
    with pytest.raises(ValueError, match="expected-source-protocol"):
        normalize_evaluation_metadata(output=tmp_path / "missing.json", **common)
    with pytest.raises(ValueError, match="does not exactly match"):
        normalize_evaluation_metadata(
            output=tmp_path / "mismatch.json",
            expected_source_protocol="wrong",
            **common,
        )
    manifest = normalize_evaluation_metadata(
        output=tmp_path / "derived.json",
        expected_source_protocol="pcrl-v0-hard-3",
        **common,
    )
    assert manifest["protocol"] == {
        "source": "pcrl-v0-hard-3",
        "target": "pcrl-v0-hard-3-anchor-grid-coarse",
    }


def test_normalization_refuses_to_overwrite_existing_derived_file(
    tmp_path: Path,
) -> None:
    source_config = tmp_path / "config.json"
    source = tmp_path / "evaluation.json"
    output = tmp_path / "derived.json"
    write_config(source_config, "source")
    write_rows(source, "source")
    output.write_text("user data", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        normalize_evaluation_metadata(
            source_evaluation=source,
            output=output,
            source_config=source_config,
            target_config=source_config,
            target_protocol_version="target-protocol",
        )
    assert output.read_text(encoding="utf-8") == "user data"
