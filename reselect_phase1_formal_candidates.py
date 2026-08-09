from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import torch


def expected_candidate_iterations() -> list[int]:
    return list(range(50, 2001, 50))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_save(payload: Any, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def evaluate(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        raise RuntimeError(f"Candidate evaluation failed; inspect {log}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reselect preserved formal candidates on validation-A.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--instances", type=int, default=100)
    parser.add_argument("--evaluator", type=Path, default=Path("evaluate_paper_faithful.py"))
    args = parser.parse_args()
    evaluator = args.evaluator if args.evaluator.is_absolute() else Path(__file__).resolve().parent / args.evaluator

    runs = sorted(path.parent for path in args.root.rglob("checkpoint.pt"))
    if len(runs) != 5:
        raise RuntimeError(f"Expected exactly five completed preserved runs, found {len(runs)}")
    records: list[dict[str, Any]] = []
    commands: list[tuple[list[str], Path]] = []
    for run in runs:
        metadata = run / "checkpoint.pt"
        candidates = sorted(run.glob("candidate_*.pt"))
        iterations = [int(path.stem.split("_")[-1]) for path in candidates]
        if iterations != expected_candidate_iterations():
            raise RuntimeError(f"Candidate grid is not complete for {run}: {iterations}")
        evaluations = []
        for candidate, iteration in zip(candidates, iterations):
            output = run / "evaluations" / "validation_a_candidates" / f"candidate_{iteration:04d}.json"
            evaluations.append({"iteration": iteration, "candidate": str(candidate), "output": str(output)})
            if output.is_file():
                payload = read_json(output)
                if (
                    payload.get("checkpoint_sha256") == sha256(candidate)
                    and payload.get("metadata_checkpoint_sha256") == sha256(metadata)
                    and payload.get("split") == "validation_a"
                    and payload.get("instances") == args.instances
                ):
                    continue
            commands.append(([
                sys.executable, str(evaluator), "--checkpoint", str(candidate),
                "--metadata-checkpoint", str(metadata), "--split", "validation_a",
                "--instances", str(args.instances), "--output", str(output),
            ], output.with_suffix(".log")))
        records.append({
            "run": str(run), "legacy_checkpoint": str(metadata),
            "legacy_checkpoint_sha256": sha256(metadata), "evaluations": evaluations,
        })

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        future_map = {executor.submit(evaluate, command, log): log for command, log in commands}
        for index, future in enumerate(as_completed(future_map), 1):
            future.result()
            print(f"[{index}/{len(future_map)}] {future_map[future]} completed", flush=True)

    for record in records:
        scored = []
        reference_seeds: list[int] | None = None
        reference_tapes: list[str] | None = None
        for item in record["evaluations"]:
            payload = read_json(Path(item["output"]))
            seeds = [int(row["instance_seed"]) for row in payload["rows"]]
            tapes = list(payload["event_tape_hashes"])
            reference_seeds = seeds if reference_seeds is None else reference_seeds
            reference_tapes = tapes if reference_tapes is None else reference_tapes
            if seeds != reference_seeds or tapes != reference_tapes:
                raise RuntimeError(f"Candidate instance/event banks differ within {record['run']}")
            scored.append((float(payload["summary"]["realized_makespan"]["mean"]), item))
        best_makespan, selected = min(scored, key=lambda pair: (pair[0], pair[1]["iteration"]))
        run = Path(record["run"])
        output = run / "checkpoint_phase1_frozen.pt"
        legacy = torch.load(record["legacy_checkpoint"], map_location="cpu", weights_only=False)
        candidate = torch.load(selected["candidate"], map_location="cpu", weights_only=False)
        frozen = dict(legacy)
        frozen["model_state"] = candidate["model_state"]
        frozen["best_iteration"] = int(selected["iteration"])
        frozen["best_makespan"] = float(best_makespan)
        frozen["phase1_selection"] = {
            "version": "phase1-validation-a-reselection-v1",
            "split": "validation_a",
            "instances": args.instances,
            "test_used": False,
            "candidate_checkpoint": selected["candidate"],
            "candidate_checkpoint_sha256": sha256(Path(selected["candidate"])),
            "legacy_checkpoint": record["legacy_checkpoint"],
            "legacy_checkpoint_sha256": record["legacy_checkpoint_sha256"],
            "validation_evaluation": selected["output"],
            "validation_evaluation_sha256": sha256(Path(selected["output"])),
        }
        if output.exists():
            existing = torch.load(output, map_location="cpu", weights_only=False)
            if existing.get("phase1_selection") != frozen["phase1_selection"]:
                raise FileExistsError(f"Refusing to overwrite a different frozen selection: {output}")
        else:
            atomic_save(frozen, output)
        record.update({
            "selected_iteration": int(selected["iteration"]),
            "validation_a_makespan": float(best_makespan),
            "frozen_checkpoint": str(output),
            "frozen_checkpoint_sha256": sha256(output),
            "test_used_for_selection": False,
        })

    # Ensure all five runs used the same validation-A bank and tape set.
    first = read_json(Path(records[0]["evaluations"][0]["output"]))
    first_seeds = [row["instance_seed"] for row in first["rows"]]
    first_tapes = first["event_tape_hashes"]
    common_bank = True
    for record in records[1:]:
        payload = read_json(Path(record["evaluations"][0]["output"]))
        common_bank &= [row["instance_seed"] for row in payload["rows"]] == first_seeds
        common_bank &= payload["event_tape_hashes"] == first_tapes
    manifest = {
        "version": "phase1-formal-candidate-reselection-v1",
        "selection_split": "validation_a", "instances": args.instances,
        "test_used_for_selection": False, "common_instance_and_event_bank": common_bank,
        "legacy_checkpoints_preserved": True, "records": records, "valid": bool(common_bank),
    }
    manifest_path = args.root / "PHASE1_CANDIDATE_RESELECTION.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not manifest["valid"]:
        raise RuntimeError("Formal candidate reselection audit failed")
    print(json.dumps({"manifest": str(manifest_path), "valid": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
