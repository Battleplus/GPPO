from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import timedelta
from pathlib import Path, PurePosixPath


EXPECTED_RUNS = 135
ITERATIONS_PER_RUN = 2000


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def run_checked(command: list[str], cwd: Path | None = None, log: Path | None = None) -> None:
    print("\n$", " ".join(command), flush=True)
    if log is None:
        result = subprocess.run(command, cwd=cwd, text=True)
    else:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as stream:
            result = subprocess.run(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        if log is not None and log.exists():
            print("\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-120:]))
        raise RuntimeError(f"Command failed with exit code {result.returncode}")


def restore_state(local_root: Path, drive_root: Path, migration_zip: Path) -> None:
    local_root.mkdir(parents=True, exist_ok=True)
    marker = local_root / "T5-10-48_literal_event" / "PHASE1_CANDIDATE_RESELECTION.json"
    drive_marker = drive_root / "T5-10-48_literal_event" / "PHASE1_CANDIDATE_RESELECTION.json"
    if not marker.exists() and drive_marker.exists():
        print("Restoring the latest formal backup from Drive ...", flush=True)
        shutil.copytree(drive_root, local_root, dirs_exist_ok=True)
    if not marker.exists():
        if not migration_zip.exists():
            raise FileNotFoundError(f"Missing migration archive: {migration_zip}")
        unpack_root = Path("/content/phase1_migration_unpack")
        if unpack_root.exists():
            shutil.rmtree(unpack_root)
        unpack_root.mkdir(parents=True)
        # PowerShell's Compress-Archive records Windows backslashes in entry
        # names.  Python on Linux treats those as literal filename characters,
        # so normalize every member before extracting on Colab.
        with zipfile.ZipFile(migration_zip) as archive:
            for member in archive.infolist():
                relative = PurePosixPath(member.filename.replace("\\", "/"))
                if relative.is_absolute() or ".." in relative.parts:
                    raise RuntimeError(f"Unsafe migration archive member: {member.filename}")
                target = unpack_root.joinpath(*relative.parts)
                if member.is_dir() or member.filename.endswith(("/", "\\")):
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
        markers = list(unpack_root.rglob("PHASE1_CANDIDATE_RESELECTION.json"))
        if len(markers) != 1:
            raise RuntimeError(f"Migration archive contains {len(markers)} protocol markers: {markers}")
        shutil.copytree(markers[0].parent.parent, local_root, dirs_exist_ok=True)
    frozen = list((local_root / "T5-10-48_literal_event").rglob("checkpoint_phase1_frozen.pt"))
    resumes = list((local_root / "T5_primary").rglob("resume_latest.pt"))
    if not marker.exists() or len(frozen) != 5:
        raise RuntimeError(f"Invalid migration state: marker={marker.exists()}, frozen={len(frozen)}")
    print(f"Migration restored: frozen={len(frozen)}, resumable PPO={len(resumes)}", flush=True)


def discover_progress(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in root.rglob("training_history.json"):
        history = read_json(path)
        if not history:
            continue
        last = history[-1]
        iteration = int(float(last.get("iteration", 0)))
        relative = str(path.parent.relative_to(root))
        rows.append({
            "run": relative,
            "iteration": iteration,
            "progress": f"{100.0 * iteration / ITERATIONS_PER_RUN:.1f}%",
            "reward": last.get("reward"),
            "makespan": last.get("realized_makespan"),
            "entropy": last.get("policy_entropy"),
            "updated": time.strftime("%H:%M:%S", time.localtime(path.stat().st_mtime)),
        })
    return sorted(rows, key=lambda row: str(row["run"]))


def gpu_status() -> str:
    try:
        return subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=name,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ], text=True).strip().splitlines()[0]
    except Exception:
        return "GPU status unavailable"


def show_progress(rows: list[dict[str, object]], total: int, elapsed: float, rate: float | None) -> None:
    percent = 100.0 * total / (EXPECTED_RUNS * ITERATIONS_PER_RUN)
    remaining = max(0, EXPECTED_RUNS * ITERATIONS_PER_RUN - total)
    eta = str(timedelta(seconds=int(remaining / rate))) if rate and rate > 0 else "estimating"
    print("\n" + "=" * 100)
    print(time.strftime("%Y-%m-%d %H:%M:%S"), f"overall={total:,}/{EXPECTED_RUNS * ITERATIONS_PER_RUN:,} ({percent:.2f}%)")
    print(f"elapsed={timedelta(seconds=int(elapsed))} | rate={rate or 0:.3f} iter/s | ETA={eta}")
    print("GPU:", gpu_status())
    print("active/recent runs:")
    for row in sorted(rows, key=lambda item: str(item["updated"]), reverse=True)[:12]:
        reward = row["reward"]
        makespan = row["makespan"]
        reward_text = f"{float(reward):.4f}" if reward is not None else "-"
        makespan_text = f"{float(makespan):.4f}" if makespan is not None else "-"
        print(f"  {row['updated']}  {row['iteration']:>4}/2000  R={reward_text:>9}  M={makespan_text:>9}  {row['run']}")
    print("=" * 100, flush=True)


def postprocess(repo: Path, root: Path) -> None:
    summary = root / "formal_summary.json"
    audit = root / "formal_artifact_audit.json"
    report_en = root / "formal_report.md"
    report_zh = root / "GPPO_REPRODUCTION_REPORT_ZH.md"
    run_checked([sys.executable, str(repo / "summarize_paper_faithful_formal.py"), "--root", str(root), "--output", str(summary)], repo)
    for metric, name in (
        ("reward", "fig8_reward.png"),
        ("realized_makespan", "fig8_realized_makespan.png"),
        ("validation_realized_makespan", "fig8_validation_makespan.png"),
    ):
        run_checked([sys.executable, str(repo / "plot_paper_faithful_fig8.py"), "--root", str(root), "--metric", metric, "--output", str(root / name)], repo)
    run_checked([sys.executable, str(repo / "audit_paper_faithful_formal.py"), "--root", str(root), "--instances", "100", "--output", str(audit)], repo)
    run_checked([sys.executable, str(repo / "report_paper_faithful_formal.py"), "--summary", str(summary), "--output", str(report_en)], repo)
    run_checked([sys.executable, str(repo / "report_paper_faithful_formal_zh.py"), "--summary", str(summary), "--audit", str(audit), "--output", str(report_zh)], repo)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--backup-seconds", type=int, default=600)
    parser.add_argument("--progress-seconds", type=int, default=20)
    args = parser.parse_args()

    repo = Path("/content/GPPO")
    drive_base = Path("/content/drive/MyDrive")
    drive_root = drive_base / "GPPO_phase1_formal"
    migration_zip = drive_base / "GPPO_phase1_migration.zip"
    local_root = Path("/content/phase1_formal")
    drive_root.mkdir(parents=True, exist_ok=True)
    restore_state(local_root, drive_root, migration_zip)
    shutil.copytree(local_root, drive_root, dirs_exist_ok=True)

    preflight_log = drive_root / "preflight_tests.log"
    run_checked([
        sys.executable, "-m", "pytest",
        "tests/test_paper_faithful.py",
        "tests/test_paper_faithful_baselines.py",
        "tests/test_paper_faithful_resume.py",
        "tests/test_paper_faithful_cuda.py", "-q",
    ], repo, preflight_log)

    matrix_log = drive_root / "colab_phase1_matrix.log"
    command = [
        sys.executable, str(repo / "run_phase1_formal_matrix.py"),
        "--frozen-protocol", str(repo / "configs/PHASE1_FROZEN_PROTOCOL.json"),
        "--formal-root", str(local_root),
        "--legacy-literal-root", str(local_root / "T5-10-48_literal_event"),
        "--python", sys.executable, "--jobs", str(args.jobs), "--device", "cuda",
    ]
    stream = matrix_log.open("a", encoding="utf-8")
    process = subprocess.Popen(command, cwd=repo, stdout=stream, stderr=subprocess.STDOUT, text=True)
    print(f"Formal matrix started: PID={process.pid}", flush=True)
    started = time.monotonic()
    last_backup = started
    previous_total = None
    previous_time = None
    smoothed_rate = None
    try:
        while process.poll() is None:
            now = time.monotonic()
            rows = discover_progress(local_root)
            total = sum(min(ITERATIONS_PER_RUN, int(row["iteration"])) for row in rows if "T5-10-48_literal_event" not in str(row["run"]))
            if previous_total is not None and previous_time is not None:
                current_rate = max(0, total - previous_total) / max(1.0, now - previous_time)
                if current_rate > 0:
                    smoothed_rate = current_rate if smoothed_rate is None else 0.8 * smoothed_rate + 0.2 * current_rate
            previous_total, previous_time = total, now
            show_progress(rows, total, now - started, smoothed_rate)
            if now - last_backup >= args.backup_seconds:
                print("Backing up resumable state to Drive ...", flush=True)
                shutil.copytree(local_root, drive_root, dirs_exist_ok=True)
                last_backup = now
            time.sleep(args.progress_seconds)
    finally:
        stream.close()
        shutil.copytree(local_root, drive_root, dirs_exist_ok=True)
    if process.wait() != 0:
        print("\n".join(matrix_log.read_text(encoding="utf-8", errors="replace").splitlines()[-150:]))
        raise RuntimeError("Formal matrix failed; rerun the notebook to resume")

    postprocess(repo, local_root)
    shutil.copytree(local_root, drive_root, dirs_exist_ok=True)
    archive = shutil.make_archive(str(drive_base / "GPPO_phase1_formal_complete"), "zip", root_dir=local_root)
    print("\nCOMPLETED")
    print("Drive results:", drive_root)
    print("Archive:", archive)
    for name in (
        "fig8_reward.png", "fig8_realized_makespan.png", "fig8_validation_makespan.png",
        "formal_summary.json", "formal_artifact_audit.json", "formal_report.md",
        "GPPO_REPRODUCTION_REPORT_ZH.md",
    ):
        print(" -", drive_root / name)


if __name__ == "__main__":
    main()
