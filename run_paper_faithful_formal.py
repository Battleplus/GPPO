from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from uav_assignment.paper_faithful_env import PAPER_SCALES, PaperScale


@dataclass(frozen=True, slots=True)
class Job:
    scale: PaperScale
    mode: str
    sync_mode: str
    seed: int

    @property
    def name(self) -> str:
        return f"{self.mode}_{self.sync_mode}_seed{self.seed}"


def parse_scale(value: str) -> PaperScale:
    normalized = value.upper().removeprefix("T")
    parts = tuple(int(part) for part in normalized.split("-"))
    for scale in PAPER_SCALES:
        if (scale.uavs, scale.parent_tasks, scale.subtasks) == parts:
            return scale
    raise argparse.ArgumentTypeError(f"unknown paper scale: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the paper-faithful formal matrix")
    parser.add_argument("--scale", type=parse_scale, action="append")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=("ppo_mlp:none", "ppo_mlp:event", "literal:none", "literal:event"),
        help="mode:sync pairs; add literal:periodic, literal:always, literal_no_gate:event, etc.",
    )
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--validation-interval", type=int, default=50)
    parser.add_argument("--validation-instances", type=int, default=100)
    parser.add_argument("--rrelu-mode", choices=("expected", "stochastic"), default="expected")
    parser.add_argument("--gate-bias-init", type=float, default=0.0)
    parser.add_argument("--gate-activation", choices=("sigmoid", "softplus"), default="sigmoid")
    parser.add_argument(
        "--gate-scope",
        choices=("task_message", "score", "aggregate"),
        default="task_message",
    )
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--rerun-completed",
        action="store_true",
        help="Retrain jobs whose output already contains a final checkpoint.pt.",
    )
    parser.add_argument("--output-root", type=Path, default=Path("outputs/paper_faithful/formal"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def jobs_from_args(args: argparse.Namespace) -> list[Job]:
    scales = tuple(args.scale or PAPER_SCALES)
    seeds = tuple(args.seed or (1, 2, 3, 4, 5))
    jobs: list[Job] = []
    for scale in scales:
        for method in args.methods:
            mode, sync_mode = method.split(":", maxsplit=1)
            if mode not in {"ppo_mlp", "literal", "literal_no_gate", "literal_single_head"}:
                raise ValueError(f"unsupported mode: {mode}")
            if sync_mode not in {"none", "event", "periodic", "always"}:
                raise ValueError(f"unsupported sync mode: {sync_mode}")
            if mode == "ppo_mlp" and sync_mode not in {"none", "event"}:
                raise ValueError("PPO controls are limited to none/event in the 2x2 matrix")
            for seed in seeds:
                jobs.append(Job(scale, mode, sync_mode, seed))
    return jobs


def command_for(job: Job, args: argparse.Namespace, output: Path) -> list[str]:
    command = [
        sys.executable,
        "train_paper_faithful.py",
        "--mode", job.mode,
        "--rrelu-mode", args.rrelu_mode,
        "--gate-bias-init", str(args.gate_bias_init),
        "--gate-activation", getattr(args, "gate_activation", "sigmoid"),
        "--gate-scope", args.gate_scope,
        "--sync-mode", job.sync_mode,
        "--scale", job.scale.name,
        "--seed", str(job.seed),
        "--iterations", str(args.iterations),
        "--rollout-steps", str(args.rollout_steps),
        "--batch-size", str(args.batch_size),
        "--update-epochs", str(args.update_epochs),
        "--validation-interval", str(args.validation_interval),
        "--validation-instances", str(args.validation_instances),
        "--output", str(output),
    ]
    resume = output / "resume_latest.pt"
    if resume.exists():
        command.extend(("--resume-from", str(resume)))
    else:
        candidates = sorted(output.glob("candidate_*.pt"))
        if candidates:
            command.extend(("--resume-from", str(candidates[-1])))
    return command


def main() -> None:
    args = parse_args()
    jobs = jobs_from_args(args)
    manifest = {
        "version": "paper-faithful-formal-runner-v1",
        "protocol": "configs/paper_faithful_protocol.json",
        "iterations": args.iterations,
        "rollout_steps": args.rollout_steps,
        "batch_size": args.batch_size,
        "update_epochs": args.update_epochs,
        "validation_interval": args.validation_interval,
        "validation_instances": args.validation_instances,
        "rrelu_mode": args.rrelu_mode,
        "gate_bias_init": args.gate_bias_init,
        "gate_activation": args.gate_activation,
        "gate_scope": args.gate_scope,
        "jobs": [
            {
                "scale": job.scale.name,
                "mode": job.mode,
                "sync_mode": job.sync_mode,
                "seed": job.seed,
                "output": str((args.output_root / job.scale.name / job.name).as_posix()),
            }
            for job in jobs
        ],
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "formal_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({"jobs": len(jobs), "manifest": str(args.output_root / "formal_manifest.json")}, ensure_ascii=False))
    if args.dry_run:
        for job in jobs:
            output = args.output_root / job.scale.name / job.name
            print(" ".join(command_for(job, args, output)))
        return
    pending = [
        job
        for job in jobs
        if args.rerun_completed
        or not (args.output_root / job.scale.name / job.name / "checkpoint.pt").exists()
    ]
    skipped = len(jobs) - len(pending)
    if skipped:
        print(json.dumps({"skipped_completed": skipped}, ensure_ascii=False))
    active: list[tuple[subprocess.Popen[str], Job]] = []
    while pending or active:
        while pending and len(active) < max(1, args.jobs):
            job = pending.pop(0)
            output = args.output_root / job.scale.name / job.name
            output.mkdir(parents=True, exist_ok=True)
            stream = (output / "runner.log").open("w", encoding="utf-8")
            process = subprocess.Popen(
                command_for(job, args, output),
                cwd=Path(__file__).resolve().parent,
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
            )
            active.append((process, job))
        still_active: list[tuple[subprocess.Popen[str], Job]] = []
        for process, job in active:
            code = process.poll()
            if code is None:
                still_active.append((process, job))
            elif code != 0:
                raise RuntimeError(f"formal job failed: {job.name} ({job.scale.name}), code={code}")
            else:
                print(
                    json.dumps(
                        {"completed": f"{job.scale.name}/{job.name}"},
                        ensure_ascii=False,
                    )
                )
        active = still_active
        if active:
            import time
            time.sleep(2.0)


if __name__ == "__main__":
    main()
