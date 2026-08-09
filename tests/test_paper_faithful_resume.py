from __future__ import annotations

import json
from types import SimpleNamespace

import torch

from run_paper_faithful_formal import Job, command_for
from train_paper_faithful import recover_candidate
from uav_assignment.paper_faithful_env import PAPER_SCALES


def runner_args() -> SimpleNamespace:
    return SimpleNamespace(
        rrelu_mode="expected",
        gate_bias_init=0.0,
        gate_warmup_iterations=0,
        gate_scope="task_message",
        iterations=2000,
        rollout_steps=512,
        batch_size=512,
        update_epochs=4,
        validation_interval=50,
        validation_instances=100,
        validation_split="validation",
    )


def test_runner_prefers_resume_then_falls_back_to_latest_candidate(tmp_path) -> None:
    job = Job(PAPER_SCALES[0], "literal", "event", 1)
    command = command_for(job, runner_args(), tmp_path)
    assert "--resume-from" not in command
    (tmp_path / "candidate_0050.pt").write_bytes(b"legacy candidate")
    (tmp_path / "candidate_0100.pt").write_bytes(b"later legacy candidate")
    command = command_for(job, runner_args(), tmp_path)
    index = command.index("--resume-from")
    assert command[index + 1] == str(tmp_path / "candidate_0100.pt")
    resume = tmp_path / "resume_latest.pt"
    resume.write_bytes(b"versioned resume placeholder")
    command = command_for(job, runner_args(), tmp_path)
    index = command.index("--resume-from")
    assert command[index + 1] == str(resume)


def test_candidate_recovery_trims_history_and_preserves_best_candidate(tmp_path) -> None:
    model = torch.nn.Linear(2, 1)
    state_50 = {key: value.clone() for key, value in model.state_dict().items()}
    with torch.no_grad():
        model.weight.add_(1.0)
    state_100 = {key: value.clone() for key, value in model.state_dict().items()}
    torch.save(
        {"iteration": 50, "model_state": state_50, "validation": {"realized_makespan": 8.0}},
        tmp_path / "candidate_0050.pt",
    )
    torch.save(
        {"iteration": 100, "model_state": state_100, "validation": {"realized_makespan": 9.0}},
        tmp_path / "candidate_0100.pt",
    )
    histories = [{"iteration": float(i)} for i in range(1, 111)]
    validations = [{"iteration": float(i)} for i in (50, 100)]
    (tmp_path / "training_history.json").write_text(json.dumps(histories), encoding="utf-8")
    (tmp_path / "validation_history.json").write_text(json.dumps(validations), encoding="utf-8")

    recovered = recover_candidate(tmp_path / "candidate_0100.pt", tmp_path, model)

    assert recovered["iteration"] == 100
    assert len(recovered["history"]) == 100
    assert recovered["best_iteration"] == 50
    assert recovered["recovery_info"]["discarded_training_rows"] == 10
    assert recovered["recovery_info"]["optimizer_state_recovered"] is False
    for key, value in state_100.items():
        assert torch.equal(model.state_dict()[key], value)
