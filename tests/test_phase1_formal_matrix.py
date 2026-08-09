from __future__ import annotations

from pathlib import Path

from run_phase1_formal_matrix import runner_command


def test_formal_runner_command_freezes_validation_a_and_cpu() -> None:
    command = runner_command(
        "python", Path("repo"), Path("out"), ("T5-10-48",), ("literal:event",), 4,
        {"gate_bias_init": 2.0, "gate_warmup_iterations": 50, "gate_activation": "sigmoid", "gate_scope": "task_message"},
    )
    assert command[command.index("--validation-split") + 1] == "validation_a"
    assert command[command.index("--validation-instances") + 1] == "100"
    assert command[command.index("--device") + 1] == "cpu"
    assert command[command.index("--gate-warmup-iterations") + 1] == "50"
    assert [command[index + 1] for index, value in enumerate(command) if value == "--seed"] == ["1", "2", "3", "4", "5"]
