from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def variant_config(protocol: dict[str, Any], name: str) -> dict[str, Any]:
    try:
        variant = next(item for item in protocol["variants"] if item["name"] == name)
    except StopIteration as error:
        raise ValueError(f"Unknown frozen gate variant: {name}") from error
    return {
        key: variant[key]
        for key in (
            "graph_mode", "gate_scope", "gate_activation", "gate_bias_init",
            "gate_warmup_iterations",
        )
    }


def build_frozen_protocol(
    screening: dict[str, Any], gate_protocol: dict[str, Any], gate_protocol_hash: str
) -> dict[str, Any]:
    if screening.get("valid") is not True:
        raise ValueError("Gate screening summary is not valid")
    if screening.get("test_used_for_selection") is not False:
        raise ValueError("Refusing to freeze a protocol selected using test data")
    if screening.get("protocol_sha256") != gate_protocol_hash:
        raise ValueError("Screening summary and gate protocol hashes differ")

    outcome = screening["outcome"]
    selected_adaptive = outcome.get("selected_adaptive")
    best_name = selected_adaptive or outcome["engineering_baseline_if_no_adaptive_passes"]
    literal = dict(gate_protocol["frozen_literal_definition"])
    literal.pop("name", None)
    literal.pop("definition_must_not_change", None)
    literal.pop("post_gate_renormalization", None)
    literal["post_gate_renormalization"] = False

    return {
        "version": "phase1-frozen-protocol-v1",
        "frozen_after_gate_screening": True,
        "selection_data": "validation_a only",
        "confirmation_only_data": ["validation_b", "test100"],
        "test_used_for_selection": False,
        "source": {
            "gate_protocol_sha256": gate_protocol_hash,
            "gate_amendment_sha256": screening["amendment_sha256"],
            "gate_training_code_commit": screening["training_code_commit"],
            "screening_outcome": outcome,
        },
        "models": {
            "GPPO-Literal": {
                **literal,
                "role": "paper-compatible interpretation; retained even if negative",
            },
            "GPPO-Best": {
                **variant_config(gate_protocol, best_name),
                "selected_variant": best_name,
                "role": "validation-selected engineering baseline",
            },
            "PPO-MLP": {
                "graph_mode": "ppo_mlp",
                "role": "ordinary PPO without graph encoder",
            },
            "NoGate": {
                **variant_config(gate_protocol, "NoGate"),
                "role": "adaptive-gate ablation",
            },
            "SingleHead": {
                **variant_config(gate_protocol, "SingleHead"),
                "role": "simple-attention structural control",
            },
        },
        "formal_training": {
            "scales": ["T5-10-48", "T10-10-53", "T15-8-66", "T20-10-92"],
            "training_seeds": [1, 2, 3, 4, 5],
            "iterations": 2000,
            "rollout_steps": 512,
            "batch_size": 512,
            "update_epochs": 4,
            "learning_rate": 0.0002,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "ppo_clip": 0.2,
            "entropy_coefficient": 0.01,
            "value_coefficient": 0.5,
            "validation_interval": 50,
            "validation_instances": 100,
            "checkpoint_selection_split": "validation_a",
            "confirmation_split": "validation_b",
            "test_instances": 100,
            "rrelu_mode": "expected",
        },
        "comparisons": [
            "Random", "Greedy", "PPO-none", "PPO-event", "GPPO-Literal-none",
            "GPPO-Literal-event", "GPPO-Best-event", "NoGate-event", "SingleHead-event",
        ],
        "communication_replay": {
            "checkpoint_policy": "same event-trained GPPO checkpoint",
            "modes": ["none", "event", "periodic", "always"],
            "retraining_for_replay": False,
        },
        "statistics": {
            "unit": "training seed after aggregating fixed test100 within seed",
            "reported": ["mean", "std", "Student-t 95% CI", "paired seed difference", "improved seed count", "effect size"],
            "cross_scale": "per-scale statistics plus macro average; never pool episodes across scales into one CI",
        },
        "immutability": "No architecture or hyperparameter changes after formal test results are read.",
    }


def model_definitions(payload: dict[str, Any]) -> str:
    lines = [
        "# Phase 1 模型定义", "",
        "本文件由 Gate 三种子筛选结果生成。模型选择只使用 validation-A；validation-B 和 test100 仅用于确认或否证。", "",
    ]
    for name, config in payload["models"].items():
        lines += [f"## {name}", "", f"作用：{config['role']}。", "", "```json", json.dumps(config, indent=2, ensure_ascii=False), "```", ""]
    lines += [
        "## 冻结边界", "",
        "正式测试结果读取后，不得改变模型结构、Gate 作用域、激活函数、初始化、warmup 或 PPO 超参数。",
        "Literal 版本及全部负消融结果必须保留；GPPO-Best 不替代 Literal 的论文兼容解释。", "",
    ]
    return "\n".join(lines)


def write_once_or_verify(path: Path, content: bytes) -> None:
    if path.exists() and path.read_bytes() != content:
        raise FileExistsError(f"Frozen artifact already exists with different content: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze Phase-1 models from preregistered Gate screening.")
    parser.add_argument("--screening", type=Path, default=Path("outputs/gate_screening/summary.json"))
    parser.add_argument("--gate-protocol", type=Path, default=Path("configs/GATE_DIAGNOSTIC_PROTOCOL.json"))
    parser.add_argument("--output", type=Path, default=Path("configs/PHASE1_FROZEN_PROTOCOL.json"))
    parser.add_argument("--definitions", type=Path, default=Path("docs/PHASE1_MODEL_DEFINITIONS.md"))
    args = parser.parse_args()
    protocol_hash = sha256(args.gate_protocol)
    payload = build_frozen_protocol(read_json(args.screening), read_json(args.gate_protocol), protocol_hash)
    write_once_or_verify(args.output, canonical_bytes(payload))
    write_once_or_verify(args.definitions, model_definitions(payload).encode("utf-8"))
    print(json.dumps({"protocol": str(args.output), "sha256": sha256(args.output), "definitions": str(args.definitions)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
