# Phase 1 模型定义

本文件由 Gate 三种子筛选结果生成。模型选择只使用 validation-A；validation-B 和 test100 仅用于确认或否证。

## GPPO-Literal

作用：paper-compatible interpretation; retained even if negative。

```json
{
  "graph_mode": "literal",
  "gate_scope": "task_message",
  "gate_activation": "sigmoid",
  "gate_bias_init": 0.0,
  "gate_warmup_iterations": 0,
  "rrelu_mode": "expected",
  "post_gate_renormalization": false,
  "role": "paper-compatible interpretation; retained even if negative"
}
```

## GPPO-Best

作用：validation-selected engineering baseline。

```json
{
  "graph_mode": "literal",
  "gate_scope": "score",
  "gate_activation": "sigmoid",
  "gate_bias_init": 0.0,
  "gate_warmup_iterations": 0,
  "selected_variant": "Adaptive-score",
  "role": "validation-selected engineering baseline"
}
```

## PPO-MLP

作用：ordinary PPO without graph encoder。

```json
{
  "graph_mode": "ppo_mlp",
  "role": "ordinary PPO without graph encoder"
}
```

## NoGate

作用：adaptive-gate ablation。

```json
{
  "graph_mode": "literal_no_gate",
  "gate_scope": "task_message",
  "gate_activation": "sigmoid",
  "gate_bias_init": 0.0,
  "gate_warmup_iterations": 0,
  "role": "adaptive-gate ablation"
}
```

## SingleHead

作用：simple-attention structural control。

```json
{
  "graph_mode": "literal_single_head",
  "gate_scope": "task_message",
  "gate_activation": "sigmoid",
  "gate_bias_init": 0.0,
  "gate_warmup_iterations": 0,
  "role": "simple-attention structural control"
}
```

## 冻结边界

正式测试结果读取后，不得改变模型结构、Gate 作用域、激活函数、初始化、warmup 或 PPO 超参数。
Literal 版本及全部负消融结果必须保留；GPPO-Best 不替代 Literal 的论文兼容解释。
