# GPPO-v2: Event-Triggered Graph PPO for Multi-UAV Task Assignment

This repository contains the accepted first-stage engineering baseline for
multi-UAV dynamic task assignment under weak communication. It is an independent
mechanism-level implementation of Yu et al., *Multi-UAV Dynamic Task Assignment
Based on Event-Triggered Graph Reinforcement Learning Under Weak Communication*
(IEEE TASE, 2025).

The original authors did not publish their complete environment, instance
generator, communication simulator, or training code. Consequently, this work
does **not** claim numerical reproduction of the paper's tables. It reproduces
the public formulas and core mechanisms, then evaluates them with a frozen,
audited engineering protocol.

## Acceptance status

GPPO-v2 passed the frozen first-stage engineering acceptance protocol:

- protocol version: `gppo-v2-hard-3`;
- implementation hash: `a62c17f721688e2ae0c36c6fe11ef1a6cced365c8468155bfacbf4f286ea01a6`;
- 8 learned methods x 5 training seeds;
- 4 evaluation scales x 100 fixed evaluation episodes;
- 40 checkpoints, 42 evaluations, 16,800 evaluation rows;
- 42 non-empty event logs containing 935,138 events;
- 92 tests passed in the accepted workspace.

Macro results from the frozen evaluation:

| Method | Deadline completion rate | Makespan |
| --- | ---: | ---: |
| Random-event | 0.72461 | 21.9621 |
| Greedy-event | 0.78071 | 20.4467 |
| GPPO-event | 0.77738 | 20.2619 |
| Always/full-state engineering upper bound | 0.77890 | 20.1806 |

GPPO-event is stable across 5/5 seeds, clearly better than Random, and close to
Greedy while using about half as many synchronization events as the Always
control. Negative results are retained: single-head attention has higher macro
deadline completion than the adaptive model, and the no-gate comparison does
not establish an adaptive-gate advantage.

See:

- [`docs/GPPO_V2_RESULTS.md`](docs/GPPO_V2_RESULTS.md) for the complete report;
- [`docs/GPPO_V2_ALIGNMENT.md`](docs/GPPO_V2_ALIGNMENT.md) for formula-to-code evidence;
- [`docs/GPPO_V2_ACCEPTANCE.md`](docs/GPPO_V2_ACCEPTANCE.md) for the acceptance decision;
- [`artifacts/summary/acceptance_analysis.json`](artifacts/summary/acceptance_analysis.json) for machine-readable gates.

## Implemented mechanisms

- UAV/task heterogeneous graph with capability and precedence relations;
- AHGNN-style UAV attention and multi-relation task update;
- dynamic `(task, UAV)` action mask and noop action;
- makespan-difference reward `R_t = M_(t-1) - M_t`;
- event-driven asynchronous execution and task reallocation;
- stale belief cache, partial synchronization, heartbeat timeout and leader
  replacement for weak communication;
- structured event logs and controlled `none/event/periodic/always` communication
  modes.

The belief cache, timeout rules, continuous-time hazards, fixed tensor capacity,
padding/noop action, mission deadlines and the four communication controls are
engineering extensions, not uniquely specified by the paper.

## Installation and tests

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Run the first-stage tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

On PowerShell:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest -q
```

## Reproduction

The frozen manifest is [`configs/gppo_v2_hard.json`](configs/gppo_v2_hard.json).
The orchestration entry point is:

```bash
python run_paper_gppo_v2.py --help
```

Individual training and evaluation entry points are:

```bash
python train_paper_gppo.py --help
python evaluate_paper_gppo.py --help
python summarize_gppo_v2.py --help
```

The accepted adaptive and single-head five-seed checkpoints are included under
[`artifacts/checkpoints`](artifacts/checkpoints). Summary CSV/JSON files and
rendered curves are included under [`artifacts/summary`](artifacts/summary).
Large raw evaluation JSON files and event logs are intentionally excluded from
Git because of repository-size limits; their hashes and audit conclusions are
preserved in the summary artifacts and reports.

## Repository structure

```text
configs/                 frozen scenario manifest
docs/                    alignment, metrics, calibration and acceptance reports
src/uav_assignment/      environment, graph model and frozen protocol helpers
tests/                   first-stage regression and protocol tests
artifacts/checkpoints/   accepted adaptive and single-head checkpoints
artifacts/summary/       formal aggregate results, audits and curves
```

PCRL and world-model development are intentionally outside this first-stage
submission.
