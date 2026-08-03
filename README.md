# GPPO-v2 and PCRL-v0 for Multi-UAV Task Assignment

This repository contains the accepted first-stage engineering baseline for
multi-UAV dynamic task assignment under weak communication. It is an independent
mechanism-level implementation of Yu et al., *Multi-UAV Dynamic Task Assignment
Based on Event-Triggered Graph Reinforcement Learning Under Weak Communication*
(IEEE TASE, 2025).

The `pcrl-hard6-progress` branch also contains the current second-stage
preference-conditioned reinforcement-learning work. PCRL is still an
engineering prototype and has **not** passed its preregistered controllability
gate. JEPA and world-model integration remain intentionally locked.

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

## PCRL-v0 progress

The frozen `gppo-v2-hard-3` implementation and checkpoints are retained as the
comparison baseline. Hard-6 adds:

- raw four-dimensional task preference conditioning for the actor and mapped
  seven-dimensional objective conditioning for the critic/PreCo path;
- an explicit legal five-group mass controller for Search, Reconnaissance,
  Strike, Recovery and Noop;
- a bounded state-dependent calibration head that preserves frozen GPPO's
  assignment/noop rhythm and within-group action ranking;
- four complete trajectories per sampled preference and full-batch cached
  PreCo directions before PPO minibatch shuffling;
- deficit-aware task-mass auxiliary loss, balanced capability coverage and
  unchanged action masks/weak-communication cache;
- anchor-only checkpoint selection, relative invalid-action guards and every
  validation candidate saved with SHA-256.

The three-seed hard-6 `calibration20` screen is complete:

| Gate | Result |
| --- | ---: |
| Primary-five L1 reduction vs true no-conditioning | 16.26%, 3/3 positive seeds |
| Primary 95% CI | [-3.34%, 35.85%] |
| Held-out-three L1 reduction | 19.81%, 3/3 positive seeds |
| Four priority directions | all positive in 3/3 seeds |
| Deadline delta vs GPPO | -0.01843 |
| Makespan increase vs GPPO | +2.00% |
| Minimum coverage delta vs GPPO | -0.01677 |
| Invalid-action delta vs GPPO | +0.05521 |

The required 30% primary improvement was not reached and the mean relative
invalid-action delta exceeded the `+0.05` calibration guard. Therefore
`pilot20`, `formal100`, JEPA and the world model remain locked. Negative and
uncertain results are preserved rather than relaxed after inspection.

See:

- [`docs/PCRL_BASELINE_PROTOCOL_HARD6.md`](docs/PCRL_BASELINE_PROTOCOL_HARD6.md);
- [`docs/PCRL_V0_OBJECTIVES_HARD6.md`](docs/PCRL_V0_OBJECTIVES_HARD6.md);
- [`docs/PCRL_V0_ACCEPTANCE_PLAN_HARD6.md`](docs/PCRL_V0_ACCEPTANCE_PLAN_HARD6.md);
- [`docs/PCRL_HARD5_GAIN_SCREEN.md`](docs/PCRL_HARD5_GAIN_SCREEN.md);
- [`artifacts/pcrl_hard6/summary/PCRL_HARD6_CALIBRATION.md`](artifacts/pcrl_hard6/summary/PCRL_HARD6_CALIBRATION.md);
- [`artifacts/pcrl_hard6/summary/summary.json`](artifacts/pcrl_hard6/summary/summary.json).

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

Install Git LFS and materialize archived model files before running experiments:

```bash
git lfs install
git lfs pull
```

For an exact dependency resolution with `uv`:

```bash
uv sync --locked
```

Alternatively, install the minimal dependency set with pip:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Run all GPPO/PCRL tests:

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

PCRL entry points are:

```bash
python train_pcrl_v0.py --help
python evaluate_pcrl_v0.py --help
python run_pcrl_v0.py --help
python analyze_pcrl_hard6_calibration.py --help
```

PCRL verifies the accepted checkpoints against their frozen hashes. The branch
stores those checkpoints under `artifacts/checkpoints`; before running PCRL,
materialize the paths used by the frozen protocol. On PowerShell:

```powershell
$base = 'outputs/paper_aligned/gppo_v2_hard/formal/train'
foreach ($method in @('gppo_event', 'gppo_event_single_head')) {
  New-Item -ItemType Directory -Force -Path "$base/$method" | Out-Null
  Copy-Item -Recurse -Force "artifacts/checkpoints/$method/seed_*" "$base/$method/"
}
```

Run the hard-6 smoke workflow:

```bash
python run_pcrl_v0.py --protocol configs/pcrl_v0_hard6.json \
  --artifact-group calibration20 --phase all --smoke \
  --methods pcrl_gppo_adaptive pcrl_gppo_adaptive_no_conditioning gppo_event \
  --seeds 11 --output-root outputs/pcrl_v0/hard6/smoke
```

The full three-seed calibration uses the same command without `--smoke` and
with `--seeds 11 12 13`. It is screening evidence, not formal evidence.

The accepted adaptive and single-head five-seed GPPO checkpoints are included
under [`artifacts/checkpoints`](artifacts/checkpoints). Summary CSV/JSON files
and rendered curves are included under [`artifacts/summary`](artifacts/summary).

The hard-6 archive additionally contains the six final PCRL/no-conditioning
checkpoints, all three-seed raw evaluation CSV/JSON files, training and
validation histories, run manifests and candidate indexes under
[`artifacts/pcrl_hard6/runs`](artifacts/pcrl_hard6/runs). PCRL checkpoint tensors
are stored with Git LFS.

The 409.25 MB GPPO formal raw event archive and the 66 intermediate PCRL
validation checkpoint tensors are intentionally excluded from ordinary Git
history. Every retained external file is listed with size and SHA-256 in
[`artifacts/manifests/external_archive_sha256.csv`](artifacts/manifests/external_archive_sha256.csv).
The 162-file repository snapshot is independently indexed in
[`artifacts/manifests/repository_sha256.csv`](artifacts/manifests/repository_sha256.csv).
See [`docs/ARCHIVE_INVENTORY.md`](docs/ARCHIVE_INVENTORY.md) for the authoritative
file inventory, retention policy, reference-paper record and explicit
exclusions.

## Repository structure

```text
configs/                 frozen scenario manifest
docs/                    alignment, metrics, calibration and acceptance reports
src/uav_assignment/      environment, graph model and frozen protocol helpers
tests/                   first-stage regression and protocol tests
artifacts/checkpoints/   accepted adaptive and single-head checkpoints
artifacts/summary/       formal aggregate results, audits and curves
artifacts/pcrl_hard6/    hard-6 manifests, raw evaluations and final checkpoints
artifacts/manifests/     SHA-256 inventory for large external evidence
uv.lock                  exact dependency resolution
```
