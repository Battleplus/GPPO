# Archive Inventory and Retention Policy

This document identifies the files required to understand, reproduce, audit,
or continue the GPPO-v2 and PCRL-v0 work. It separates versioned evidence from
large local evidence and from superseded prototypes. Presence in the archive
does not imply that a method passed acceptance.

## Authoritative files in Git

### Project and environment

- `README.md`: scope, status, results, installation and reproduction commands.
- `pyproject.toml`: package metadata and direct dependency constraints.
- `requirements.txt`: minimal pip-compatible dependency list.
- `uv.lock`: exact cross-platform dependency resolution for the current
  `uav-gppo-stage1` package.
- `.gitattributes`: text/binary policy and Git LFS policy for PCRL checkpoints.
- `.gitignore`: generated-output and local-environment exclusions.

### GPPO-v2 implementation and protocol

- `src/uav_assignment/gppo_v2.py`
- `src/uav_assignment/paper_env.py`
- `src/uav_assignment/paper_models.py`
- `configs/gppo_v2_hard.json`
- `run_paper_gppo_v2.py`
- `train_paper_gppo.py`
- `evaluate_paper_gppo.py`
- `summarize_gppo_v2.py`
- `analyze_gppo_v2.py`
- `analyze_gppo_v2_acceptance.py`
- `audit_gppo_v2_formal.py`
- `audit_gppo_v2_parameters.py`

### PCRL-v0 implementation and protocol

- `src/uav_assignment/pcrl_models.py`
- `src/uav_assignment/pcrl_training.py`
- `src/uav_assignment/pcrl_v0.py`
- `src/uav_assignment/preco.py`
- `src/uav_assignment/pcrl_oracle.py`
- `configs/pcrl_v0.json`
- `configs/pcrl_v0_frozen_checkpoints.json`
- `configs/pcrl_v0_hard3.json` through `configs/pcrl_v0_hard6.json`
- `train_pcrl_v0.py`
- `evaluate_pcrl_v0.py`
- `evaluate_pcrl_oracle.py`
- `run_pcrl_v0.py`
- `summarize_pcrl_v0.py`
- `analyze_pcrl_attainability.py`
- `analyze_pcrl_headroom.py`
- `analyze_pcrl_hard5_calibration.py`
- `analyze_pcrl_hard6_calibration.py`
- `normalize_pcrl_evaluation_metadata.py`

### Reports and acceptance records

- `docs/GPPO_V2_ALIGNMENT.md`: public formula to implementation mapping.
- `docs/GPPO_V2_METRICS.md`: metric definitions.
- `docs/GPPO_V2_RESULTS.md`: frozen five-seed results.
- `docs/GPPO_V2_ACCEPTANCE.md`: first-stage acceptance decision.
- `docs/PCRL_BASELINE_PROTOCOL_HARD6.md`: current PCRL screen protocol.
- `docs/PCRL_V0_OBJECTIVES_HARD6.md`: objective and preference definitions.
- `docs/PCRL_V0_ACCEPTANCE_PLAN_HARD6.md`: preregistered gates.
- `docs/PCRL_HARD5_GAIN_SCREEN.md`: hard-5 diagnostic conclusion.
- `docs/PCRL_V0_PRELIMINARY_CONCLUSION.md`: earlier negative result record.
- All other files under `docs/` retain earlier protocol and calibration history.

### Accepted GPPO artifacts

- `artifacts/checkpoints/gppo_event/seed_1` through `seed_5`.
- `artifacts/checkpoints/gppo_event_single_head/seed_1` through `seed_5`.
- `artifacts/summary/`: aggregate CSV/JSON, parameter and artifact audits,
  comparison tables, training curves and validation curves.

These ten checkpoints are the frozen first-stage checkpoints. Their inclusion
does not erase the documented negative ablations.

### PCRL hard-6 artifacts

- `artifacts/pcrl_hard6/summary/`: compact three-seed manifests and aggregate
  calibration decision.
- `artifacts/pcrl_hard6/runs/calibration20/seed_11_run`
- `artifacts/pcrl_hard6/runs/calibration20/seed_12_run`
- `artifacts/pcrl_hard6/runs/calibration20/seed_13_run`

Each seed archive contains:

- the run manifest;
- raw evaluation CSV and JSON for frozen GPPO, conditioned PCRL and true
  no-conditioning PCRL;
- the final conditioned and no-conditioning checkpoints;
- training and validation histories;
- the candidate index containing the saved-candidate hashes.

The six final PCRL checkpoints are stored with Git LFS. Intermediate validation
checkpoint tensors are omitted because this was a calibration screen, not
formal evidence; their hashes remain in the candidate indexes and external
archive manifest.

### Tests

All files under `tests/` are retained. They cover GPPO environment and graph
alignment, hard-3 through hard-6 protocol contracts, PCRL conditioning,
preference sampling, PreCo behavior, Oracle diagnostics, evaluation metadata,
and result summarization.

## Checksummed external archive

`artifacts/manifests/repository_sha256.csv` lists the 162 repository files in
the archive snapshot (excluding the manifest itself), with byte size and
SHA-256. For LFS files, the hash describes the materialized checkpoint content,
not the small Git pointer.

`artifacts/manifests/external_archive_sha256.csv` provides one row per retained
external file with category, retention status, relative path, byte size and
SHA-256. It currently covers:

| Category | Files | Approximate size | Git status |
| --- | ---: | ---: | --- |
| GPPO-v2 formal raw runs and event logs | 259 | 409.25 MB | Local, checksummed |
| PCRL hard-6 intermediate candidate tensors | 66 | 53.29 MB | Local, checksummed |
| Superseded GPPO/world-model prototype code | 21 | 0.15 MB | Local, checksummed |
| Two supplied reference-paper PDFs | 2 | 9.26 MB | Local, checksummed; copyright |

The external GPPO files remain under
`outputs/paper_aligned/gppo_v2_hard/formal`. The intermediate PCRL candidates
remain under `outputs/pcrl_v0/hard6/calibration20`. They are deliberately not
placed in ordinary Git history.

The legacy prototype includes the early `world_model.py`, `train_world_model.py`,
basic environment/PPO modules and their tests. It is retained for research
history only. It is not the current GPPO-v2/PCRL-v0 implementation, and it must
not be cited as completed world-model integration.

## Reference literature

The two supplied PDFs are checksummed but not redistributed through this
repository:

- Yu et al., *Multi-UAV Dynamic Task Assignment Based on Event-Triggered Graph
  Reinforcement Learning Under Weak Communication*, IEEE TASE, 2025.
- *Preference Controllable Reinforcement Learning with Advanced Multi-Objective
  Optimization*.

Future work also depends on *From Observations to Events: Event-Aware World
Model for Reinforcement Learning* and JEPA literature. Those references should
be added to the external archive when stable source files and citation metadata
are selected. Their mention does not unlock world-model implementation.

## Deliberate exclusions

The following are not research records and should not be archived:

- `.venv/`, `venv/`, `__pycache__/` and `.pytest_cache/`;
- smoke-only outputs used solely to check that a command runs;
- editor settings, OS metadata and temporary logs;
- duplicate intermediate experiments superseded by a documented hard protocol.

## Integrity and retrieval

Install Git LFS before cloning or pulling checkpoints:

```bash
git lfs install
git lfs pull
```

Verify an external file against the manifest with PowerShell:

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath <path>
```

The PCRL archive remains a negative calibration result: the primary L1
improvement was 16.26%, below the fixed 30% gate, and the relative invalid-action
guard failed. `pilot20`, `formal100`, JEPA and world-model integration remain
locked.
