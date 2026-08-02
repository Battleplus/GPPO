# PCRL-v0 hard-4 baseline protocol

## Status and version boundary

`pcrl-v0-hard-4` is a new engineering protocol. It is not a relabeling of
hard-3, and no hard-2 or hard-3 checkpoint, evaluation row, confidence
interval or summary may be presented as hard-4 evidence. Hard-4 changes the
formal preference family to a calibrated `priority_share=0.40` family while
retaining the hard-3 assignment-priority decay `2.0` and suite-specific target
semantics.

Artifacts use the isolated namespace `outputs/pcrl_v0/hard4/`. Pilot and
formal artifacts are further isolated under `pilot20/` and `formal100/`.
Neither group may initialize the other, and their rows are never pooled.

The machine-readable contract is
[`configs/pcrl_v0_hard4.json`](../configs/pcrl_v0_hard4.json). Hard-2 and
hard-3 configuration and documentation remain historical records and are not
modified by this protocol.

## Frozen GPPO reference

Hard-4 retains `gppo-v2-hard-3`, its scenarios, action masks, weak-
communication cache, task generators and frozen GPPO checkpoints. The frozen
implementation SHA-256 is
`a62c17f721688e2ae0c36c6fe11ef1a6cced365c8468155bfacbf4f286ea01a6`.
Checkpoint paths and hashes come from
`configs/pcrl_v0_frozen_checkpoints.json`.

Frozen GPPO is re-evaluated against hard-4 preferences; it is not retrained.
PCRL and its no-conditioning ablation require fresh hard-4 training. A hard-2
or hard-3 PCRL checkpoint cannot serve as a hard-4 result.

## Frozen preference family

For a single-priority profile, the priority task receives `0.40` and each
other task receives `0.20`. The eight formal profiles are:

| profile | task preference |
| --- | --- |
| `balanced` | `(0.25, 0.25, 0.25, 0.25)` |
| `calibration_search_priority` | `(0.40, 0.20, 0.20, 0.20)` |
| `calibration_reconnaissance_priority` | `(0.20, 0.40, 0.20, 0.20)` |
| `calibration_strike_priority` | `(0.20, 0.20, 0.40, 0.20)` |
| `calibration_recovery_priority` | `(0.20, 0.20, 0.20, 0.40)` |
| `calibration_search_strike_interp` | `(0.30, 0.20, 0.30, 0.20)` |
| `calibration_recon_recovery_interp` | `(0.20, 0.30, 0.20, 0.30)` |
| `calibration_asymmetric_search_recon_recovery` | `(0.30, 0.26, 0.20, 0.24)` |

The first five are formal anchors. The final three are pre-registered convex
interpolations and are not inserted as named training anchors. Continuous
hard-4 training preferences must remain inside the convex hull of the four
single-priority anchors. The historical unbounded hard-2/hard-3 sampler is not
part of the hard-4 contract.

## Suites and budgets

The primary suite remains `controllability_phase`: `phase_staggered`, deadline
scale `0.70`, nominal target. The secondary `chain_compatibility` suite uses
the original chain generator, deadline scale `1.0`, and the dependency-
feasible target. Suites are summarized separately.

`pilot20` trains five seeds for 20 updates with 18 episodes per update and
evaluates 20 episodes per cell on seeds `76000..76019`. It is a go/no-go
engineering artifact, not formal evidence. `formal100` trains fresh policies
for 100 updates with 18 episodes per update and evaluates 100 episodes per
cell on seeds `50000..50099`.

## Run identity

A conforming checkpoint, evaluation row and run manifest records the protocol
and artifact group, protocol/config hashes, scenario identity, frozen GPPO
implementation and source-checkpoint hashes, PCRL implementation hash, method
and conditioning identity, training/evaluation seeds, suite and scale, budget,
decay, target mode, profile-family id, priority/background shares, and exact
profile name/vector. The complete required-field list and rejection rules are
in `run_identity` in the machine-readable configuration.

Missing identity fields make an artifact non-conforming. In particular,
`protocol_version=pcrl-v0-hard-4`, `priority_share=0.40`, decay `2.0`, the
exact eight-profile family, and a known `pilot20` or `formal100` group are
mandatory. No world model, JEPA encoder or learned event trigger is in scope.
