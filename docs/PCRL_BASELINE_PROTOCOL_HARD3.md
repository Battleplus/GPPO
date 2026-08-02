# PCRL-v0 hard-3 baseline protocol

## Status and version boundary

`pcrl-v0-hard-3` is a separately versioned PCRL controllability protocol. It
changes two coupled protocol quantities relative to `pcrl-v0-hard-2`: the
assignment-priority decay coefficient is `2.0` instead of `1.0`, and the
preference target is selected per evaluation suite. The primary
`phase_staggered` suite uses the nominal user preference; the original `chain`
compatibility suite retains the dependency-feasible projection. The frozen
GPPO reference, task generators, deadlines, methods, seeds, budgets and
acceptance thresholds are unchanged.

Hard-2 checkpoints and evaluation rows are retained as negative preliminary
evidence. They must not be relabeled, pooled with hard-3, or used to initialize
a hard-3 PCRL policy. A hard-3 run uses the namespace
`outputs/pcrl_v0/hard3/` and records both protocol version and decay coefficient
in its checkpoint, evaluation rows and run manifest.

The machine-readable protocol is
[`configs/pcrl_v0_hard3.json`](../configs/pcrl_v0_hard3.json). The hard-2 file
[`configs/pcrl_v0.json`](../configs/pcrl_v0.json) remains unchanged.

## Frozen GPPO reference

| Item | Frozen value |
| --- | --- |
| base scenario | `gppo-v2-hard-3` |
| GPPO implementation hash | `a62c17f721688e2ae0c36c6fe11ef1a6cced365c8468155bfacbf4f286ea01a6` |
| train scales | `3x16`, `3x20`, `4x24` |
| evaluation scales | `2x12`, `3x16`, `3x20`, `4x24` |
| learned seeds | `1..5` |
| evaluation seeds | `50000..50099` |
| updates / episodes per update | `100 / 18` |
| adaptive reference | `gppo_event` |
| competitive reference | `gppo_event_single_head` |

Checkpoint paths and SHA-256 values remain those in
[`configs/pcrl_v0_frozen_checkpoints.json`](../configs/pcrl_v0_frozen_checkpoints.json).
The six frozen GPPO implementation files are not modified by this protocol.

## Hard-3 changes and rationale

### Suite-specific target semantics

Hard-2 projected every preference onto the original chain count constraints
`recon <= 2*search`, `strike <= recon`, and `recovery <= strike`. Those
constraints remain correct for the original S-R-A-H-S chain compatibility
suite. They are not the target set of the primary `phase_staggered` generator:
rotating the starting phase across chains deliberately balances aggregate task
types and exposes them concurrently. Applying the original chain projection in
that suite changes a requested recovery/search/etc. anchor before the policy
acts and therefore weakens the controllability test.

Hard-3 consequently defines:

- `controllability_phase`: `preference_target_mode=nominal`;
- `chain_compatibility`:
  `preference_target_mode=dependency_feasible_projection`.

All hard-3 training uses the primary suite and its nominal target. Evaluation
rows record the selected target mode, nominal vector, selected allocation
target, and errors against both the nominal and dependency-feasible vectors.
Code defaults and `configs/pcrl_v0.json` remain dependency-feasible, so an
unspecified target mode exactly preserves hard-2 behavior.

### Assignment-priority decay

For an assignment made at time `t` under deadline `D`, hard-3 credits

\[
  k_2(t)=\exp\!\left(-2t/\max(1,D)\right).
\]

Hard-2 used `k_1(t)=exp(-t/max(1,D))`. On the interval `0 <= t <= D`, one
task type's largest possible share under the hard-2 kernel, when it is assigned
at `0` and the other three are assigned at `D`, is

\[
  \frac{1}{1+3e^{-1}}\approx0.475.
\]

That algebraically excludes the nominal `0.70` priority anchor before any
policy is trained. With hard-3 the corresponding envelope is

\[
  \frac{1}{1+3e^{-2}}\approx0.711,
\]

so the `0.70` anchor is no longer excluded by the metric's decay range. This
is a metric calibration, not evidence that the environment guarantees every
requested profile in every episode. Task dependencies, simultaneous legal
actions and UAV capability constraints still determine empirical
controllability and remain part of the acceptance audit.

The preference continues to mean *scheduling-priority mass*: earlier unique
assignments receive more mass than later assignments. It is not a requested
final completion proportion, and the four task types are not independent
throughput channels.

## Unchanged suites and comparisons

| suite | release mode | deadline | role |
| --- | --- | --- | --- |
| `controllability_phase` | `phase_staggered` | `0.70` times scale deadline | primary training and controllability |
| `chain_compatibility` | original `chain` | unchanged scale deadline | evaluation-only compatibility |

The corresponding target modes are `nominal` and
`dependency_feasible_projection`, respectively. Results are never pooled
across suites because both the task generator and target semantics differ.

The required methods remain greedy preference, frozen adaptive and single-head
GPPO-event, fixed-weight PPO/GPPO, adaptive and single-head PCRL, and the true
no-conditioning ablation. All learned methods use seeds `1..5`; all methods use
identical suite, profile, scale and evaluation seeds.

## Run identity requirements

A conforming hard-3 artifact records at least:

- `protocol_version = pcrl-v0-hard-3`;
- `assignment_priority_decay = 2.0`;
- frozen GPPO and PCRL implementation hashes;
- scenario/config hash, method, graph mode and conditioning flag;
- training/evaluation seed, scale, profile, suite and deadline scale.
- `preference_target_mode` and the selected allocation target.

If the coefficient is absent, differs from `2.0`, the target mode is absent or
does not match the suite, or a row mixes protocol versions, it is not hard-3
evidence. No world model, JEPA encoder or learned event trigger is in scope.
