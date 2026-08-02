# PCRL-v0 baseline protocol

## Status and scope

This document freezes the comparison protocol for `pcrl-v0-hard-2`. The accepted GPPO-v2 implementation and artifacts remain read-only. PCRL-v0 is a preference-conditioned extension implemented only in new wrapper/model/training files. No world model, JEPA encoder, or learned event trigger is trained in this stage; the model interface accepts an inert `event_input` payload for future integration.

The protocol is an independent, mechanism-level reproduction. It must not be reported as numerical reproduction of the source paper because the paper does not release its environment, task generator, communication simulator, or training code.

## Frozen GPPO reference

| Item | Frozen value |
| --- | --- |
| scenario | `gppo-v2-hard-3` |
| scenario file | [`configs/gppo_v2_hard.json`](../configs/gppo_v2_hard.json) |
| environment/model/training implementation hash | `a62c17f721688e2ae0c36c6fe11ef1a6cced365c8468155bfacbf4f286ea01a6` |
| train scales | `3x16`, `3x20`, `4x24` |
| evaluation scales | `2x12`, `3x16`, `3x20`, `4x24` |
| evaluation seeds | `50000..50099` (100 episodes per scale) |
| learned seeds | `1..5` |
| updates / episodes per update | `100 / 18` |
| graph-aligned reference | `gppo_event` (adaptive) |
| stronger engineering control | `gppo_event_single_head` |

Formal GPPO evidence is in [`docs/GPPO_V2_ACCEPTANCE.md`](GPPO_V2_ACCEPTANCE.md), [`docs/GPPO_V2_RESULTS.md`](GPPO_V2_RESULTS.md), and `outputs/paper_aligned/gppo_v2_hard/formal/`.
Exact source checkpoint SHA-256 values are frozen in [`configs/pcrl_v0_frozen_checkpoints.json`](../configs/pcrl_v0_frozen_checkpoints.json).

The checkpoint paths below are the only GPPO initialization sources allowed for formal PCRL runs:

- `outputs/paper_aligned/gppo_v2_hard/formal/train/gppo_event/seed_{1..5}/checkpoint.pt`
- `outputs/paper_aligned/gppo_v2_hard/formal/train/gppo_event_single_head/seed_{1..5}/checkpoint.pt`
- `outputs/paper_aligned/gppo_v2_hard/formal/train/ppo_event/seed_{1..5}/checkpoint.pt` for the scalarized PPO control

Each checkpoint must report the frozen implementation hash. PCRL initialization rejects a changed hash, a non-event synchronization mode, or a graph-mode mismatch.

## Task-release and deadline suites

PCRL-v0 separates controllability from compatibility instead of pooling two materially different task generators:

| suite | task release | deadline | role |
| --- | --- | --- | --- |
| `controllability_phase` | `phase_staggered` rolling chains | `0.70` times the frozen scale deadline | primary training and preference-controllability evaluation |
| `chain_compatibility` | original synchronized `chain` lifecycle | unchanged frozen scale deadline | evaluation-only compatibility and efficiency audit |

The `phase_staggered` generator changes only task-chain phase offsets. It does not change the frozen action mask, event synchronization, weak-communication cache, UAV dynamics, task dependencies, or GPPO implementation. Phase staggering is necessary because synchronized S-R-A-H-S chains expose too few simultaneous legal task types to identify preference control. The stricter `0.70` deadline scale prevents assignment/completion saturation.

All formal PCRL policies are trained on `phase_staggered`. Each resulting checkpoint is evaluated separately on both suites. Frozen GPPO and heuristic controls are evaluated under the same suite-specific generator and deadline as PCRL, but their source checkpoints are never retrained. Results from the two suites must be labeled and summarized separately; the compatibility rows must not be pooled into the primary preference gate.

## PCRL methods

The required comparison set is:

1. `greedy_preference`: deterministic preference-weighted duration/priority heuristic.
2. `gppo_event`: frozen adaptive GPPO, wrapped only for preference metrics.
3. `gppo_event_single_head`: frozen single-head GPPO, wrapped only for preference metrics.
4. `ls_ppo_none_fixed_balanced`: one fixed balanced preference, linear scalarization, PPO graph control.
5. `ls_gppo_adaptive_fixed_balanced`: one fixed balanced preference, linear scalarization, adaptive graph control.
6. `pcrl_gppo_adaptive`: continuous per-episode preferences, ray-similarity/PreCo update, adaptive graph.
7. `pcrl_gppo_single_head`: the same PCRL update with the single-head backbone.
8. `pcrl_gppo_adaptive_no_conditioning`: no-preference-conditioning ablation.

`sdmgrad_gppo_adaptive` is an optional MOO ablation. Every learned method uses five independent seeds and the same scenario, scales, suite-specific deadlines, event synchronization, action mask, and training budget. PCRL samples one continuous four-dimensional task preference per episode, projects it to a dependency-feasible allocation target, maps the target to seven optimization weights, and stores a preference id so minibatch updates group samples by episode preference. The conditional policy also receives the dynamic four-dimensional allocation deficit. The no-conditioning ablation ignores both the preference vector and this deficit.

## Evaluation profiles

Training samples are continuous Dirichlet mixtures with jittered anchors; exact evaluation vectors are not inserted into training. Evaluation uses:

- training anchors: balanced, search, reconnaissance, strike, recovery;
- held-out interpolations: search/strike `(0.40,0.10,0.40,0.10)`, reconnaissance/recovery `(0.10,0.40,0.10,0.40)`, and smooth `(0.35,0.25,0.25,0.15)`.

The primary controllability target is the dependency-feasible projection of the nominal task preference. The primary realized quantity is availability-normalized, priority-weighted unique assignment coverage: the first legal assignment of a task before the deadline contributes `exp(-assignment_time/max(1, deadline))`, and repeated/reallocation assignments do not contribute again. `preference_l1` compares the normalized mix of this coverage against the feasible target. The nominal user vector and `preference_l1_nominal` are retained separately.

Unweighted unique assignments, assigned processing effort, completed-task mix, reachable-normalized completion, minimum type coverage, GPPO deadline completion, makespan, communication events, invalid actions, and reallocation success are mandatory anti-gaming diagnostics.

## Acceptance gates

PCRL-v0 is accepted only when all gates are evaluated on the primary `controllability_phase` suite across the five seeds, including negative and inconclusive outcomes:

- preference L1 error is at least 30% lower than the no-conditioning control;
- each priority profile has the expected monotonic response, and held-out interpolations remain controllable;
- macro deadline completion rate is no more than `0.03` below frozen `gppo_event`;
- macro makespan is no more than 5% above frozen `gppo_event`;
- at least 4/5 seeds satisfy the paired controllability and efficiency gates;
- minimum required-task coverage is not reduced by more than `0.05` versus the frozen reference.

The 30% preference-error threshold remains the frozen confirmatory gate. Preliminary or exploratory evidence may be reported with paired confidence intervals and monotonicity, but it may not silently replace or lower this gate. The `chain_compatibility` suite is reported as a separate transfer/compatibility result and cannot rescue a failed primary gate.

No gate may be passed by making a task type unavailable or by abandoning necessary tasks. Preference regret is reported relative to the best observed method under the same profile/scale and suite (not called an oracle). Pareto hypervolume and IGD are computed separately per suite over the seven maximization objectives after an explicitly documented normalization.

## Reproducibility record

Every PCRL checkpoint and evaluation row records: protocol version, source GPPO checkpoint metadata, frozen GPPO hash, PCRL implementation hash, scenario/config hash, graph mode, algorithm, preference-conditioning flag, training seed, evaluation seed, scale, profile, task-release mode, deadline scale, and switch status. The formal manifest is frozen only after smoke tests pass.
