# PCRL-v0 hard-3 acceptance and statistical plan

## Current execution status

The pre-training anchor-strength screen did not identify a practically
meaningful candidate with the required 40% frozen-mask Oracle headroom. The
best eligible candidate (`priority_share=0.40`) achieved 30.04% point headroom
with a three-eval-seed paired bootstrap interval of 16.70% to 41.01%.
Accordingly, formal hard-3 PCRL and no-conditioning training have not started.
See [`PCRL_V0_ANCHOR_CALIBRATION_RESULT.md`](PCRL_V0_ANCHOR_CALIBRATION_RESULT.md).

## Pre-registration boundary

This plan applies only to `pcrl-v0-hard-3` with
`assignment_priority_decay=2.0`. Hard-2 results remain historical exploratory
evidence and are not pooled into hard-3 estimates, confidence intervals,
Pareto fronts or acceptance decisions. The previously stated 30% confirmatory
gate is unchanged, but its target is suite-specific: nominal in the primary
phase-staggered suite and dependency-feasible in the original-chain
compatibility suite.

## Smoke gates

Before formal training, verify:

1. `lambda=1.0` exactly reproduces hard-2 vector rewards and metrics;
2. hard-3 records `lambda=2.0` in checkpoints, evaluation rows and manifests;
3. first unique assignment credit equals `exp(-2t/max(1,D))`;
4. repeated/reallocation assignment adds zero task-priority credit;
5. invalid, negative or non-finite decay coefficients are rejected;
6. preference deficit and vector reward use the same configured coefficient;
7. action masks, weak-communication state and the frozen GPPO hash are unchanged;
8. the five anchors, three held-out profiles and runtime preference switch work;
9. omitted target mode reproduces the hard-2 dependency-feasible target;
10. phase evaluation records `nominal`, chain evaluation records
    `dependency_feasible_projection`, and their deficits use those targets.

A smoke artifact missing protocol version or coefficient is invalid even when
its numerical tests pass.

## Formal design

Train every learned method for `100` updates and `18` episodes per update using
seeds `1..5`. Evaluate `100` episodes for every method, four scale, five anchor
profiles and three held-out profiles in each suite:

- primary `controllability_phase`: `phase_staggered`, deadline scale `0.70`,
  nominal target;
- secondary `chain_compatibility`: `chain`, deadline scale `1.0`,
  dependency-feasible target.

Frozen GPPO controls are evaluated under the hard-3 metric without retraining.
PCRL policies must be newly trained with hard-3 rewards; a hard-2 PCRL
checkpoint cannot serve as a hard-3 learned result.

## Statistical analysis

Aggregate episodes within training seed before computing five-seed 95% Student
t intervals. Report paired PCRL-minus-no-conditioning and PCRL-minus-GPPO
differences, their intervals, per-seed gate status and Holm-adjusted p-values.
An interval crossing zero is inconclusive. Suites and protocol versions are
never pooled.

The primary error is hard-3 `preference_l1` against the nominal target in
`controllability_phase`. In `chain_compatibility`, `preference_l1` uses the
dependency-feasible target. Every suite also reports both invariant diagnostic
families (`*_nominal` and `*_dependency_feasible`), per-profile response,
held-out interpolation, preference regret, hypervolume and IGD. Regret and
Pareto reference fronts are empirical within the same suite and protocol.

## Acceptance gates

Hard-3 is accepted only if all primary-suite gates pass:

- preference L1 is at least 30% lower than true no-conditioning;
- all four priority directions respond monotonically;
- held-out interpolation remains controllable;
- macro deadline completion is no more than `0.03` below frozen GPPO-event;
- macro makespan is no more than 5% above frozen GPPO-event;
- at least 4/5 training seeds satisfy paired controllability and efficiency;
- minimum required-task coverage loss is at most `0.05`;
- no task abandonment, availability manipulation, action-mask mismatch or
  protocol/coefficient/target-mode mismatch is present.

The coefficient calibration makes the `0.70` anchor algebraically compatible
with the metric range; it does not relax any acceptance threshold. Negative or
inconclusive hard-3 results must be retained. World-model work does not begin
until these PCRL controllability gates are actually demonstrated.
