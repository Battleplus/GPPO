# PCRL-v0 hard-3 preference, reward and metric definitions

## Preference semantics

The user supplies a nominal four-way task preference

\[
 p^{task}=(p_s,p_r,p_a,p_h),\qquad p_i\ge0,\quad\sum_i p_i=1.
\]

Hard-3 resolves the policy-conditioning allocation target by suite:

- primary `controllability_phase`: the normalized nominal vector itself;
- secondary `chain_compatibility`: its projection onto
  `reconnaissance <= 2 * search`, `strike <= reconnaissance`, and
  `recovery <= strike`.

The projection describes aggregate feasibility of the original S-R-A-H-S
chain, whereas the phase-staggered generator rotates chain phases and is
designed to test the nominal scheduling request directly. Hard-3 does not
change the seven-objective mapping, the `0.05` task floor, or the fixed
efficiency/communication/safety weights. An omitted target mode retains the
hard-2 dependency-feasible default.

In either suite, the selected target denotes scheduling-priority mass rather than final completion
proportions. A conditional policy receives the seven-dimensional optimization
preference and the four-dimensional closed-loop deficit between target and
current priority mass. The no-conditioning ablation receives neither variable
preference information nor a dynamic deficit.

## Hard-3 priority mass

For task `j` of type `i`, let `t_j` be its first accepted assignment time and
let `D` be the episode deadline. If `t_j <= D`, its unique hard-3 priority
credit is

\[
 c_j=\exp\!\left(-\lambda t_j/\max(1,D)\right),\qquad \lambda=2.0.
\]

Repeated assignments and reallocations of the same task add no further credit.
For type `i`, with `A_i` tasks observed as available,

\[
 q_i=\frac{\sum_{j\in i}c_j}{\max(A_i,1)},\qquad
 m_i=\frac{q_i}{\sum_k q_k}.
\]

`m` is `priority_weighted_assignment_mix`. `preference_l1`, `preference_l2`,
and `preference_cosine` compare `m` with the suite-selected target. The
`*_nominal` fields always compare with the user request, while
`*_dependency_feasible` fields always compare with its chain-feasible
projection. Thus primary-suite `preference_l1` equals
`preference_l1_nominal`; compatibility-suite `preference_l1` equals
`preference_l1_dependency_feasible`.

The coefficient is part of the protocol, not a learned hyperparameter. Hard-2
uses `lambda=1.0`; hard-3 uses `lambda=2.0`. The two versions therefore define
different reward scales and must be trained and summarized independently.

With four task types, `lambda=2.0` gives the idealized one-early/three-at-
deadline upper share `1/(1+3e^-2) ~= 0.711`, making the `0.70` anchor compatible
with the kernel range. This bound only removes a known algebraic contradiction;
it does not waive empirical monotonicity, efficiency or anti-gaming checks.

## Seven-dimensional reward

The transition reward remains

\[
 \mathbf r_t=[\Delta q_s,\Delta q_r,\Delta q_a,\Delta q_h,
 \tilde r_{makespan},-\tilde c_{comm},-\tilde c_{safety}].
\]

Only the `q_i` assignment-priority increments change through `lambda=2.0`.
Makespan improvement, communication and safety normalization are unchanged.
PCRL retains the vector reward; scalar controls expose the dot product with the
seven-dimensional optimization preference.

## Required diagnostics

The primary scheduling-priority metric is insufficient by itself. Every result
also reports:

- unweighted unique-assignment mix and processing-effort mix;
- deadline completion and per-type completion coverage;
- reachable- and availability-normalized completion;
- minimum required-task coverage;
- makespan, communication events, invalid actions and reallocation success;
- suite target mode, selected allocation target, and nominal and
  dependency-feasible preference errors.

A result cannot claim controllability if it improves priority L1 by abandoning
a necessary task type, shrinking availability, or violating the frozen action
mask. Pareto hypervolume, IGD and preference regret use the same seven-quality
normalization as hard-2, but are computed only within hard-3 and separately per
evaluation suite.

## Evidence boundary

The PCRL paper supports preference-conditioned policies, vector critics,
continuous preference sampling, MOO/PreCo updates, unseen-preference tests and
multi-seed evaluation. The decay coefficient, dependency projection,
phase-staggered generator, weak-communication cache and deadline are explicit
project engineering choices. Hard-3 is not a numerical reproduction of an
unreleased paper environment, and world-model integration remains out of scope.
