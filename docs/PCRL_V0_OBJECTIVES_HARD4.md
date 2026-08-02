# PCRL-v0 hard-4 objectives and metrics

## Preference semantics

Hard-4 retains scheduling-priority mass as the task preference semantics. A
preference controls when unique tasks are first assigned; it is not a request
for final completion-count proportions. The primary phase-staggered suite
compares the realized priority mix directly with the nominal hard-4 profile.
The chain compatibility suite compares with the existing dependency-feasible
projection.

Hard-4 fixes the single-priority share to `a=0.40` and the three background
shares to `b=(1-a)/3=0.20`. Its held-out profiles are convex combinations of
the four single-priority anchors:

- search/strike: `0.5 P_search + 0.5 P_strike`;
- reconnaissance/recovery: `0.5 P_recon + 0.5 P_recovery`;
- asymmetric: `0.5 P_search + 0.3 P_recon + 0.2 P_recovery`.

All vectors and stable names are frozen in `configs/pcrl_v0_hard4.json`.

## Priority mass and reward

For a task first assigned at time `t` under deadline `D`, hard-4 credits

\[
  c(t)=\exp(-2t/\max(1,D)).
\]

Repeated assignments and reallocations do not add task-priority credit. The
four availability-normalized credits are normalized into
`priority_weighted_assignment_mix`. Primary L1 is the sum of absolute
differences between that mix and the suite-selected target.

The seven-dimensional transition reward remains

\[
  [\Delta q_s,\Delta q_r,\Delta q_a,\Delta q_h,
    \tilde r_{makespan},-\tilde c_{comm},-\tilde c_{safety}].
\]

Hard-4 does not change makespan, communication or safety normalization. PCRL
uses vector rewards. The no-conditioning ablation removes only the conditional
inputs at the policy/value-network boundary: its requested preference is
replaced by the fixed balanced default and its `preference_deficit` is omitted
(and treated as zero by the model). It retains the same episode-preference
sampler, vector-reward tapes, preference identifiers and grouped PreCo
multi-objective optimization signal as conditional PCRL. This matched design
isolates the causal contribution of network conditioning without changing the
training distribution or objective signal.

Consequently, `pcrl_gppo_adaptive_no_conditioning` is not a policy trained with
a completely neutral objective. Requested preferences may affect its grouped
training gradient, but they cannot enter its policy or value network and cannot
change its action at evaluation time. Its role is specifically a
conditioning-input ablation; it must not be described as a neutral-objective or
preference-free-training baseline.

## Required anti-gaming diagnostics

Every result reports the priority-weighted mix and L1 together with unweighted
assignment and processing-effort mixes, deadline completion, per-type and
minimum coverage, available/reachable task counts, makespan, communication,
invalid actions and reallocation outcomes. A lower preference error obtained
by abandoning task types, changing availability, using another action mask or
mixing protocol versions is invalid.

Metric, profile and suite identity are inseparable. Rows without exact decay,
target mode, profile vector, priority share and artifact-group identity cannot
enter a hard-4 aggregate.

## Evidence boundary

The 0.40 family, decay, phase-staggered generator, weak-communication cache and
acceptance gates are project engineering choices. Hard-4 is not a numerical
reproduction of an unpublished environment. Hard-3 calibration and Oracle
results motivate the new version but are not hard-4 learned evidence and are
not pooled with it.
