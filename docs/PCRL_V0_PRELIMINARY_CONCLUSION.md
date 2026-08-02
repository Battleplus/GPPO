# PCRL-v0 preliminary conclusion

## Status

`PCRL-v0` is an engineering implementation and a reproducible preliminary
experiment. It is **not accepted as a paper-grade baseline**. The frozen
confirmatory gates remain unchanged; no world-model or JEPA integration is
authorized until the preference-control protocol is repaired.

## Literature boundary

The Preference Controllable RL paper supports the following choices used here:

- one preference-conditioned policy rather than one policy per preference;
- a vector critic and continuous simplex preference sampling;
- episode/value estimation followed by PreCo or another MOO gradient update;
- unseen-preference evaluation and multiple training seeds.

The event-triggered UAV paper supports the heterogeneous graph, action/task
relationships, event synchronization, weak-communication cache, and dynamic
reassignment framing. It does not provide the environment or training code,
so this project cannot claim numerical reproduction of either paper.

The following are explicitly project extensions: dependency-feasible
preference projection, `phase_staggered` task release, the 0.70 deadline
stress scale, priority-weighted unique assignment coverage,
`preference_deficit`, and the legal-action task-type auxiliary loss.

## Preliminary evidence

The primary five-seed, 20-update run contains 9,600 evaluation rows. Means are
computed per seed before the confidence interval is formed:

| method | preference L1 | deadline completion | makespan |
| --- | ---: | ---: | ---: |
| frozen GPPO-event | 0.411633 | 0.648753 | 17.7238 |
| PCRL adaptive | 0.403491 | 0.650065 | 17.8630 |
| no conditioning | 0.409739 | 0.647664 | 17.7070 |

The paired PCRL minus no-conditioning L1 difference is `-0.006248`, with 95%
CI `[-0.012993, 0.000497]`, exact sign-flip `p=0.125`, and improvement in
4/5 seeds. The relative reduction is only `1.525%`, not the frozen `30%`
gate; all paired efficiency CIs cross zero. The aggregate monotonicity and
held-out interpolation checks pass, but the stable-seed gate is `0/5`.

The separate original `chain` compatibility suite contains another 9,600
rows. PCRL remains compatible with the environment (makespan increase versus
GPPO is `0.608%`), but its L1 reduction versus no-conditioning is only
`1.65%`; it also cannot rescue the primary gate.

## Metric attainability finding

For a task assigned before the deadline,
`exp(-assignment_time / deadline)` lies in `[e^-1, 1]`. If all four task
types are fully assigned by the deadline, the largest possible normalized
share of one type is therefore approximately

```text
1 / (1 + 3 * exp(-1)) = 0.475
```

The acceptance guard also prevents the policy from simply dropping the other
types. With the observed GPPO minimum coverage near `0.492` and an allowed
loss of `0.05`, each non-priority type must retain coverage near `c=0.442`.
Even the optimistic bound is then

```text
1 / (1 + 3 * c * exp(-1)) = 0.673
```

which is still below the search profile's projected target of `0.70`.
Reaching the exact `0.70` anchor would require violating or nearly exhausting
the anti-gaming coverage allowance. This proves an exact-anchor mismatch, but
it does not prove that the frozen 30% relative-improvement gate is impossible
and does not by itself explain the observed 1.5% improvement. In the small
gain sweep, larger explicit gains did not materially improve L1 and were
accompanied by worse makespan.

The audit also found that the original-chain inequalities are not necessary
constraints for `phase_staggered`: rotated chains and dependency-free arrivals
change the aggregate opportunity set. Applying the original S-R-A-H-S
projection in the primary suite substantially distorts the reconnaissance,
strike, and recovery requests. A corrected protocol must use suite-specific
target semantics rather than treating one projection as universally feasible.

## Required next experiment

Do not spend the formal 100-update budget on the current hard-2 metric. First
run a constrained full-state Oracle headroom audit and then a new pilot with
these documented fixes:

1. use the nominal preference target for `phase_staggered` and the original
   dependency projection only for the original `chain` compatibility suite;
2. calibrate the temporal priority decay on disjoint seeds while retaining
   completion, makespan, communication, safety, and per-type coverage gates;
3. require the constrained Oracle to demonstrate at least 40% macro L1
   headroom before asking a learned policy to pass the confirmatory 30% gate.

The original 30% gate must remain attached to the current protocol result and
be reported as failed. A repaired metric requires a new protocol version and a
fresh five-seed pilot before any formal 100-update run. World-model event
prediction remains out of scope until that pilot demonstrates reliable
preference control.
