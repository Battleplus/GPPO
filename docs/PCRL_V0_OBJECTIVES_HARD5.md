# PCRL-v0 hard-5 objectives and preference semantics

The seven-dimensional vector reward remains:

`[search, reconnaissance, strike, recovery, makespan efficiency,
communication cost, safety cost]`.

The first four coordinates measure task-type contribution. The final three
retain the GPPO efficiency, communication and safety objectives. The mapped
seven-dimensional preference is used only where the objective dimension must
match the vector critic and PreCo similarity objective.

## Raw task preference versus mapped objective preference

Let `u in simplex(4)` be the user's scheduling request and `w in simplex(7)` be
the protocol mapping used for multi-objective optimization.

- `pi(a | graph, u, deficit)` controls assignments.
- `V(graph, w)` predicts seven objective returns.
- PreCo receives `(w, return_vector)` with equal dimensionality.
- The legal-action alignment loss targets `u`, not `w[:4]`.

This 4D/7D split is a project adaptation. The PCRL paper supports a
preference-conditioned actor, vector critic, per-episode preference sampling,
unseen-preference evaluation and PreCo/MOO updates, but it does not prescribe
this UAV-specific split.

## Capability-controlled identifiability

Preference control is not identifiable when one task type has a structurally
smaller executable action set. Hard-5 therefore balances initial capable-UAV
coverage across all four task types. This changes no action-mask rule: an edge
or assignment remains legal only when the corresponding sampled capability is
above the frozen threshold.

## Metrics

Primary controllability remains the L1 distance between the
priority-weighted assignment mix and the suite-specific target. Reports also
retain L2, cosine alignment, nominal and dependency-feasible sensitivities,
per-type coverage, deadline completion, makespan, communications, invalid
actions, reallocation success, regret, hypervolume and IGD.

No controllability claim is valid if it is produced by losing required-task
coverage or violating the frozen GPPO efficiency guards.
