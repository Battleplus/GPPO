# PCRL-v0 anchor-strength calibration plan

## Purpose and protocol boundary

This calibration is a pre-training feasibility screen. It does not change the
frozen `gppo-v2-hard-3` implementation, does not accept PCRL-v0, and does not
authorize JEPA or world-model integration. The current hard-3 formal protocol
keeps `assignment_priority_decay=2.0`; any selected decay other than `2.0`
requires a new protocol version before training.

## Candidate preference families

For priority share `a`, each non-priority task receives
`b=(1-a)/3`. The coarse grid is:

`a in {0.35, 0.375, 0.40, 0.425, 0.45, 0.475, 0.50}`.

Each candidate is evaluated on the same eight-profile family:

1. balanced `(0.25, 0.25, 0.25, 0.25)`;
2. four single-task anchors with one component `a` and three components `b`;
3. the midpoint of the search and strike anchors;
4. the midpoint of the reconnaissance and recovery anchors;
5. the preregistered asymmetric convex combination
   `0.5*search + 0.3*reconnaissance + 0.2*recovery`.

Historical hard-2 interpolation profiles are not mixed into this macro because
they do not stay inside every candidate family. If several candidates pass,
select the smallest passing `a` with `a >= 0.40`.

## Oracle and comparator

The primary diagnostic is
`oracle_perfect_information_mask_faithful`: it reads true physical state but
may select only actions legal under the frozen action mask. Strategic waiting
is disabled. `oracle_perfect_information_relaxed_wait` changes the action set
and is reported only as a secondary diagnostic.

The surrogate comparator is the mean of all five frozen `gppo_event`
checkpoints. Within each `eval_seed x profile x scale` cell, GPPO metrics are
first averaged over training seeds. Oracle and GPPO are then paired on the
identical evaluation seed and cell. This comparison is not the formal PCRL
denominator; formal acceptance still requires a newly trained hard-3 true
no-conditioning ablation.

## Statistical units and seed separation

- Oracle implementation tuning: `72500..72519`;
- coarse anchor screen: `73000..73002`;
- expanded anchor selection: `73000..73029`;
- locked-candidate confirmation: `74000..74049`;
- later 20-update pilot evaluation: `75000..75099`;
- formal evaluation remains `50000..50099`.

The independent unit is `eval_seed`. Profile and scale rows are aggregated
inside each evaluation seed before paired bootstrap confidence intervals are
computed. GPPO training seeds and profile/scale rows are never treated as
independent bootstrap samples.

## Screening and stopping rules

For the coarse three-seed screen, report point estimates and per-direction
responses only; its confidence interval is descriptive. A candidate advances
to the 30-seed selection run only if its frozen-mask Oracle point headroom is
at least 40% relative to the five-checkpoint frozen GPPO mean, all four priority
directions respond positively, and point-estimate efficiency guards pass.

The locked confirmation requires:

- the one-sided 95% lower confidence bound for Oracle headroom is at least 40%;
- deadline-completion delta lower bound is at least `-0.03`;
- makespan-increase upper bound is at most `0.05`;
- minimum-type-coverage delta lower bound is at least `-0.05`;
- no strategic waits, action-mask violations, missing cells, or mixed profile
  vectors occur.

If no coarse candidate reaches 40% point headroom, do not spend the 30-seed or
five-training-seed budget. Retain the negative result and state that the 30%
learned-policy gate is not yet supported by the frozen environment/action
design. Failure of the heuristic Oracle is evidence of missing demonstrated
headroom, not a mathematical proof of impossibility.

Only after a candidate passes locked confirmation should a new protocol freeze
its anchors, same-family held-out profiles, continuous training sampler, and
metric coefficient. Then retrain the true no-conditioning ablation before a
five-seed, 20-update PCRL pilot.
