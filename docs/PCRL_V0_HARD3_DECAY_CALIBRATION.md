# PCRL-v0 hard-3 assignment-priority decay calibration

## Scope and status

This is a small, pre-training metric calibration, not a learned PCRL result.
It uses one frozen training seed and five evaluation episodes per
scale-preference cell.  The primary suite is fixed to `phase_staggered`,
deadline scale `0.70`, the nominal preference target, four evaluation scales,
and eight preference profiles.  Evaluation seeds are `61000..61004`.

The formal comparator is frozen GPPO-event seed 1:

- checkpoint SHA-256:
  `5e419533809124c111733e579cf350e41ea6a8814a6607fb2dc40246090c9065`;
- frozen implementation SHA-256:
  `a62c17f721688e2ae0c36c6fe11ef1a6cced365c8468155bfacbf4f286ea01a6`.

The no-conditioning rows are explicitly diagnostic only.  They use the
historical 20-update hard-2 checkpoint with SHA-256
`8fba31c5c3bb8b8862911b13cc4a80f676d99ccf3d82957c2ecb796b82797c98`.
They must not be cited as a hard-3-trained ablation or pooled with a future
hard-3 confirmatory run.

Raw artifacts are under
`outputs/pcrl_v0/hard3/decay_calibration/`.  Every method-decay directory
contains 160 paired evaluation rows.

## Macro results

| decay | frozen GPPO L1 | diagnostic no-conditioning L1 | 30% PCRL gate from diagnostic comparator | 40% Oracle-headroom threshold |
| ---: | ---: | ---: | ---: | ---: |
| 1.0 | 0.653452 | 0.652154 | 0.456508 | 0.391293 |
| 1.5 | 0.655411 | 0.654150 | 0.457905 | 0.392490 |
| 2.1 | 0.657546 | 0.656428 | 0.459499 | 0.393857 |
| 2.5 | 0.659456 | 0.658525 | 0.460967 | 0.395115 |
| 3.0 | 0.662862 | 0.662218 | 0.463553 | 0.397331 |

The last two columns are comparator thresholds, not observed PCRL or Oracle
performance.  For example, if `decay=2.1` is selected independently, a learned
PCRL policy must reach macro L1 at most `0.459499` for the frozen 30% gate, and
the constrained Oracle must reach at most `0.393857` to demonstrate 40%
headroom relative to this diagnostic no-conditioning comparator.

At `decay=2.1`, profile-level frozen-GPPO L1 is:

| profile | L1 |
| --- | ---: |
| balanced | 0.146018 |
| search | 0.954216 |
| reconnaissance | 0.827375 |
| strike | 0.891151 |
| recovery | 0.944682 |
| search-strike interpolation | 0.639559 |
| reconnaissance-recovery interpolation | 0.566249 |
| smooth interpolation | 0.291118 |

## Invariance and interpretation

For both policies, changing decay from `1.0` to any other grid point produced
exactly zero paired change in deadline completion, makespan, minimum type
coverage, communication count, invalid actions, and completed tasks.  This is
the expected control: decay only recomputes priority mass for these
non-conditioned policies; it does not alter their action trajectories.

Macro L1 of both non-conditioned comparators increases slightly and
monotonically over this grid.  Individual priorities move in opposite
directions: larger decay lowers search and reconnaissance error but raises
recovery and strike error.  Even choosing the best decay separately for every
profile gives diagnostic no-conditioning macro L1 `0.639047`, only a small
change from the fixed-grid values.  Consequently:

1. the grid does not manufacture a 30% preference-control improvement for an
   unconditioned policy;
2. non-conditioned L1 cannot identify a scientifically preferred decay;
3. decay selection must be based on a constrained full-state Oracle evaluated
   on disjoint calibration seeds, with completion, makespan, communication,
   safety, and per-type coverage constraints retained;
4. after selection, the decay and target semantics must be frozen before any
   new PCRL checkpoints are trained.

Because this audit has only one training seed and five episodes per cell, it
must not be used for confidence intervals, formal acceptance, or claims about
training stability.
