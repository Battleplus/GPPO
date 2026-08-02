# PCRL Oracle delay and prerequisite audit

## Protocol boundary

Two Oracle variants must be reported separately.

- `oracle_perfect_information_mask_faithful` reads the true physical state but
  selects only actions legal under the frozen GPPO action mask. Strategic noop
  is disabled. This is the primary headroom diagnostic.
- `oracle_perfect_information_relaxed_wait` additionally permits an intentional
  one-boundary noop while assignments are feasible. This changes the frozen
  action set and is only a secondary relaxed upper bound. It is not a GPPO or
  PCRL baseline.

The relaxed executor does not modify `paper_env.py` or its action mask. It uses
the existing rejected-action clock transition and removes the artificial
invalid-action and safety accounting. Physical completions, exogenous events,
communication synchronization and termination remain those of the frozen
environment.

## Scoring change

The mask-faithful Oracle now values all active descendants of a prerequisite,
discounted by graph depth. The previous direct-successor-only score undervalued
search or reconnaissance actions that unlock a preferred strike or recovery
task two or three edges later. Terminal rollout selection also treats configured
deadline-completion and minimum-type-coverage shortfalls as constraints before
preference L1.

The relaxed rollout may wait one physical boundary when the best assignment
available after the boundary has a higher prerequisite-aware score than the
best current assignment by `rollout_wait_margin`. The evaluator default margin
is `0.05`; waiting requires the explicit `--allow-strategic-wait` flag.

## Preliminary paired evidence

This calibration uses `2x12`, eval seeds `70000..70002`, decay `2.0`, and four
historical diagnostic profiles (`balanced`, `search_strike_interp`,
`recon_recovery_interp`, and `smooth_interp`), for 12 paired episodes per
variant. It uses a deadline-completion floor of `0.60` and a minimum-type
coverage floor of `0.40`. These are not the four new single-task moderate
anchors, so this run audits the Oracle executor and prerequisite scoring only;
it is not an anchor-headroom gate or a formal five-seed experiment. CIs in
generated summaries first aggregate profile/scale cells by `eval_seed`;
correlated rows are not counted as independent samples.

| Variant | Preference L1 | Deadline completion | Min type coverage | Makespan | Waits |
| --- | ---: | ---: | ---: | ---: | ---: |
| No prerequisite score, frozen mask | 0.4051 | 0.6458 | 0.4444 | 16.9668 | 0.00 |
| Prerequisite-aware, frozen mask | 0.3657 | 0.6528 | 0.3889 | 16.2712 | 0.00 |
| Prerequisite-aware, relaxed wait | 0.3833 | 0.6597 | 0.4167 | 16.0149 | 2.33 |

Artifacts:

- `outputs/pcrl_v0/oracle_moderate_audit_no_prerequisite/`
- `outputs/pcrl_v0/oracle_moderate_audit_mask_faithful/`
- `outputs/pcrl_v0/oracle_moderate_audit_relaxed_wait/`

Prerequisite-aware scoring improves L1 by `9.72%` and deadline completion by
`0.0069` relative to the direct-successor control, but its minimum-type coverage
loss is `0.0556`, slightly larger than the PCRL anti-gaming allowance of `0.05`.
Relaxed waiting restores part of that coverage and improves makespan, but gives
back some preference alignment. Relative to the direct-successor control, the
combined relaxed variant improves L1 by only `5.38%`.

## Acceptance implication

The preliminary no-conditioning L1 is about `0.4097`; a 30% reduction requires
L1 at or below about `0.2868`. The primary mask-faithful Oracle mean is `0.3657`
on this historical-profile executor audit, and the relaxed-wait mean is
`0.3833`. Neither supports the proposed 30% learned-policy gate, but neither is
a complete moderate-anchor comparison. The three-seed mask-faithful CI is wide
(`0.2823..0.4491`) and its lower endpoint barely crosses the threshold, so this
is evidence against claiming attainability, not a proof of impossibility.

Do not lower the frozen 30% acceptance gate from this calibration. Report it as
unjustified under the current nominal target and priority-weighted assignment
metric, then require a larger paired Oracle audit before spending the formal
100-update PCRL budget.
