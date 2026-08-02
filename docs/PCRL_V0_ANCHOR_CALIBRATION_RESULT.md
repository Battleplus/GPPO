# PCRL-v0 coarse anchor calibration result

## Decision

Do not start the 30-evaluation-seed expansion, five-training-seed PCRL pilot,
or formal PCRL run. No practically meaningful candidate (`priority_share >=
0.40`) reached the preregistered 40% Oracle-vs-frozen-GPPO point-headroom
screen. JEPA and world-model integration remain blocked.

This is a pre-training surrogate screen, not the formal 30% PCRL acceptance
comparison. Formal acceptance still requires a newly trained hard-3 true
no-conditioning ablation as the denominator.

## Design

- protocol identity: `pcrl-v0-hard-3-anchor-grid-coarse`;
- frozen base: `gppo-v2-hard-3`;
- assignment-priority decay: `2.0`;
- primary suite: `phase_staggered`, nominal target, deadline scale `0.70`;
- priority shares: `0.35, 0.375, 0.40, 0.425, 0.45, 0.475, 0.50`;
- each share: eight same-family profiles, four scales, eval seeds
  `73000..73002`;
- Oracle: true state, frozen action mask, 12-candidate terminal rollout, no
  strategic waiting;
- comparator: five frozen `gppo_event` checkpoints, averaged within each
  eval-seed/profile/scale cell before inference;
- independent unit: three paired eval-seed macros;
- uncertainty: 10,000-sample paired eval-seed bootstrap.

The three-seed confidence intervals are descriptive coarse-screen intervals.
They are not confirmatory intervals.

## Paired results

| priority share | frozen GPPO L1 | mask-faithful Oracle L1 | headroom | paired 95% CI | efficiency guards |
| ---: | ---: | ---: | ---: | ---: | :---: |
| 0.350 | 0.223657 | 0.133056 | 40.51% | [25.70%, 50.92%] | pass |
| 0.375 | 0.247858 | 0.161868 | 34.69% | [20.61%, 46.49%] | pass |
| 0.400 | 0.274536 | 0.192055 | 30.04% | [16.70%, 41.01%] | pass |
| 0.425 | 0.303304 | 0.221673 | 26.91% | [13.52%, 36.17%] | pass |
| 0.450 | 0.333323 | 0.245755 | 26.27% | [13.48%, 33.14%] | pass |
| 0.475 | 0.364899 | 0.278179 | 23.77% | [11.65%, 30.33%] | pass |
| 0.500 | 0.397261 | 0.311650 | 21.55% | [10.91%, 27.78%] | pass |

All macro guard decisions use the conservative paired-bootstrap boundary. The
Oracle improved deadline completion and minimum coverage and did not increase
makespan at any share. No Oracle row used strategic waiting or violated the
frozen action-space label.

## Directional bottleneck

At the smallest practically meaningful candidate, `priority_share=0.40`, the
per-profile Oracle headroom was:

| profile | headroom |
| --- | ---: |
| balanced | 54.69% |
| search priority | 19.45% |
| reconnaissance priority | 4.38% |
| strike priority | 20.51% |
| recovery priority | 43.22% |
| search-strike interpolation | 26.96% |
| recon-recovery interpolation | 45.63% |
| asymmetric interpolation | 39.21% |

The macro is therefore not evidence that every priority direction is equally
controllable. Reconnaissance is the clearest structural bottleneck, followed
by search and strike. Increasing anchor strength worsens Oracle L1 faster than
it creates comparator error, so no stronger candidate recovers the 40% gate.

## Interpretation and stop rule

The diagnostic `0.35` share reaches 40% only as a point estimate, has a lower
confidence bound of 25.70%, and gives the preferred task only 1.62 times the
weight of each non-preferred task. It is below the preregistered practical
candidate floor and is not advanced merely to make the gate pass.

The first eligible candidate, `0.40`, reaches only 30.04% point headroom. The
coarse stopping rule therefore applies: do not spend the 30-seed selection or
five-training-seed budget. Preserve the result as negative evidence that the
30% learned-policy target is not currently supported by demonstrated headroom
under the frozen environment and action mask.

The Oracle is a strong heuristic, not a mathematically optimal solver. This
result does not prove that 40% headroom is impossible. It proves that the
current implementation has not demonstrated enough headroom to justify PCRL
training. A future attempt must improve the frozen-mask scheduler or revise the
environment/action design under a new protocol; it must not silently weaken
the anchor, change decay, or use relaxed waiting under the hard-3 label.

## Artifact integrity

Raw Oracle and GPPO evaluations are under
`outputs/pcrl_v0/hard3/anchor_grid_coarse/`. Oracle runs made with the hard-3
wrapper config and GPPO runs made with the base scenario config used exactly
equal physical `scenario` and `scale_deadlines` objects. Analysis uses derived
metadata-normalized copies under `derived/`; raw artifacts are unchanged.
Every derived copy records source and derived SHA-256 hashes, config paths and
hashes, exact equality flags, and source/target protocol identities.

Paired reports are under `headroom/share_*`. The frozen implementation hash is
`a62c17f721688e2ae0c36c6fe11ef1a6cced365c8468155bfacbf4f286ea01a6`,
and all five frozen GPPO checkpoint hashes still match
`configs/pcrl_v0_frozen_checkpoints.json`.
