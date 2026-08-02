# PCRL-v0 hard-5 acceptance and calibration plan

Hard-4 is a preserved negative pilot. Hard-5 starts with `calibration10`, not a
formal run.

## Calibration10

- training seeds: 11, 12, 13;
- frozen source checkpoint seeds: 1, 2, 3 respectively;
- budget: 10 updates x 12 episodes/update;
- validation: every 2 updates, 10 episodes, base seed 78500;
- evaluation: primary suite only, seeds 79000..79009;
- calibration results cannot initialize or count as pilot/formal evidence.

The preregistered diagnostic matrix is:

- A: raw 4D actor preference only;
- B: balanced capability coverage only;
- C: A plus B;
- D: C plus constrained checkpoint selection.

C and D have the same training mechanism; they differ only in selection. A
single winner must be frozen before a fresh five-seed pilot. The 30% gate is not
weakened during calibration.

## Constrained checkpoint selection

Before training, the zero-residual policy is evaluated as the exact frozen GPPO
reference on the same profile-paired validation tape. A candidate is feasible
only if all conditions hold:

- deadline completion drop <= 0.03;
- makespan increase <= 5%;
- minimum task coverage loss <= 0.05;
- macro invalid actions <= 0.

Among feasible candidates, minimize preference L1, then prefer higher deadline
completion, lower makespan, higher coverage and the earlier update. If no
candidate is feasible, select a maximin-margin diagnostic checkpoint and mark
`selection_passed=false`; it cannot be accepted as pilot or formal evidence.

## Pilot and formal gates

A fresh five-seed pilot is unlocked only after calibration shows a credible
direction. Initial PCRL acceptance still requires:

- at least 30% L1 reduction versus true no-conditioning;
- correct monotonic response for all four priority profiles;
- controllability on held-out interpolation profiles;
- deadline completion drop no more than 0.03 versus frozen GPPO-event;
- makespan increase no more than 5%;
- minimum coverage loss no more than 0.05;
- stability in at least 4 of 5 training seeds.

Formal100 remains locked until every pilot gate passes. World-model and JEPA
work remains locked until PCRL passes.
