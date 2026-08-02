# PCRL-v0 hard-6 acceptance plan

## Checkpoint selection

- Only the five training anchor profiles participate in checkpoint selection.
- The three interpolation profiles are held out until evaluation.
- Deadline, makespan and minimum-coverage guards remain paired against the
  frozen GPPO validation tape.
- Invalid actions use a relative guard:
  `PCRL - GPPO <= 0.05` actions per episode.
- Update 0 and every validation candidate are saved with metrics and SHA-256 so
  selection can be reproduced offline without retraining.

## Calibration gate

The three-seed `calibration20` screen must satisfy all of the following before
`pilot20` is unlocked:

- primary-five L1 reduction versus true no-conditioning is at least 30%;
- all three seeds improve;
- Search, Reconnaissance, Strike and Recovery priority directions are positive
  in all three seeds;
- held-out-three mean L1 reduction is at least 20% and all three seeds improve;
- deadline, makespan, coverage and relative invalid-action guards pass;
- the selected checkpoint is feasible, not a diagnostic fallback.

Formal acceptance retains the 30% mean gate, at least 4/5 improving seeds,
paired confidence intervals, correct monotonic directions, held-out control and
all efficiency/safety guards. Failure keeps pilot, formal, JEPA and the world
model locked.

