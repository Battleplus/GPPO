# Phase 1B Disturbance Calibration Report

## Scope

This report validates the configurable, replayable and auditable multi-source disturbance layer. It does not claim that GPPO, preference learning or a learned world model is effective. Calibration uses three seeds beginning at 60,000,000 and is disjoint from train, validation and formal test100 banks. The controller is oracle-reconciled solely to isolate physical disturbance severity from policy/cache quality.

## Calibration result

| Severity | Mean realized makespan | Drop rate | Mean delivered delay | Minimum energy | Completion rate |
|---|---:|---:|---:|---:|---:|
| off | 16.935 | 0.000% | 0.000 | 1.000 | 1.000 |
| weak | 18.673 | 0.980% | 0.124 | 0.914 | 1.000 |
| medium | 19.971 | 10.241% | 0.385 | 0.757 | 1.000 |
| strong | 20.903 | 28.840% | 0.837 | 0.564 | 1.000 |

All 12 episodes terminated, all actions accepted, and no NaN/Inf was serialized. Makespan, packet loss, delay and energy depletion show an ordered degradation from off through strong, while strong retains successful episodes rather than collapsing the environment.

## Event coverage in the replayable sample tape

| Event type | Count |
|---|---:|
| delay_profile | 1 |
| energy_profile | 1 |
| link_state | 4020 |
| network_partition | 1 |
| task_arrival | 1 |
| task_cancellation | 1 |
| task_deadline_change | 1 |
| task_priority_change | 1 |
| uav_failure | 1 |
| uav_recovery | 1 |
| wind_field | 1 |

## Reproducibility and data interface

- Configuration, tape, event log and trajectory carry SHA-256 identifiers.
- All-off equivalence was checked on the frozen T5 test100 bank for 2,000 paired decisions; observations, action masks, rewards, event tape, true state and metrics matched exactly.
- Physical occurrence and observed time are stored separately.
- The lossless `sample_trajectory.json.gz` contains partial observation, true graph state, belief cache, legal mask, communication history, UAV/task state, future 1–5 decision-event targets and seven unscalarized objective components.
- All-off byte/step equivalence is protected by automated regression tests.

## Acceptance

Machine audit: `DISTURBANCE_IMPLEMENTATION_AUDIT.json` (`valid=true`).
