# PCRL-v0 hard-5 baseline protocol

`pcrl-v0-hard-5` is a new engineering protocol. It does not overwrite or
relabel the failed hard-4 pilot. The accepted `gppo-v2-hard-3` source files and
five checkpoint SHA-256 values remain frozen.

## Why hard-5 exists

Hard-4 reduced preference L1 by only 11.13% versus true no-conditioning,
Search monotonicity failed, and the paired makespan upper confidence bound
exceeded the allowed 5% increase. Inspection found two protocol-level causes:

1. the legacy UAV sampler systematically forced Search below the capability
   threshold for every nonzero UAV index;
2. the actor received the first four entries of a mapped seven-objective
   preference, not the raw four-dimensional user task preference.

Hard-5 changes only the PCRL wrapper and PCRL model. `paper_env.py`, the GPPO
model, accepted GPPO checkpoints, event synchronization, weak-communication
cache, action mask, task dynamics and failure behavior are unchanged.

## Hard-5 environment contract

The PCRL wrapper uses `balanced_cyclic_min2_v1`. At episode reset, task type
`t` is executable by cyclic UAV specialists `t mod n` and `(t+1) mod n`, where
`n` is the number of active UAVs. Every task type therefore has exactly
`min(2,n)` initially capable UAVs. Non-specialists are below threshold. UAV
failures may reduce coverage later and are not masked.

Frozen GPPO, no-conditioning and PCRL policies are re-evaluated in this exact
same wrapper environment. Their source checkpoints are not retrained or
modified.

## Preference boundary

Hard-5 uses `split_task_objective_v2`:

- actor input: raw task preference (4) plus dynamic deficit (4);
- action-alignment target: raw task preference (4), restricted to legal types;
- critic input: mapped objective preference (7);
- PreCo/grouped MOO input: the original mapped objective preference (7).

The no-conditioning control retains the same sampled preferences, vector reward
tapes, preference IDs and PreCo directions, but neutralizes task preference,
objective preference and deficit at the network boundary.

The compatibility suite retains the literal raw user request for actor control
while its deficit and primary metric use the dependency-feasible target. This
intentional mismatch is reported only as a transfer diagnostic; it is never
pooled with the primary nominal suite.

## Evidence boundary

This remains an independent mechanism-level reproduction because neither paper
publishes the complete UAV environment and training implementation. No JEPA,
event-aware world model or learned trigger is developed before hard-5 PCRL
passes its controllability and efficiency gates.
