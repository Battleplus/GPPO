# PCRL-v0 hard-6 baseline protocol

`pcrl-v0-hard-6` is a separately versioned controllability protocol. It does
not replace or reinterpret hard-4/hard-5 results.

## Frozen baseline

- Base protocol: `gppo-v2-hard-3`.
- Frozen GPPO implementation hash:
  `a62c17f721688e2ae0c36c6fe11ef1a6cced365c8468155bfacbf4f286ea01a6`.
- Adaptive GPPO-event remains the mechanism-aligned efficiency reference.
- Single-head GPPO-event remains the stronger competitive reference.
- Frozen GPPO files and checkpoints must not be modified.

## Why hard-6 exists

The hard-5 gain screen improved primary-five L1 by only 10.86% from gain 1 to
gain 4 and increased invalid actions. Code audit also found that PreCo was
estimated inside shuffled minibatches from fragments of one episode per
preference, while checkpoint selection included held-out profiles.

Hard-6 therefore changes the PCRL control and training protocol, not the
frozen environment or GPPO baseline.

## Locked scope

JEPA, event-aware world models and learned event prediction remain locked.
Only current real, preset or random events may be used until PCRL passes.

