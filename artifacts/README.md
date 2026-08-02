# Included formal artifacts

This directory contains the lightweight evidence needed to inspect the accepted
GPPO-v2 first-stage result:

- `checkpoints/gppo_event/seed_1..5`: adaptive GPPO-event checkpoints;
- `checkpoints/gppo_event_single_head/seed_1..5`: stronger single-head control;
- `summary/`: aggregate tables, per-seed results, comparisons, audits and curves.

The complete local formal run contained 40 learned checkpoints, 42 evaluation
files, 16,800 evaluation rows and 42 event logs with 935,138 event records.
Large raw evaluations and event logs are excluded from Git. Their protocol and
artifact audits are recorded in `summary/formal_artifact_audit.json` and the
documents under `docs/`.

Do not describe these artifacts as a numerical reproduction of the source
paper. They are evidence for the frozen independent engineering protocol.
