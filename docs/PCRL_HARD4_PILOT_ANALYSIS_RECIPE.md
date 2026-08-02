# PCRL hard4 pilot analysis recipe

## Scope

This recipe prepares the comparison path only. It does not authorize or start
training. The pilot numerator and denominator must be newly trained under the
same hard4 protocol, budget and training seeds:

- numerator: `pcrl_gppo_adaptive`;
- formal preference denominator:
  `pcrl_gppo_adaptive_no_conditioning`;
- efficiency-only comparator: frozen `gppo_event`.

Historical hard2 no-conditioning checkpoints and frozen GPPO preference L1 are
not valid denominators for the 30% preference gate.

## Required evaluation grid

Evaluate all three methods with identical training-seed labels `1..5`, scales,
evaluation seeds and dynamic profile vectors. The strict summarizer rejects a
missing or duplicate `(training_seed, profile, scale, eval_seed)` cell.

The dynamic family at the locked share contains:

- primary macro: balanced plus the four
  `calibration_*_priority` anchors;
- held-out test: `calibration_search_strike_interp`,
  `calibration_recon_recovery_interp`, and
  `calibration_asymmetric_search_recon_recovery`.

Do not mix historical hard2 interpolation profiles into either set.

Every row must record `protocol_version=pcrl-v0-hard-4`,
`artifact_group=pilot20`, `evaluation_suite=controllability_phase`,
`preference_profile_family=dynamic-priority-share-v1`, 20 updates, 18 episodes
per update, 20 evaluation episodes, `phase_staggered`, nominal target, deadline
scale `0.70`, assignment-priority decay `2.0`, priority/background shares
`0.40/0.20`, and the exact four-dimensional profile vector. The strict path
rejects `formal100`/pilot pooling. The no-conditioning method ID must end in
`_no_conditioning`. If `preference_conditioning` is present, it is also
cross-checked.

## Summary command

After the pilot evaluations exist, run:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
D:\anaconda\python.exe summarize_pcrl_v0.py `
  --inputs `
    outputs\pcrl_v0\hard4\pilot20\eval\pcrl_gppo_adaptive `
    outputs\pcrl_v0\hard4\pilot20\eval\pcrl_gppo_adaptive_no_conditioning `
    outputs\pcrl_v0\hard4\pilot20\eval\gppo_event `
  --pcrl-method pcrl_gppo_adaptive `
  --no-conditioning-method pcrl_gppo_adaptive_no_conditioning `
  --gppo-method gppo_event `
  --hard4-pilot `
  --hard4-protocol-version pcrl-v0-hard-4 `
  --hard4-priority-share 0.40 `
  --hard4-training-seeds 1 2 3 4 5 `
  --hypervolume-samples 10000 `
  --output outputs\pcrl_v0\hard4\pilot20\summary
```

The authoritative pilot outputs are `summary.json` under
`hard4_pilot_acceptance` and `PCRL_HARD4_PILOT.md`.

## Statistical interpretation

Rows are first macro-averaged within each training seed. The 30% quantity is
`1 - PCRL_L1 / no_conditioning_L1` on the five paired seed macros, with a 95%
Student-t interval. Same-family held-out improvement is calculated separately.
Monotonicity compares each priority anchor with balanced on matched scale and
evaluation-seed cells. DCR, makespan and minimum-coverage guards compare PCRL
with GPPO in a separate block.

With only five paired training seeds, the smallest possible two-sided exact
sign-flip p-value is `0.0625`. The report therefore retains the exact result but
does not claim exact `p < 0.05` significance.
