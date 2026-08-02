# PCRL-v0 acceptance and statistical plan

## Smoke gates before formal training

The following tests must pass before any formal run: vector reward conservation; availability and reachable denominator accounting; exact GPPO checkpoint initialization; preference-dependent logits with unchanged action masks; seven-output critic; distinct preference per episode; grouped multi-preference PPO/PreCo update; phase-staggered exposure of multiple legal task types; dynamic `preference_deficit` updates; true no-conditioning invariance to both preference and deficit; held-out profile evaluation; and runtime preference switching. The frozen GPPO implementation hash is checked in the same test invocation.

## Formal design

For each learned method, train seeds 1–5 using `configs/pcrl_v0.json` and `phase_staggered` task release with the configured `0.70` deadline scale. Evaluate 100 episodes for each of four scales, each of the five anchors, and three held-out interpolations in both suites:

- primary `controllability_phase`: `phase_staggered`, deadline scale `0.70`;
- secondary `chain_compatibility`: original `chain`, deadline scale `1.0`.

Keep all per-episode rows, event logs, checkpoint metadata, suite labels, and failures. Baselines are evaluated under the same suite, scale, profile, and episode seeds; frozen GPPO checkpoints are never retrained or overwritten. Do not combine rows from the two suites for a confidence interval or acceptance gate.

## Statistics

Aggregate episodes within each training seed first, then compute macro means across scales separately for each suite. Confidence intervals are 95% t intervals across the five training seeds, not across correlated episodes. Report paired PCRL-minus-no-conditioning and PCRL-minus-frozen-event-GPPO differences, confidence intervals, the number of seeds satisfying each gate, and Holm-adjusted p-values when hypotheses are tested. Inconclusive intervals crossing zero remain inconclusive; no significance wording is permitted when the interval is inconclusive. The primary preference gate uses `preference_l1` against the dependency-feasible target; `preference_l1_nominal` is a descriptive sensitivity measure.

Preference regret is relative to the best observed method under the same profile, scale, and suite. Pareto hypervolume and IGD use the seven-objective normalization in `PCRL_V0_OBJECTIVES.md`; the reference front is empirical, not an oracle. Compatibility-suite results are transfer diagnostics and do not replace primary controllability evidence.

## Failure criteria

Reject or label PCRL-v0 as provisional if any primary-suite condition occurs: fewer than four stable seeds; preference error reduction below the frozen 30% gate; monotonicity failure in a priority direction; held-out interpolation collapse; deadline completion drop above 0.03; makespan increase above 5%; minimum required-task coverage loss above 0.05; action-mask or hash mismatch; or evidence that preference matching is achieved by leaving necessary task types unfinished. A preliminary 20-update run, a single seed, or a compatibility-suite result cannot waive a failed gate. All negative and inconclusive ablations remain in the final report.
