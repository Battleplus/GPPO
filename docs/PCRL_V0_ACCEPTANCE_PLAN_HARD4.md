# PCRL-v0 hard-4 acceptance plan

## Pre-registration boundary

This plan applies only to `pcrl-v0-hard-4`, decay `2.0`, priority share
`0.40`, and the exact dynamic eight-profile family in the hard-4 config.
Hard-2, hard-3, coarse calibration, pilot and formal rows remain separate.
Hard-4 training has not started at protocol creation time.

## Pilot20

The `pilot20` group trains PCRL and the true no-conditioning ablation for 20
updates, 18 episodes per update, and seeds `1..5`. Frozen GPPO controls use the
same five checkpoint seeds. Each method is evaluated for 20 episodes per
profile/scale/suite using evaluation seeds `76000..76019`.

Here, true no-conditioning has a narrow, preregistered meaning. It removes the
requested preference and dynamic `preference_deficit` only from the policy and
value-network inputs, replacing them at that boundary with the fixed balanced
default and zero deficit. It deliberately preserves the same sampled episode
preferences, vector rewards, preference IDs and grouped multi-objective/PreCo
training directions used by conditional PCRL. This comparison isolates
conditioning, not the broader effect of preference-aware multi-objective
training. It is therefore not evidence against a completely neutral-objective
policy, which is outside the hard-4 denominator definition.

The pilot is a go/no-go diagnostic. It must report paired training-seed
differences, 95% intervals, all five per-seed outcomes, the four directional
responses, held-out interpolation and efficiency/coverage guards. It cannot
be called formal evidence and its checkpoints cannot initialize formal runs.

The formal stage remains locked unless the pilot shows:

- lower macro primary-suite L1 than true no-conditioning;
- improvement in at least four of five paired seeds;
- all four priority directions respond relative to balanced;
- deadline completion is no more than `0.03` below frozen GPPO-event;
- makespan is no more than 5% above frozen GPPO-event;
- minimum required-task coverage loss is at most `0.05`;
- no task abandonment, identity mismatch or action-mask mismatch.

An interval crossing zero is inconclusive. Passing this pilot authorizes a
fresh formal run; it does not itself accept hard-4.

## Formal100

The `formal100` group trains fresh policies for 100 updates and 18 episodes
per update using seeds `1..5`. It evaluates 100 episodes per cell on seeds
`50000..50099`. The five anchors and three held-out profiles are all included;
the phase and chain suites are summarized separately.

Formal acceptance requires every primary-suite gate:

- macro preference L1 at least 30% lower than true no-conditioning;
- monotonic response in all four priority directions;
- held-out interpolation better than its matched no-conditioning cells;
- deadline completion no more than `0.03` below frozen GPPO-event;
- makespan no more than 5% above frozen GPPO-event;
- minimum required-task coverage loss at most `0.05`;
- at least four of five training seeds satisfy preference and efficiency
  guards;
- no abandonment, availability manipulation, action-mask mismatch or run-
  identity violation.

Point estimates, paired confidence intervals and absolute L1 differences are
reported together. Five seeds provide limited exact-test resolution; exact
two-sided sign-flip p-values and any multiplicity adjustment are descriptive,
not substitutes for effect-size and stability gates.

## Artifact integrity and stopping rules

Before aggregation, validate every required identity field in the hard-4
config. Reject missing or mixed artifact groups, wrong profile vectors,
priority share other than `0.40`, decay other than `2.0`, frozen checkpoint or
implementation hash mismatch, and relabeled hard-2/hard-3 rows.

If pilot gates fail or are inconclusive, retain the negative artifact and do
not start formal100. If formal gates fail, retain the formal negative result.
World-model or JEPA work does not begin on the basis of an unaccepted hard-4
result.
