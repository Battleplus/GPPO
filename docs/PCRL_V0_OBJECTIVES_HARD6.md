# PCRL-v0 hard-6 objectives and controller

The seven-dimensional reward vector and four-dimensional user task preference
retain the hard-5 definitions. Hard-6 changes how the actor realizes the task
preference.

## Explicit five-group mass controller

Legal actions are aggregated into Search, Reconnaissance, Strike, Recovery and
Noop groups. The controller:

1. computes a deficit-aware target over currently legal task types;
2. obtains the detached assignment/noop mass from the frozen GPPO backbone;
3. preserves that assignment/noop rhythm;
4. mixes the current group mass with the target through a bounded,
   state-dependent `alpha` head;
5. adds group log-mass corrections while preserving the GPPO ranking inside
   each task group and retaining the original action mask.

The mass auxiliary uses the same dynamic target and includes Noop. States with
fewer than two legal task types do not contribute a false controllability
gradient.

## PreCo sampling corrections

- Four complete episodes reuse each sampled preference.
- PreCo directions are computed once from the full rollout batch before PPO
  minibatch shuffling and then cached.
- The estimated preference-group value uses trajectory-initial vector returns.
- Dynamic per-objective whitening is disabled; only the final scalar policy
  direction is standardized.
- The positive-quality transform is fixed to `softplus`.

