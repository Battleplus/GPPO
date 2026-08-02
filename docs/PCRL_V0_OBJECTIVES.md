# PCRL-v0 preference, reward, and metric definitions

## Preference semantics

The user controls a four-dimensional nominal task profile

\[
 p^{task}=(p_s,p_r,p_{strike},p_{recovery}),\qquad p_i\ge0,\quad\sum_i p_i=1.
\]

The strict S-R-A-H-S lifecycle makes some nominal proportions impossible. PCRL-v0 therefore projects the nominal profile onto the dependency-feasible simplex `recon <= 2 search`, `strike <= recon`, `recovery <= strike` using Dykstra projection. The projected vector is the allocation target and policy-conditioning task vector; the nominal vector is retained and its error is also reported. `map_task_preference_to_full` maps the feasible target to seven optimization weights. Seventy percent of the mass remains on the four task objectives and 30% is reserved for efficiency, communication, and safety with fixed weights `(0.15, 0.05, 0.10)`. A 0.05 floor is applied inside the task simplex, so a priority profile cannot obtain a high similarity score by assigning zero weight to a necessary task type.

The implementation supports one conditional policy `π(a|s,p)`, not one network per preference. Training samples one continuous preference per episode. Jittered anchors are used only to improve coverage; exact held-out interpolation vectors are excluded from training. `PreferencePaperEnv.set_preference` supports a runtime switch without rebuilding the model.

The policy receives both the seven-dimensional optimization preference and a four-dimensional closed-loop allocation deficit

\[
d_t=p^{feasible}-\operatorname{mix}(q_t),
\]

where `q_t` is the priority-weighted assignment coverage accumulated by time `t`. The deficit is refreshed after every assignment so a static preference can produce closed-loop correction toward its target. In the no-conditioning ablation, the user preference is replaced by one fixed balanced default and `preference_deficit` is replaced by zeros; neither input can vary with the requested profile.

## Task-release semantics

The primary controllability suite trains and evaluates on `phase_staggered` rolling task chains with a deadline equal to `0.70` times the frozen scale-specific GPPO deadline. This exposes multiple legal task types often enough to identify allocation preferences while preserving S-R-A-H-S dependencies. The original `chain` generator and unscaled deadline form a separate evaluation-only compatibility suite. Metrics, confidence intervals, Pareto fronts, and regret are computed separately for the two suites.

## Seven-dimensional vector reward

At each transition the wrapper returns

\[
\mathbf r_t=[\Delta c_s,\Delta c_r,\Delta c_{strike},\Delta c_{recovery},
 \tilde r_{makespan},-\tilde c_{comm},-\tilde c_{safety}].
\]

The first four terms are increments in deadline-valid priority-weighted unique assignment coverage. The first accepted assignment of a task before the deadline contributes `exp(-assignment_time/max(1, deadline))` to its type, normalized by the number of task instances observed as available. This represents not merely whether a task was eventually touched, but which task types received earlier allocation priority, while avoiding double counting reallocation attempts. The report also records unweighted unique assignments and assigned processing effort by type. The efficiency term is the frozen paper makespan improvement divided by `max(1, deadline, previous_makespan)`. Communication is the increment in synchronized communication events divided by `max_decisions`. Safety is the increment in invalid actions, UAV failure events, and unresolved reallocation cost divided by `max_tasks + max_uavs`. The scalar reward exposed to a scalar PPO interface is `p · r_t`; PCRL training retains the complete vector.

The safety and failure terms are reported separately from controllability. Exogenous failures are not interpreted as a learned policy advantage; they are included to make the optimization vector auditable and to prevent hidden failure costs from being omitted.

## Coverage and preference error

For type `i`, let `A_i` be the number of tasks observed as available and let `P_i` be the sum of `exp(-t/max(1,deadline))` for its uniquely assigned tasks. The primary controllability coverage is `q_i=P_i/max(A_i,1)` and the primary realized allocation-priority mix is `q/sum(q)`. Unweighted unique assignment mix and assigned processing-effort mix are secondary controllability diagnostics. Deadline completion coverage `D_i/max(A_i,1)`, raw completion mix, and reachable-normalized completion mix remain efficiency/safety diagnostics. If no task is assigned, the realized mix is the zero vector and its error is explicitly reported rather than hidden.

For target task profile `p`, report

\[
 E_1=\|q/\sum q-p\|_1,\quad
 E_2=\|q/\sum q-p\|_2,\quad
 S=\frac{q^Tp}{\|q\|_2\|p\|_2}.
\]

`preference_l1`, `preference_l2`, and `preference_cosine` use the dependency-feasible target. The corresponding `*_nominal` fields compare the same realized mix against the unprojected user request and must not be substituted for the primary feasible-target gate. The reports include raw, availability-normalized, and reachable-normalized values, plus minimum per-type coverage. A controllability claim is invalid if it is accompanied by a material loss of required-task coverage or GPPO efficiency.

## PCRL update

The model outputs a seven-dimensional conditioned value vector. Vector GAE uses `A=R−V(s,p)`. For linear-scalarization controls, the actor direction is `A p`. PCRL uses episode-grouped PreCo: each preference id obtains a separate objective-direction matrix, a subgradient of the ray-similarity objective

\[
\Psi(p,v)=-\tfrac12\|\max_i(v_i/p_i)p-v\|^2,
\]

and a min-norm MOO correction blended with the explicit linear preference direction. Grouping is required because the batch may contain different preferences; using `preferences[0]` for the entire batch is prohibited. `sdmgrad` removes the similarity bias as an optional MOO ablation.

Because the UAV task chains can make objective advantages highly correlated, PCRL-v0 also uses a masked action-type alignment auxiliary. It aggregates probability mass only over actions already declared legal by the frozen action mask and minimizes cross-entropy to the task preference restricted to currently legal task types. It cannot unlock dependencies or make an illegal assignment. This is a project-specific identifiability improvement, not a claim that the PCRL paper specifies this exact loss; all GPPO efficiency and minimum-coverage gates remain mandatory to prevent superficial alignment.

## Pareto and regret reporting

Each profile/scale point is projected to seven maximization objectives: four coverage values, a monotone makespan quality `1/(1+makespan/deadline)`, communication quality `1−communication_events/max_decisions`, and safety quality `1−invalid_actions/max_decisions` (clipped to `[0,1]`). Hypervolume is calculated after this stated normalization and with the zero reference. IGD uses the empirical nondominated front of all methods under the same profile/scale. Preference regret is the difference between the best observed method's mean scalarized utility and the method's utility under the same profile/scale; it is not a claim of comparison with an unimplemented oracle.

## Evidence boundary

The PCRL paper motivates preference-conditioned policies, vector critics, continuous simplex sampling, PreCo/MOO updates, unseen preferences, and multi-seed evaluation. It does not uniquely specify this project's task generator, weak-communication cache, event timing, deadline, reward scaling, task floor, fixed padding/noop action, or failure model. Those are project protocol choices and must be labeled as such in reports. World-model event prediction is deliberately outside PCRL-v0.
