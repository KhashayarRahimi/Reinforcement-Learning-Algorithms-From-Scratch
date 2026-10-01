# Migration record

Original modules and notebooks are retained in `archive/`. Shared grid-world generation, probability sampling, rewards, trajectories, and feature extraction now live in `environments/`, `core/`, `policies/`, and `value_functions/`.

The historical path extraction is available as `utils.follow_policy`, and the goal-distance heuristic as `policies.GoalDirectedPolicy`.

| Historical source | Current destination |
|---|---|
| Dynamic Programming | `dynamic_programming` |
| Monte Carlo, including notebook-only off-policy methods | `monte_carlo` |
| Temporal-Difference Learning | `temporal_difference` |
| n-step Bootstrapping | `n_step` |
| Planning and Learning, including notebook-only prioritized sweeping | `planning` |
| On-policy Prediction/Control with Approximation | `function_approximation` |
| Eligibility Traces | `eligibility_traces` |
| Policy Gradient Methods | `policy_gradient` |

## Algorithm-level mapping

| Historical name | Current class |
|---|---|
| `policy_evaluation`, `policy_iteration` | `PolicyEvaluation`, `PolicyIteration` |
| `monte_carlo_prediction`, `OnPolicy_MC_prediction` | `FirstVisitPrediction`, `OnPolicyControl` |
| `off_monte_carlo_prediction`, `off_MC_control` | `OffPolicyPrediction`, `OffPolicyControl` |
| `TD_zero`, `sarsa`, `Q_learning`, `double_Q_learning` | `TDZero`, `Sarsa`, `QLearning`, `DoubleQLearning` |
| `n_step_TD`, `n_step_sarsa`, `off_policy_n_step_sarsa` | `NStepTD`, `NStepSarsa`, `OffPolicyNStepSarsa` |
| `n_step_Tree_Backup`, `Q_sigma` | `TreeBackup`, `QSigma` |
| `random_sample_Q_planning`, `Dyna_Q`, `prioritized_sweeping` | `RandomSamplePlanning`, `DynaQ`, `PrioritizedSweeping` |
| `gradient_monte_carlo`, `semi_gradient_TD`, `semi_gradient_nstep_TD`, `LSTD` | `GradientMonteCarlo`, `SemiGradientTD`, `NStepSemiGradientTD`, `LSTD` |
| `semi_gradient_sarsa`, `semi_gradient_n_step_sarsa` | `SemiGradientSarsa`, `NStepSemiGradientSarsa` |
| `differential_semi_gradient_sarsa`, `differential_semi_gradient_n_step_sarsa` | `DifferentialSarsa`, `DifferentialNStepSarsa` |
| `semi_gradient_TD_lambda`, `true_online_TD_lambda` | `SemiGradientTDLambda`, `TrueOnlineTDLambda` |
| `sarsa_lambda`, `true_online_sarsa` | `SarsaLambda`, `TrueOnlineSarsaLambda` |
| `monte_carlo_policy_gradient`, `baseline` | `REINFORCE`, `REINFORCEWithBaseline` |
| `one_step_actor_critic`, `eligibility_traces_actor_critic_episodic`, `eligibility_traces_actor_critic_continuing` | `OneStepActorCritic`, `EpisodicTraceActorCritic`, `ContinuingTraceActorCritic` |

Grid generation, neighbor enumeration, action stochasticity, and reward functions are consolidated in `GridWorld`; explicit transition tables use `FiniteMDP`. The various `arbitrary_policy` helpers map to `TabularPolicy` or `DeterministicPolicy`. Trajectory generation maps to `generate_episode`; path extraction maps to `follow_policy`. `extract_features` maps to `GridFeatures`, and `pi_theta` maps to `LinearSoftmaxPolicy`. The historical `H_theta` goal-distance heuristic maps to `GoalDirectedPolicy`. Printed tables and ad hoc notebook outputs are replaced with `Result` data and notebook plots.

The historical `sample_model` helper maps to `DeterministicModel.from_environment` for known deterministic environments; Dyna-Q learns the same model incrementally through `observe`.

The grid-world rules in the README resolve inconsistent reward, wall, hole, and stochastic-action handling in the historical code. Numerical results can differ from historical outputs. Archived files are references, not part of the installed package.
