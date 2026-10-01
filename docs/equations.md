# Equation map

Notation: `s` is a state, `a` an action, `r` the next reward, `γ` the discount, `α` the step size, `v` a state value, and `q` an action value. Terminal successors have zero bootstrap value.

| Family | Core update |
|---|---|
| Policy evaluation | `v(s) ← Σ_a π(a|s) Σ_{s',r} p(s',r|s,a)[r + γv(s')]` |
| TD(0) | `v(s) ← v(s) + α[r + γv(s') − v(s)]` |
| Sarsa | `q(s,a) ← q(s,a) + α[r + γq(s',a') − q(s,a)]` |
| Q-learning | `q(s,a) ← q(s,a) + α[r + γ max_b q(s',b) − q(s,a)]` |
| Double Q-learning | Choose the maximizing action with one table, evaluate with the other. |
| n-step TD/Sarsa | Sum discounted rewards through `t+n`, then bootstrap if nonterminal. |
| Off-policy n-step Sarsa | Multiply the n-step TD error by the intervening action-probability ratios. |
| Tree Backup / Q(σ) | Backward recursion mixes sampled and expected action values; `σ=0` is Tree Backup. |
| Dyna-Q | One real Q-learning update, then `n_planning` updates from a learned model. |
| LSTD | Solve `(Σ x_t(x_t−γx_{t+1})ᵀ+λI)w = Σ x_t r_{t+1}`. |
| Semi-gradient TD(λ) | `e ← γλe+x(s)`; `w ← w+αδe`. |
| True online TD(λ) | Use a Dutch trace and a correction for the prior prediction. |
| REINFORCE | `θ ← θ + αγ^t G_t ∇log π_θ(a_t|s_t)`. |
| Actor–critic | Critic uses a TD error `δ`; actor uses `δ∇log π_θ(a|s)`, optionally with traces. |

The implementation documents precise terminal and sampling cases beside each update. Tests use one- and two-step MDPs with inspectable targets.
