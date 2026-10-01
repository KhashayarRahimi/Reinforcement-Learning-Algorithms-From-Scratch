# Extending the package

## New environment

Implement `states`, `actions(state)`, `reset(seed=None)`, and `step(action)` from `core/types.py`. Return `(next_state, reward, terminated)`. For dynamic programming also implement `transitions(state, action)` as a probability distribution. Test that sampling and the model agree, probabilities sum to one, and terminal rewards are correct.

## New policy

Implement `probabilities(state, actions)`. Include every available action, including zero-probability actions, and ensure probabilities are nonnegative and total one. Sampling uses an explicitly owned random generator. Parameterized policies can add `grad_log` for policy-gradient updates.

## New value function

Tabular classes expose `__getitem__` and `update(target, alpha)`. Linear classes expose feature vectors, weights, prediction, and update. Validate feature dimensions. Test a single hand-calculated update.

## New algorithm

Place the class in its mathematical family and export it from that package. Pass dependencies and hyperparameters to the constructor; return `Result` from `train(episodes, max_steps=...)`, or `run()` for exact planning. Put the update equation in the docstring. Terminal bootstrap is zero. Test a one- or two-step MDP, an edge case, and a practical convergence property if inexpensive. Add a notebook cell importing the class.
