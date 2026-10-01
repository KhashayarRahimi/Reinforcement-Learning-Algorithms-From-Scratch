# Reinforcement Learning Algorithms From Scratch

Readable Python implementations of reinforcement-learning algorithms following Sutton and Barto's *Reinforcement Learning: An Introduction*. The package uses NumPy for linear algebra and no RL framework.

## Philosophy

Each algorithm exposes its update rule. Environments own transitions and rewards; policies own action probabilities; value functions own parameters. Algorithms compose these components. Notebooks are experiments, while `src/` is the implementation. Original files are retained in `archive/` for comparison; their outputs were not used as correctness references because several implementations have inconsistent transitions or global-state dependencies.

## Installation

Python 3.10 or newer is required.

```bash
python -m pip install -e .
python -m pip install -e '.[dev]'  # tests, plotting, Jupyter
```

## Quick start

```python
from rl_from_scratch import GridWorld, QLearning

env = GridWorld.random(5, 4, holes=4, seed=39)
agent = QLearning(env, alpha=0.1, gamma=0.99, epsilon=0.1, seed=39)
result = agent.train(episodes=10_000, max_steps=500)
print(result.policy[env.start], result.episode_returns[-10:])
```

For exact planning with a known model:

```python
from rl_from_scratch.dynamic_programming import PolicyIteration

optimal = PolicyIteration(env, gamma=0.9).run()
print(optimal.policy[env.start], optimal.values[env.start])
```

## Architecture and algorithms

| Package | Algorithms |
|---|---|
| `dynamic_programming` | Policy evaluation, policy iteration |
| `monte_carlo` | First-visit prediction, epsilon-soft control, weighted off-policy prediction and control |
| `temporal_difference` | TD(0), Sarsa, Q-learning, Double Q-learning |
| `n_step` | n-step TD, Sarsa, off-policy Sarsa, Tree Backup, Q(σ) |
| `planning` | Random-sample Q-planning, Dyna-Q, prioritized sweeping |
| `function_approximation` | Gradient MC, semi-gradient TD and Sarsa, n-step variants, LSTD, differential Sarsa variants |
| `eligibility_traces` | Semi-gradient TD(λ), true online TD(λ), Sarsa(λ), true online Sarsa(λ) |
| `policy_gradient` | REINFORCE, baseline, one-step actor–critic, episodic and continuing trace actor–critic |

Shared components are in `core/`, `environments/`, `policies/`, and `value_functions/`. `Environment` supplies `reset`, `step`, `actions`, and finite `states`; model-based methods also need `transitions(s, a)` returning `(probability, next_state, reward, terminated)`. `Result` contains learned values, a greedy policy, episode returns and lengths, and optional parameters.

## Mathematical conventions

- Terminal transitions have zero bootstrap value. A time limit ends collection without changing the environment's terminal flag.
- `GridWorld` uses four named actions. Invalid moves leave the state unchanged with the step reward. A hole gives its reward and returns to start by default; `hole_resets=False` makes it terminal.
- `slip` is the chance to replace the requested action with a uniformly sampled action. The transition model aggregates equal outcomes.
- Seeds use local random generators. Episodes have a configurable `max_steps` limit.
- Monte Carlo methods skip parameter updates for episodes cut off by `max_steps`, because the complete return is unknown.
- Continuing average-reward methods require a nonterminal environment and report `average_reward` in `Result.values`.
- n-step methods collect a trajectory and apply updates in time order, keeping returns explicit. This update timing differs from a fully interleaved online implementation.
- `Result.episode_returns` stores undiscounted reward sums. `OffPolicyNStepSarsa` requires explicit behavior and target policies.

See [equation notes](docs/equations.md), [migration notes](docs/migration.md), and the [extension guide](docs/extending.md).

## Repository structure

```text
src/rl_from_scratch/   reusable package
tests/                 small equation-based pytest tests
examples/              scripts runnable without Jupyter
notebooks/             chapter experiments and plots
docs/                  equations, migration, extension guide
archive/               original notebooks and modules
```

## Run and test

```bash
python examples/grid_q_learning.py
python -m pytest
```

Open notebooks after installing the package in the same environment. A new algorithm should include a hand-calculated small-MDP test and an imported notebook demonstration.
