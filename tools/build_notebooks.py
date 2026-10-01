"""Create short, reproducible notebooks that import the package."""

from __future__ import annotations

import json
from pathlib import Path


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}


EXPERIMENTS = {
    "01_dynamic_programming": (
        "Policy evaluation computes a Bellman expectation; policy iteration alternates evaluation and greedy improvement.",
        "from rl_from_scratch.dynamic_programming import PolicyEvaluation, PolicyIteration\nfrom rl_from_scratch.policies import TabularPolicy\nenv = GridWorld(3, 3, goal=(2, 2))\nrandom_policy = TabularPolicy({s: {a: 0.25 for a in env.actions(s)} for s in env.states if env.actions(s)})\nagent = PolicyEvaluation(env, random_policy, gamma=0.9)\nresult = agent.run()\noptimal = PolicyIteration(env, gamma=0.9).run()",
        "print('Random-policy value:', result.values[env.start])\nprint('Improved first action:', optimal.policy[env.start])",
        "import numpy as np\nimport matplotlib.pyplot as plt\nvalues = np.array([[optimal.values[r, c] for c in range(env.cols)] for r in range(env.rows)])\nplt.imshow(values); plt.colorbar(label='Optimal state value'); plt.title('Policy iteration'); plt.show()",
    ),
    "02_monte_carlo": (
        "First-visit MC averages complete returns. Epsilon-soft control improves a sampled action-value estimate.",
        "from rl_from_scratch.monte_carlo import FirstVisitPrediction, OnPolicyControl, OffPolicyPrediction, OffPolicyControl\nfrom rl_from_scratch.policies import TabularPolicy\nenv = GridWorld(3, 3)\nrandom_policy = TabularPolicy({s: {a: 0.25 for a in env.actions(s)} for s in env.states if env.actions(s)})\nagent = OnPolicyControl(env, gamma=0.9, epsilon=0.1, seed=7)\nresult = agent.train(300, max_steps=100)",
        "print('Greedy first action:', result.policy[env.start])\nprint('First ten returns:', result.episode_returns[:10])",
        "import matplotlib.pyplot as plt\nplt.plot(result.episode_returns, alpha=0.5); plt.xlabel('Episode'); plt.ylabel('Return'); plt.title('Monte Carlo control'); plt.show()",
    ),
    "03_temporal_difference": (
        "Q-learning bootstraps from the best successor action; Sarsa bootstraps from the action actually selected.",
        "from rl_from_scratch.temporal_difference import TDZero, Sarsa, QLearning, DoubleQLearning\nenv = GridWorld(3, 3)\nagent = QLearning(env, alpha=0.2, gamma=0.9, epsilon=0.1, seed=7)\nresult = agent.train(300, max_steps=100)",
        "print('Q at start:', {a: result.action_values.get((env.start, a), 0) for a in env.actions(env.start)})\nprint('Greedy first action:', result.policy[env.start])",
        "import matplotlib.pyplot as plt\nplt.plot(result.episode_returns); plt.xlabel('Episode'); plt.ylabel('Return'); plt.title('Q-learning'); plt.show()",
    ),
    "04_n_step": (
        "The n-step target sums n discounted rewards, then bootstraps. Tree Backup and Q(σ) mix expected and sampled values.",
        "from rl_from_scratch.n_step import NStepTD, NStepSarsa, OffPolicyNStepSarsa, TreeBackup, QSigma\nenv = GridWorld(3, 3)\nagent = NStepSarsa(env, n=3, alpha=0.2, gamma=0.9, epsilon=0.1, seed=7)\nresult = agent.train(300, max_steps=100)",
        "print('Greedy first action:', result.policy[env.start])\nprint('Final ten returns:', result.episode_returns[-10:])",
        "import matplotlib.pyplot as plt\nplt.plot(result.episode_returns); plt.xlabel('Episode'); plt.ylabel('Return'); plt.title('Three-step Sarsa'); plt.show()",
    ),
    "05_planning": (
        "Dyna-Q learns from real transitions and repeats Q-learning updates sampled from its model. Sweeping gives priority to large TD errors.",
        "from rl_from_scratch.planning import DeterministicModel, RandomSamplePlanning, DynaQ, PrioritizedSweeping\nenv = GridWorld(3, 3)\nagent = DynaQ(env, n_planning=5, alpha=0.2, gamma=0.9, epsilon=0.1, seed=7)\nresult = agent.train(200, max_steps=100)",
        "print('Observed model entries:', len(agent.model.experience))\nprint('Greedy first action:', result.policy[env.start])",
        "import matplotlib.pyplot as plt\nplt.plot(result.episode_lengths); plt.xlabel('Episode'); plt.ylabel('Steps'); plt.title('Dyna-Q planning'); plt.show()",
    ),
    "06_approximation_prediction": (
        "A linear value function uses explicit features. Semi-gradient TD updates its weights toward a one-step target.",
        "from rl_from_scratch.function_approximation import GradientMonteCarlo, SemiGradientTD, NStepSemiGradientTD, LSTD\nfrom rl_from_scratch.value_functions import GridFeatures, LinearValue\nfrom rl_from_scratch.policies import TabularPolicy\nenv = GridWorld(3, 3)\npolicy = TabularPolicy({s: {a: 0.25 for a in env.actions(s)} for s in env.states if env.actions(s)})\nvalue = LinearValue(GridFeatures(env.rows, env.cols, env.goal), dimension=3)\nagent = SemiGradientTD(env, policy, value, alpha=0.01, gamma=0.9, seed=7)\nresult = agent.train(300, max_steps=100)",
        "print('Weights:', result.parameters)\nprint('Estimated start value:', result.values[env.start])",
        "import matplotlib.pyplot as plt\nplt.plot(result.episode_returns); plt.xlabel('Episode'); plt.ylabel('Return'); plt.title('Linear TD prediction'); plt.show()",
    ),
    "07_approximation_control": (
        "Semi-gradient Sarsa updates a linear action-value function using the sampled next action.",
        "from rl_from_scratch.function_approximation import SemiGradientSarsa, NStepSemiGradientSarsa, DifferentialSarsa, DifferentialNStepSarsa\nfrom rl_from_scratch.value_functions import GridFeatures, LinearActionValue\nenv = GridWorld(3, 3)\nbase = GridFeatures(env.rows, env.cols, env.goal)\nactions = tuple(env.actions(env.start))\ndef features(state, action):\n    import numpy as np\n    x = np.zeros(3 * len(actions))\n    x[actions.index(action)*3:(actions.index(action)+1)*3] = base(state)\n    return x\nvalue = LinearActionValue(features, dimension=3*len(actions))\nagent = SemiGradientSarsa(env, value, alpha=0.01, gamma=0.9, epsilon=0.1, seed=7)\nresult = agent.train(300, max_steps=100)",
        "print('Greedy first action:', result.policy[env.start])\nprint('Weight norm:', sum(result.parameters ** 2) ** 0.5)",
        "import matplotlib.pyplot as plt\nplt.plot(result.episode_returns); plt.xlabel('Episode'); plt.ylabel('Return'); plt.title('Linear Sarsa control'); plt.show()",
    ),
    "08_eligibility_traces": (
        "Eligibility traces assign a TD error to recently active features. Dutch traces add a correction for true-online learning.",
        "from rl_from_scratch.eligibility_traces import SemiGradientTDLambda, TrueOnlineTDLambda, SarsaLambda, TrueOnlineSarsaLambda\nfrom rl_from_scratch.value_functions import GridFeatures, LinearValue\nfrom rl_from_scratch.policies import TabularPolicy\nenv = GridWorld(3, 3)\npolicy = TabularPolicy({s: {a: 0.25 for a in env.actions(s)} for s in env.states if env.actions(s)})\nvalue = LinearValue(GridFeatures(env.rows, env.cols, env.goal), dimension=3)\nagent = TrueOnlineTDLambda(env, policy, value, alpha=0.01, gamma=0.9, lam=0.8, seed=7)\nresult = agent.train(300, max_steps=100)",
        "print('Weights:', result.parameters)\nprint('Start value:', result.values[env.start])",
        "import matplotlib.pyplot as plt\nplt.plot(result.episode_returns); plt.xlabel('Episode'); plt.ylabel('Return'); plt.title('True online TD(lambda)'); plt.show()",
    ),
    "09_policy_gradient": (
        "REINFORCE weights the score gradient by a sampled return. Actor–critic replaces that return with a TD error.",
        "from rl_from_scratch.policy_gradient import REINFORCE, REINFORCEWithBaseline, OneStepActorCritic, EpisodicTraceActorCritic, ContinuingTraceActorCritic\nfrom rl_from_scratch.policies import LinearSoftmaxPolicy\nfrom rl_from_scratch.value_functions import GridFeatures, LinearValue\nenv = GridWorld(3, 3)\nbase = GridFeatures(env.rows, env.cols, env.goal)\nactions = tuple(env.actions(env.start))\ndef actor_features(state, action):\n    import numpy as np\n    x = np.zeros(3*len(actions))\n    x[actions.index(action)*3:(actions.index(action)+1)*3] = base(state)\n    return x\nactor = LinearSoftmaxPolicy(actor_features, dimension=3*len(actions))\ncritic = LinearValue(base, dimension=3)\nagent = REINFORCEWithBaseline(env, actor, critic=critic, actor_alpha=0.001, critic_alpha=0.01, gamma=0.9, seed=7)\nresult = agent.train(300, max_steps=100)",
        "print('Greedy first action:', result.policy[env.start])\nprint('Critic start value:', result.values[env.start])",
        "import matplotlib.pyplot as plt\nplt.plot(result.episode_returns); plt.xlabel('Episode'); plt.ylabel('Return'); plt.title('REINFORCE with baseline'); plt.show()",
    ),
}


def main() -> None:
    destination = Path("notebooks")
    destination.mkdir(exist_ok=True)
    for name, (explanation, experiment, inspect, visualization) in EXPERIMENTS.items():
        cells = [
            markdown(f"# {name[3:].replace('_', ' ').title()}\n\n{explanation}\n\nThe code imports the implementation from `rl_from_scratch` and keeps the experiment here."),
            code("from rl_from_scratch import GridWorld"),
            markdown("## Define the environment, algorithm, and training run"),
            code(experiment),
            markdown("## Inspect the learned result"),
            code(inspect),
            markdown("## Visualize learning"),
            code(visualization),
        ]
        notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
        (destination / f"{name}.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
