import pytest

from rl_from_scratch.core import generate_episode
from rl_from_scratch.dynamic_programming import PolicyEvaluation, PolicyIteration
from rl_from_scratch.environments import FiniteMDP
from rl_from_scratch.monte_carlo import FirstVisitPrediction, OffPolicyPrediction
from rl_from_scratch.n_step import NStepTD, NStepSarsa, OffPolicyNStepSarsa, TreeBackup, QSigma
from rl_from_scratch.policies import DeterministicPolicy


def chain():
    return FiniteMDP({("s0", "a"): [(1, "s1", 1, False)], ("s1", "a"): [(1, "end", 2, True)]}, "s0")


def test_bellman_evaluation_and_policy_iteration():
    env = FiniteMDP({("s", "a"): [(1, "end", 1, True)], ("s", "b"): [(1, "end", 2, True)]}, "s")
    assert PolicyEvaluation(env, DeterministicPolicy({"s": "a"})).run().values["s"] == 1
    assert PolicyIteration(env).run().policy == {"s": "b"}


def test_episode_and_first_visit_return():
    env = chain()
    policy = DeterministicPolicy({"s0": "a", "s1": "a"})
    episode = generate_episode(env, policy)
    assert episode.length == 2 and episode.return_ == 3
    values = FirstVisitPrediction(env, policy, gamma=0.5).train(1).values
    assert values == {"s0": 2, "s1": 2}


def test_monte_carlo_does_not_learn_from_incomplete_episode():
    loop = FiniteMDP({("s", "a"): [(1, "s", 1, False)]}, "s")
    result = FirstVisitPrediction(loop, DeterministicPolicy({"s": "a"})).train(2, max_steps=3)
    assert result.values == {}
    assert result.episode_lengths == [3, 3]


def test_off_policy_weighted_prediction_and_n_step_targets():
    env = chain()
    policy = DeterministicPolicy({"s0": "a", "s1": "a"})
    assert OffPolicyPrediction(env, policy, policy, gamma=0.5).train(1).action_values["s0", "a"] == 2
    assert NStepTD(env, policy, n=2, alpha=1, gamma=0.5).train(1).values["s0"] == 2
    assert NStepSarsa(env, n=2, alpha=1, gamma=0.5, epsilon=0).train(1).action_values["s0", "a"] == 2
    assert TreeBackup(env, n=2, alpha=1, gamma=0.5, epsilon=0).train(1).action_values["s0", "a"] == 2
    assert QSigma(env, n=2, alpha=1, gamma=0.5, epsilon=0, sigma=1).train(1).action_values["s0", "a"] == 2
    assert OffPolicyNStepSarsa(env, policy, policy, n=2, alpha=1, gamma=0.5).train(1).action_values["s0", "a"] == 2
