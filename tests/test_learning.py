import numpy as np
import pytest

from rl_from_scratch.eligibility_traces import SemiGradientTDLambda, TrueOnlineTDLambda
from rl_from_scratch.environments import FiniteMDP, GridWorld
from rl_from_scratch.function_approximation import LSTD, SemiGradientTD, DifferentialSarsa
from rl_from_scratch.planning import DeterministicModel, DynaQ, PrioritizedSweeping, RandomSamplePlanning
from rl_from_scratch.policies import DeterministicPolicy, LinearSoftmaxPolicy
from rl_from_scratch.policy_gradient import REINFORCE, ContinuingTraceActorCritic
from rl_from_scratch.temporal_difference import QLearning, Sarsa, DoubleQLearning, TDZero
from rl_from_scratch.value_functions import LinearValue, LinearActionValue


def one_step():
    return FiniteMDP({("s", "a"): [(1, "end", 2, True)]}, "s")


@pytest.mark.parametrize("algorithm", [QLearning, Sarsa, DoubleQLearning])
def test_td_control_one_step_target(algorithm):
    result = algorithm(one_step(), alpha=1, epsilon=0, seed=1).train(1)
    assert result.action_values["s", "a"] == 2


def test_td_prediction_and_episode_cap():
    env = one_step()
    policy = DeterministicPolicy({"s": "a"})
    assert TDZero(env, policy, alpha=1).train(1).values["s"] == 2
    loop = FiniteMDP({("s", "a"): [(1, "s", -1, False)]}, "s")
    assert QLearning(loop).train(1, max_steps=3).episode_lengths == [3]


def test_q_learning_converges_on_one_state_loop_and_seeded_runs_match():
    loop = FiniteMDP({("s", "a"): [(1, "s", 1, False)]}, "s")
    value = QLearning(loop, alpha=0.1, gamma=0.5, epsilon=0).train(200, max_steps=1)
    assert value.action_values["s", "a"] == pytest.approx(2, rel=1e-3)
    first = QLearning(GridWorld(2, 2, slip=0.2, seed=3), seed=4).train(20, max_steps=10)
    second = QLearning(GridWorld(2, 2, slip=0.2, seed=3), seed=4).train(20, max_steps=10)
    assert first.episode_returns == second.episode_returns


def test_model_planning_and_predecessors():
    env = one_step()
    model = DeterministicModel()
    model.observe("s", "a", "end", 2, True)
    assert model.predecessors["end"] == {("s", "a")}
    assert DeterministicModel.from_environment(env).experience == model.experience
    assert RandomSamplePlanning(env, model, alpha=1).train(1).action_values["s", "a"] == 2
    assert DynaQ(env, n_planning=1, alpha=1, epsilon=0).train(1).action_values["s", "a"] == 2
    assert PrioritizedSweeping(env, n_planning=1, alpha=1, epsilon=0).train(1).action_values["s", "a"] == 2


def test_linear_td_lstd_and_traces():
    env = one_step()
    policy = DeterministicPolicy({"s": "a"})
    make_value = lambda: LinearValue(lambda s: [1], 1)
    assert SemiGradientTD(env, policy, make_value(), alpha=1).train(1).values["s"] == 2
    assert LSTD(env, policy, make_value(), regularization=1e-9).train(1).values["s"] == pytest.approx(2)
    assert SemiGradientTDLambda(env, policy, make_value(), alpha=1).train(1).values["s"] == 2
    assert TrueOnlineTDLambda(env, policy, make_value(), alpha=1).train(1).values["s"] == 2


def test_reinforce_single_action_zero_gradient():
    actor = LinearSoftmaxPolicy(lambda s, a: [1], 1)
    result = REINFORCE(one_step(), actor).train(1)
    assert result.policy == {"s": "a"}
    np.testing.assert_array_equal(actor.weights, [0])


def test_continuing_methods_reject_terminal_environment():
    env = one_step()
    with pytest.raises(ValueError):
        DifferentialSarsa(env, LinearActionValue(lambda s, a: [1], 1)).train(1)
    with pytest.raises(ValueError):
        ContinuingTraceActorCritic(
            env, LinearSoftmaxPolicy(lambda s, a: [1], 1), LinearValue(lambda s: [1], 1)
        ).train(1)
