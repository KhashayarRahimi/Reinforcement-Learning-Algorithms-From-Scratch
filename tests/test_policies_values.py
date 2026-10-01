import numpy as np
import pytest

from rl_from_scratch.policies import EpsilonGreedyPolicy, LinearSoftmaxPolicy, TabularPolicy
from rl_from_scratch.value_functions import LinearValue, TabularActionValues, TabularValues


def test_epsilon_greedy_and_tie_probabilities():
    policy = EpsilonGreedyPolicy({("s", "a"): 2, ("s", "b"): 0}, epsilon=0.2)
    assert policy.probabilities("s", ("a", "b")) == {"a": pytest.approx(0.9), "b": pytest.approx(0.1)}
    assert EpsilonGreedyPolicy({}, epsilon=0).probabilities("s", ("a", "b")) == {"a": 0.5, "b": 0.5}


def test_policy_validation_and_softmax_gradient():
    with pytest.raises(ValueError):
        TabularPolicy({"s": {"a": 0.8}}).probabilities("s", ("a",))
    actor = LinearSoftmaxPolicy(lambda s, a: [1 if a == "a" else 0], 1)
    assert actor.probabilities("s", ("a", "b")) == {"a": 0.5, "b": 0.5}
    assert actor.grad_log("s", "a", ("a", "b"))[0] == pytest.approx(0.5)


def test_tabular_and_linear_updates_match_equations():
    v = TabularValues()
    q = TabularActionValues()
    v.update("s", 4, 0.25)
    q.update(("s", "a"), 2, 0.5)
    assert v["s"] == 1
    assert q["s", "a"] == 1
    linear = LinearValue(lambda s: [1, 2], 2)
    linear.update("s", 2, 0.5)
    np.testing.assert_allclose(linear.weights, [1, 2])
