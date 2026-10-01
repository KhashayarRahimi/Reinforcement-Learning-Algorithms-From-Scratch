"""Action distributions used by tabular and sampling algorithms."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from rl_from_scratch.core.types import Action, State


def _check_distribution(distribution: Mapping[Action, float], actions: Sequence[Action]) -> None:
    if set(distribution) != set(actions) or any(p < 0 for p in distribution.values()) or abs(sum(distribution.values()) - 1) > 1e-9:
        raise ValueError("policy must give nonnegative probabilities summing to one over available actions")


class TabularPolicy:
    def __init__(self, distributions: Mapping[State, Mapping[Action, float]]) -> None:
        self.distributions = {s: dict(probs) for s, probs in distributions.items()}

    def probabilities(self, state: State, actions: Sequence[Action]) -> dict[Action, float]:
        probabilities = dict(self.distributions[state])
        _check_distribution(probabilities, actions)
        return probabilities


class DeterministicPolicy:
    def __init__(self, actions: Mapping[State, Action]) -> None:
        self.chosen = dict(actions)

    def probabilities(self, state: State, actions: Sequence[Action]) -> dict[Action, float]:
        chosen = self.chosen[state]
        if chosen not in actions:
            raise ValueError("chosen action is unavailable")
        return {a: float(a == chosen) for a in actions}


class EpsilonGreedyPolicy:
    def __init__(self, action_values: Mapping[tuple[State, Action], float], epsilon: float = 0.1) -> None:
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be in [0, 1]")
        self.action_values, self.epsilon = action_values, epsilon

    def probabilities(self, state: State, actions: Sequence[Action]) -> dict[Action, float]:
        if not actions:
            return {}
        best = max(self.action_values.get((state, a), 0.0) for a in actions)
        greedy = [a for a in actions if self.action_values.get((state, a), 0.0) == best]
        return {a: self.epsilon / len(actions) + ((1 - self.epsilon) / len(greedy) if a in greedy else 0.0) for a in actions}


class SoftmaxPolicy:
    def __init__(self, preferences: Mapping[tuple[State, Action], float], temperature: float = 1.0) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.preferences, self.temperature = preferences, temperature

    def probabilities(self, state: State, actions: Sequence[Action]) -> dict[Action, float]:
        if not actions:
            return {}
        logits = [self.preferences.get((state, a), 0.0) / self.temperature for a in actions]
        baseline = max(logits)
        weights = [math.exp(x - baseline) for x in logits]
        total = sum(weights)
        return {a: w / total for a, w in zip(actions, weights)}
