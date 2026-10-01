"""A simple goal-distance heuristic for grid-world demonstrations."""

from __future__ import annotations

from collections.abc import Sequence

from rl_from_scratch.core.types import Action, State
from rl_from_scratch.environments import GridWorld


class GoalDirectedPolicy:
    def __init__(self, env: GridWorld) -> None:
        self.env = env

    def probabilities(self, state: State, actions: Sequence[Action]) -> dict[Action, float]:
        if not actions:
            return {}
        def score(action: Action) -> float:
            return sum(p * (reward - abs(next_state[0] - self.env.goal[0]) - abs(next_state[1] - self.env.goal[1]))
                       for p, next_state, reward, _ in self.env.transitions(state, action))
        chosen = max(actions, key=score)
        return {action: float(action == chosen) for action in actions}
