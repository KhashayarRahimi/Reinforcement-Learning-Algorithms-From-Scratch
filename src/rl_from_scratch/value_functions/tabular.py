"""Minimal tabular value representations."""

from __future__ import annotations

from rl_from_scratch.core.types import Action, State


class TabularValues:
    def __init__(self, initial: float = 0.0) -> None:
        self.initial = initial
        self.values: dict[State, float] = {}

    def __getitem__(self, state: State) -> float:
        return self.values.get(state, self.initial)

    def update(self, state: State, target: float, alpha: float) -> None:
        self.values[state] = self[state] + alpha * (target - self[state])


class TabularActionValues:
    def __init__(self, initial: float = 0.0) -> None:
        self.initial = initial
        self.values: dict[tuple[State, Action], float] = {}

    def __getitem__(self, key: tuple[State, Action]) -> float:
        return self.values.get(key, self.initial)

    def update(self, key: tuple[State, Action], target: float, alpha: float) -> None:
        self.values[key] = self[key] + alpha * (target - self[key])
