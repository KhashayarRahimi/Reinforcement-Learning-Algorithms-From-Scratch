"""Small contracts shared by algorithms. States and actions must be hashable."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Mapping, Protocol, Sequence

State = Hashable
Action = Hashable


@dataclass(frozen=True)
class Transition:
    state: State
    action: Action
    reward: float
    next_state: State
    terminated: bool


@dataclass
class Episode:
    transitions: list[Transition] = field(default_factory=list)

    @property
    def return_(self) -> float:
        return sum(step.reward for step in self.transitions)

    @property
    def length(self) -> int:
        return len(self.transitions)

    @property
    def terminated(self) -> bool:
        return bool(self.transitions and self.transitions[-1].terminated)


@dataclass
class Result:
    values: dict[State, float] = field(default_factory=dict)
    action_values: dict[tuple[State, Action], float] = field(default_factory=dict)
    policy: dict[State, Action] = field(default_factory=dict)
    episode_returns: list[float] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    parameters: object | None = None


class Environment(Protocol):
    @property
    def states(self) -> Sequence[State]: ...

    def actions(self, state: State) -> Sequence[Action]: ...

    def reset(self, seed: int | None = None) -> State: ...

    def step(self, action: Action) -> tuple[State, float, bool]: ...


class ModelEnvironment(Environment, Protocol):
    def transitions(self, state: State, action: Action) -> Sequence[tuple[float, State, float, bool]]: ...


class Policy(Protocol):
    def probabilities(self, state: State, actions: Sequence[Action]) -> Mapping[Action, float]: ...
