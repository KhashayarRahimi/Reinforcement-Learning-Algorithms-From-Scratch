"""An explicit finite MDP for teaching and exact Bellman tests."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence

from rl_from_scratch.core.types import Action, State


class FiniteMDP:
    def __init__(
        self, transitions: Mapping[tuple[State, Action], Sequence[tuple[float, State, float, bool]]],
        start: State, *, seed: int | None = None,
    ) -> None:
        self._model = {key: tuple(outcomes) for key, outcomes in transitions.items()}
        if not self._model:
            raise ValueError("MDP must have at least one transition")
        states = {start}
        for (state, _), outcomes in self._model.items():
            if not outcomes or any(p < 0 for p, *_ in outcomes):
                raise ValueError("outcomes must have nonnegative probabilities")
            if abs(sum(p for p, *_ in outcomes) - 1.0) > 1e-9:
                raise ValueError("transition probabilities must sum to one")
            states.add(state)
            states.update(next_state for _, next_state, _, _ in outcomes)
        self._states = tuple(sorted(states, key=repr))
        self.start = start
        self._state = start
        self._rng = random.Random(seed)

    @property
    def states(self) -> tuple[State, ...]:
        return self._states

    def actions(self, state: State) -> tuple[Action, ...]:
        return tuple(action for s, action in self._model if s == state)

    def transitions(self, state: State, action: Action) -> tuple[tuple[float, State, float, bool], ...]:
        return self._model[(state, action)]

    def reset(self, seed: int | None = None) -> State:
        if seed is not None:
            self._rng.seed(seed)
        self._state = self.start
        return self._state

    def step(self, action: Action) -> tuple[State, float, bool]:
        outcomes = self.transitions(self._state, action)
        index = self._rng.choices(range(len(outcomes)), weights=[o[0] for o in outcomes])[0]
        _, self._state, reward, done = outcomes[index]
        return self._state, reward, done
