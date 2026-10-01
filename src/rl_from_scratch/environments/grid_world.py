"""Finite grid world with explicit slip, wall and hole semantics."""

from __future__ import annotations

import random

from rl_from_scratch.core.types import State

GridState = tuple[int, int]
GridAction = str
ACTIONS: dict[GridAction, GridState] = {
    "down": (1, 0), "up": (-1, 0), "right": (0, 1), "left": (0, -1),
}


class GridWorld:
    def __init__(
        self, rows: int = 5, cols: int = 4, *, start: GridState = (0, 0),
        goal: GridState | None = None, holes: tuple[GridState, ...] = (),
        slip: float = 0.0, step_reward: float = -1.0, goal_reward: float = 10.0,
        hole_reward: float = -3.0, hole_resets: bool = True, seed: int | None = None,
    ) -> None:
        if rows < 1 or cols < 1 or not 0 <= slip <= 1:
            raise ValueError("rows/cols must be positive and slip in [0, 1]")
        goal = goal if goal is not None else (rows - 1, cols - 1)
        valid = lambda s: isinstance(s, tuple) and len(s) == 2 and 0 <= s[0] < rows and 0 <= s[1] < cols
        if not valid(start) or not valid(goal) or start == goal or any(not valid(h) for h in holes):
            raise ValueError("start, goal and holes must be distinct valid grid locations")
        if start in holes or goal in holes or len(set(holes)) != len(holes):
            raise ValueError("start and goal cannot be holes")
        self.rows, self.cols, self.start, self.goal = rows, cols, start, goal
        self.holes, self.slip = frozenset(holes), slip
        self.step_reward, self.goal_reward, self.hole_reward = step_reward, goal_reward, hole_reward
        self.hole_resets = hole_resets
        self._states = tuple((r, c) for r in range(rows) for c in range(cols))
        self._state = start
        self._rng = random.Random(seed)

    @classmethod
    def random(
        cls, rows: int, cols: int, holes: int = 0, *, seed: int = 0,
        start: GridState | None = None, goal: GridState | None = None,
        random_start_goal: bool = False, **kwargs: object,
    ) -> "GridWorld":
        """Generate a seeded grid with a monotone start-to-goal path free of holes."""
        if rows < 1 or cols < 1 or rows * cols < 2:
            raise ValueError("grid needs at least two cells")
        rng = random.Random(seed)
        all_cells = [(r, c) for r in range(rows) for c in range(cols)]
        if random_start_goal:
            if start is not None or goal is not None:
                raise ValueError("choose either explicit endpoints or random_start_goal")
            start, goal = rng.sample(all_cells, 2)
        start = (0, 0) if start is None else start
        goal = (rows - 1, cols - 1) if goal is None else goal
        if start not in all_cells or goal not in all_cells or start == goal:
            raise ValueError("start and goal must be distinct grid cells")
        path = {start}
        r, c = start
        while (r, c) != goal:
            moves = ([(r + (1 if goal[0] > r else -1), c)] if r != goal[0] else []) + ([(r, c + (1 if goal[1] > c else -1))] if c != goal[1] else [])
            r, c = rng.choice(moves)
            path.add((r, c))
        free = [cell for cell in all_cells if cell not in path]
        if not 0 <= holes <= len(free):
            raise ValueError("too many holes for a guaranteed open path")
        return cls(rows, cols, start=start, goal=goal, holes=tuple(rng.sample(free, holes)), seed=seed, **kwargs)

    @property
    def states(self) -> tuple[State, ...]:
        return self._states

    def actions(self, state: State) -> tuple[GridAction, ...]:
        if state not in self._states:
            raise ValueError("state is outside the grid")
        return () if state == self.goal or (state in self.holes and not self.hole_resets) else tuple(ACTIONS)

    def transitions(self, state: State, action: GridAction) -> tuple[tuple[float, GridState, float, bool], ...]:
        if action not in ACTIONS or not self.actions(state):
            raise ValueError("invalid action or terminal state")
        outcomes: dict[tuple[GridState, float, bool], float] = {}
        for actual in ACTIONS:
            probability = (1 - self.slip if actual == action else 0) + self.slip / len(ACTIONS)
            if probability == 0:
                continue
            dr, dc = ACTIONS[actual]
            candidate = (state[0] + dr, state[1] + dc)
            if candidate not in self._states:
                candidate = state
            if candidate == self.goal:
                next_state, reward, done = candidate, self.goal_reward, True
            elif candidate in self.holes:
                next_state, reward, done = (self.start if self.hole_resets else candidate), self.hole_reward, not self.hole_resets
            else:
                next_state, reward, done = candidate, self.step_reward, False
            key = next_state, reward, done
            outcomes[key] = outcomes.get(key, 0.0) + probability
        return tuple((p, s, r, d) for (s, r, d), p in outcomes.items())

    def reset(self, seed: int | None = None) -> GridState:
        if seed is not None:
            self._rng.seed(seed)
        self._state = self.start
        return self._state

    def step(self, action: GridAction) -> tuple[GridState, float, bool]:
        outcomes = self.transitions(self._state, action)
        selected = self._rng.choices(outcomes, weights=[o[0] for o in outcomes])[0]
        _, self._state, reward, done = selected
        return self._state, reward, done
