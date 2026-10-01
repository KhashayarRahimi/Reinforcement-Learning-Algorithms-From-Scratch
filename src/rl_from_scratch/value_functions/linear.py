"""Linear value functions: v(s,w)=w·x(s), q(s,a,w)=w·x(s,a)."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from rl_from_scratch.core.types import Action, State

FeatureMap = Callable[[State], Sequence[float]]
ActionFeatureMap = Callable[[State, Action], Sequence[float]]


class GridFeatures:
    """Bias and normalized row/column distances from the goal."""

    def __init__(self, rows: int, cols: int, goal: tuple[int, int]) -> None:
        self.rows, self.cols, self.goal = rows, cols, goal

    def __call__(self, state: State) -> np.ndarray:
        row, col = state
        return np.asarray([1.0, (self.goal[0] - row) / max(1, self.rows - 1), (self.goal[1] - col) / max(1, self.cols - 1)])


class LinearValue:
    def __init__(self, features: FeatureMap, dimension: int) -> None:
        if dimension < 1:
            raise ValueError("dimension must be positive")
        self.features = features
        self.weights = np.zeros(dimension, dtype=float)

    def x(self, state: State) -> np.ndarray:
        x = np.asarray(self.features(state), dtype=float)
        if x.shape != self.weights.shape:
            raise ValueError("feature dimension does not match weights")
        return x

    def __call__(self, state: State) -> float:
        return float(self.weights @ self.x(state))

    def update(self, state: State, target: float, alpha: float) -> None:
        self.weights += alpha * (target - self(state)) * self.x(state)


class LinearActionValue:
    def __init__(self, features: ActionFeatureMap, dimension: int) -> None:
        if dimension < 1:
            raise ValueError("dimension must be positive")
        self.features = features
        self.weights = np.zeros(dimension, dtype=float)

    def x(self, state: State, action: Action) -> np.ndarray:
        x = np.asarray(self.features(state, action), dtype=float)
        if x.shape != self.weights.shape:
            raise ValueError("feature dimension does not match weights")
        return x

    def __call__(self, state: State, action: Action) -> float:
        return float(self.weights @ self.x(state, action))

    def update(self, state: State, action: Action, target: float, alpha: float) -> None:
        self.weights += alpha * (target - self(state, action)) * self.x(state, action)
