"""Differentiable softmax policy for policy-gradient methods."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from rl_from_scratch.core.types import Action, State


class LinearSoftmaxPolicy:
    def __init__(self, features: Callable[[State, Action], Sequence[float]], dimension: int) -> None:
        if dimension < 1:
            raise ValueError("dimension must be positive")
        self.features = features
        self.weights = np.zeros(dimension, dtype=float)

    def x(self, state: State, action: Action) -> np.ndarray:
        x = np.asarray(self.features(state, action), dtype=float)
        if x.shape != self.weights.shape:
            raise ValueError("feature dimension does not match weights")
        return x

    def probabilities(self, state: State, actions: Sequence[Action]) -> dict[Action, float]:
        if not actions:
            return {}
        logits = np.array([self.weights @ self.x(state, a) for a in actions])
        weights = np.exp(logits - max(logits))
        return dict(zip(actions, (weights / weights.sum()).tolist()))

    def grad_log(self, state: State, action: Action, actions: Sequence[Action]) -> np.ndarray:
        probabilities = self.probabilities(state, actions)
        return self.x(state, action) - sum((p * self.x(state, a) for a, p in probabilities.items()), np.zeros_like(self.weights))
