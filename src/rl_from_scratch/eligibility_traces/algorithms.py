"""Accumulating and Dutch eligibility traces for linear values."""

from __future__ import annotations

import random

import numpy as np

from rl_from_scratch.core import Environment, Result
from rl_from_scratch.core.types import Policy
from rl_from_scratch.function_approximation.algorithms import _result
from rl_from_scratch.utils import choose, validate_learning
from rl_from_scratch.value_functions import LinearValue, LinearActionValue


class _TracePrediction:
    def __init__(self, env: Environment, policy: Policy, value: LinearValue, *, alpha: float = 0.1, gamma: float = 0.99, lam: float = 0.9, seed: int | None = None) -> None:
        if not 0 <= lam <= 1:
            raise ValueError("lam must be in [0,1]")
        self.env, self.policy, self.value, self.alpha, self.gamma, self.lam, self.seed = env, policy, value, alpha, gamma, lam, seed

    def _train(self, episodes: int, max_steps: int, true_online: bool) -> Result:
        validate_learning(self.alpha, self.gamma, episodes, max_steps)
        rng = random.Random(self.seed)
        result = Result()
        for _ in range(episodes):
            state = self.env.reset()
            trace = np.zeros_like(self.value.weights)
            old_value = 0.0
            total = 0.0
            length = 0
            for length in range(1, max_steps + 1):
                actions = self.env.actions(state)
                if not actions:
                    length = 0
                    break
                action = choose(self.policy.probabilities(state, actions), rng)
                next_state, reward, done = self.env.step(action)
                total += reward
                x = self.value.x(state)
                v = self.value(state)
                v_next = 0.0 if done else self.value(next_state)
                delta = reward + self.gamma * v_next - v
                if true_online:
                    # Dutch trace and correction from Sutton & Barto, Eq. 12.11.
                    trace = self.gamma * self.lam * trace + (1 - self.alpha * self.gamma * self.lam * (trace @ x)) * x
                    self.value.weights += self.alpha * (delta + v - old_value) * trace - self.alpha * (v - old_value) * x
                    old_value = v_next
                else:
                    trace = self.gamma * self.lam * trace + x
                    self.value.weights += self.alpha * delta * trace
                state = next_state
                if done:
                    break
            result.episode_returns.append(total)
            result.episode_lengths.append(length)
        return _result(self.env, self.value, result)


class SemiGradientTDLambda(_TracePrediction):
    def train(self, episodes: int, *, max_steps: int = 1000) -> Result:
        return self._train(episodes, max_steps, False)


class TrueOnlineTDLambda(_TracePrediction):
    def train(self, episodes: int, *, max_steps: int = 1000) -> Result:
        return self._train(episodes, max_steps, True)


class _TraceControl:
    def __init__(self, env: Environment, value: LinearActionValue, *, alpha: float = 0.1, gamma: float = 0.99, lam: float = 0.9, epsilon: float = 0.1, seed: int | None = None) -> None:
        if not 0 <= lam <= 1 or not 0 <= epsilon <= 1:
            raise ValueError("lam and epsilon must be in [0,1]")
        self.env, self.value, self.alpha, self.gamma, self.lam, self.epsilon, self.seed = env, value, alpha, gamma, lam, epsilon, seed

    def _choose(self, state, rng):
        actions = self.env.actions(state)
        if not actions:
            return None
        return rng.choice(list(actions)) if rng.random() < self.epsilon else max(actions, key=lambda a: self.value(state, a))

    def _train(self, episodes: int, max_steps: int, true_online: bool) -> Result:
        validate_learning(self.alpha, self.gamma, episodes, max_steps)
        rng = random.Random(self.seed)
        result = Result()
        for _ in range(episodes):
            state = self.env.reset()
            action = self._choose(state, rng)
            trace = np.zeros_like(self.value.weights)
            old_q = 0.0
            total = 0.0
            length = 0
            for length in range(1, max_steps + 1):
                if action is None:
                    length = 0
                    break
                next_state, reward, done = self.env.step(action)
                next_action = None if done else self._choose(next_state, rng)
                total += reward
                x = self.value.x(state, action)
                q = self.value(state, action)
                q_next = 0.0 if next_action is None else self.value(next_state, next_action)
                delta = reward + self.gamma * q_next - q
                if true_online:
                    trace = self.gamma * self.lam * trace + (1 - self.alpha * self.gamma * self.lam * (trace @ x)) * x
                    self.value.weights += self.alpha * (delta + q - old_q) * trace - self.alpha * (q - old_q) * x
                    old_q = q_next
                else:
                    trace = self.gamma * self.lam * trace + x
                    self.value.weights += self.alpha * delta * trace
                state, action = next_state, next_action
                if done:
                    break
            result.episode_returns.append(total)
            result.episode_lengths.append(length)
        return _result(self.env, self.value, result)


class SarsaLambda(_TraceControl):
    def train(self, episodes: int, *, max_steps: int = 1000) -> Result:
        return self._train(episodes, max_steps, False)


class TrueOnlineSarsaLambda(_TraceControl):
    def train(self, episodes: int, *, max_steps: int = 1000) -> Result:
        return self._train(episodes, max_steps, True)
