"""Linear prediction and control methods from Sutton and Barto."""

from __future__ import annotations

import random

import numpy as np

from rl_from_scratch.core import Environment, Result, generate_episode
from rl_from_scratch.core.types import Policy
from rl_from_scratch.utils import validate_learning
from rl_from_scratch.value_functions import LinearValue, LinearActionValue


def _result(env: Environment, value: LinearValue | LinearActionValue, histories: Result) -> Result:
    if isinstance(value, LinearValue):
        histories.values = {s: value(s) for s in env.states}
    else:
        histories.action_values = {(s, a): value(s, a) for s in env.states for a in env.actions(s)}
        histories.policy = {s: max(env.actions(s), key=lambda a: value(s, a)) for s in env.states if env.actions(s)}
    histories.parameters = value.weights.copy()
    return histories


class _Prediction:
    def __init__(self, env: Environment, policy: Policy, value: LinearValue, *, alpha: float = 0.1, gamma: float = 0.99, n: int = 1, seed: int | None = None) -> None:
        if n < 1:
            raise ValueError("n must be positive")
        self.env, self.policy, self.value, self.alpha, self.gamma, self.n, self.seed = env, policy, value, alpha, gamma, n, seed

    def _train(self, episodes: int, max_steps: int, kind: str) -> Result:
        validate_learning(self.alpha, self.gamma, episodes, max_steps)
        rng = random.Random(self.seed)
        result = Result()
        for _ in range(episodes):
            episode = generate_episode(self.env, self.policy, rng=rng, max_steps=max_steps)
            steps = episode.transitions
            if kind == "mc" and not episode.terminated:
                result.episode_returns.append(episode.return_)
                result.episode_lengths.append(episode.length)
                continue
            for t, step in enumerate(steps):
                h = len(steps) if kind == "mc" else min(t + self.n, len(steps))
                G = sum(self.gamma ** (i - t) * steps[i].reward for i in range(t, h))
                if h < len(steps) or (h == len(steps) and steps and not steps[-1].terminated):
                    G += self.gamma ** (h - t) * self.value(steps[h - 1].next_state)
                self.value.update(step.state, G, self.alpha)
            result.episode_returns.append(episode.return_)
            result.episode_lengths.append(episode.length)
        return _result(self.env, self.value, result)


class GradientMonteCarlo(_Prediction):
    """w ← w+α[G_t−v̂(S_t,w)]∇v̂(S_t,w)."""

    def train(self, episodes: int, *, max_steps: int = 1000) -> Result:
        return self._train(episodes, max_steps, "mc")


class SemiGradientTD(_Prediction):
    """Semi-gradient one-step TD prediction."""

    def train(self, episodes: int, *, max_steps: int = 1000) -> Result:
        self.n = 1
        return self._train(episodes, max_steps, "td")


class NStepSemiGradientTD(_Prediction):
    def train(self, episodes: int, *, max_steps: int = 1000) -> Result:
        return self._train(episodes, max_steps, "td")


class LSTD:
    """Solve (Σ x_t(x_t−γx_{t+1})ᵀ + λI)w=Σ x_t R_{t+1}."""

    def __init__(self, env: Environment, policy: Policy, value: LinearValue, *, gamma: float = 0.99, regularization: float = 1e-6, seed: int | None = None) -> None:
        if not 0 <= gamma <= 1 or regularization <= 0:
            raise ValueError("invalid gamma or regularization")
        self.env, self.policy, self.value, self.gamma, self.regularization, self.seed = env, policy, value, gamma, regularization, seed

    def train(self, episodes: int, *, max_steps: int = 1000) -> Result:
        if episodes < 1 or max_steps < 1:
            raise ValueError("episodes and max_steps must be positive")
        rng = random.Random(self.seed)
        d = len(self.value.weights)
        A = self.regularization * np.eye(d)
        b = np.zeros(d)
        result = Result()
        for _ in range(episodes):
            episode = generate_episode(self.env, self.policy, rng=rng, max_steps=max_steps)
            for step in episode.transitions:
                x = self.value.x(step.state)
                next_x = np.zeros(d) if step.terminated else self.value.x(step.next_state)
                A += np.outer(x, x - self.gamma * next_x)
                b += x * step.reward
            result.episode_returns.append(episode.return_)
            result.episode_lengths.append(episode.length)
        self.value.weights = np.linalg.solve(A, b)
        return _result(self.env, self.value, result)


class _Control:
    def __init__(self, env: Environment, value: LinearActionValue, *, n: int = 1, alpha: float = 0.1, gamma: float = 0.99, epsilon: float = 0.1, beta: float = 0.1, seed: int | None = None) -> None:
        if n < 1 or not 0 <= epsilon <= 1 or not 0 < beta <= 1:
            raise ValueError("invalid n, epsilon or beta")
        self.env, self.value, self.n, self.alpha, self.gamma, self.epsilon, self.beta, self.seed = env, value, n, alpha, gamma, epsilon, beta, seed

    def _choose(self, state, rng):
        actions = self.env.actions(state)
        if not actions:
            return None
        if rng.random() < self.epsilon:
            return rng.choice(list(actions))
        return max(actions, key=lambda a: self.value(state, a))

    def _train(self, episodes: int, max_steps: int, differential: bool) -> Result:
        validate_learning(self.alpha, self.gamma, episodes, max_steps)
        rng = random.Random(self.seed)
        average_reward = 0.0
        result = Result()
        continuing_state = None
        continuing_action = None
        for _ in range(episodes):
            state = continuing_state if differential and continuing_state is not None else self.env.reset()
            action = continuing_action if differential and continuing_action is not None else self._choose(state, rng)
            steps = []
            for _ in range(max_steps):
                if action is None:
                    break
                next_state, reward, done = self.env.step(action)
                if differential and done:
                    raise ValueError("differential Sarsa requires a continuing environment")
                next_action = None if done else self._choose(next_state, rng)
                steps.append((state, action, reward, next_state, next_action, done))
                if done:
                    break
                state, action = next_state, next_action
            if differential and steps:
                continuing_state, continuing_action = steps[-1][3], steps[-1][4]
            for t, (state, action, _, _, _, _) in enumerate(steps):
                h = min(t + self.n, len(steps))
                if differential:
                    G = sum(steps[i][2] - average_reward for i in range(t, h))
                    if steps[h - 1][4] is not None:
                        G += self.value(steps[h - 1][3], steps[h - 1][4])
                    delta = G - self.value(state, action)
                    average_reward += self.beta * delta
                else:
                    G = sum(self.gamma ** (i - t) * steps[i][2] for i in range(t, h))
                    if steps[h - 1][4] is not None:
                        G += self.gamma ** (h - t) * self.value(steps[h - 1][3], steps[h - 1][4])
                self.value.update(state, action, G, self.alpha)
            result.episode_returns.append(sum(step[2] for step in steps))
            result.episode_lengths.append(len(steps))
        result = _result(self.env, self.value, result)
        if differential:
            result.values["average_reward"] = average_reward
        return result


class SemiGradientSarsa(_Control):
    def train(self, episodes: int, *, max_steps: int = 1000) -> Result:
        self.n = 1
        return self._train(episodes, max_steps, False)


class NStepSemiGradientSarsa(_Control):
    def train(self, episodes: int, *, max_steps: int = 1000) -> Result:
        return self._train(episodes, max_steps, False)


class DifferentialSarsa(_Control):
    """Average-reward Sarsa; use a continuing environment."""

    def train(self, episodes: int, *, max_steps: int = 1000) -> Result:
        self.n = 1
        return self._train(episodes, max_steps, True)


class DifferentialNStepSarsa(_Control):
    def train(self, episodes: int, *, max_steps: int = 1000) -> Result:
        return self._train(episodes, max_steps, True)
