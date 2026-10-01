"""One-step temporal-difference prediction and control."""

from __future__ import annotations

import random

from rl_from_scratch.core import Environment, Result, generate_episode
from rl_from_scratch.core.types import Policy
from rl_from_scratch.policies import EpsilonGreedyPolicy
from rl_from_scratch.utils import choose, greedy_policy, validate_learning
from rl_from_scratch.value_functions import TabularActionValues, TabularValues


class TDZero:
    """V(S) ← V(S)+α[R+γV(S')−V(S)]."""

    def __init__(self, env: Environment, policy: Policy, *, alpha: float = 0.1, gamma: float = 0.99, seed: int | None = None) -> None:
        self.env, self.policy, self.alpha, self.gamma, self.seed = env, policy, alpha, gamma, seed

    def train(self, episodes: int, *, max_steps: int = 1000) -> Result:
        validate_learning(self.alpha, self.gamma, episodes, max_steps)
        rng = random.Random(self.seed)
        v = TabularValues()
        result = Result()
        for _ in range(episodes):
            episode = generate_episode(self.env, self.policy, rng=rng, max_steps=max_steps)
            for step in episode.transitions:
                v.update(step.state, step.reward + (0 if step.terminated else self.gamma * v[step.next_state]), self.alpha)
            result.episode_returns.append(episode.return_)
            result.episode_lengths.append(episode.length)
        result.values = v.values
        return result


class _TDControl:
    def __init__(self, env: Environment, *, alpha: float = 0.1, gamma: float = 0.99, epsilon: float = 0.1, seed: int | None = None) -> None:
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be in [0,1]")
        self.env, self.alpha, self.gamma, self.epsilon, self.seed = env, alpha, gamma, epsilon, seed

    def _run(self, episodes: int, max_steps: int, kind: str) -> Result:
        validate_learning(self.alpha, self.gamma, episodes, max_steps)
        rng = random.Random(self.seed)
        q = TabularActionValues()
        q2 = TabularActionValues()
        result = Result()
        for _ in range(episodes):
            state = self.env.reset()
            total = 0.0
            length = 0
            combined = {key: q[key] + q2[key] for s in self.env.states for a in self.env.actions(s) for key in [(s, a)]} if kind == "double" else q.values
            policy = EpsilonGreedyPolicy(combined, self.epsilon)
            if not self.env.actions(state):
                result.episode_returns.append(0.0)
                result.episode_lengths.append(0)
                continue
            action = choose(policy.probabilities(state, self.env.actions(state)), rng)
            for length in range(1, max_steps + 1):
                next_state, reward, done = self.env.step(action)
                total += reward
                next_actions = self.env.actions(next_state) if not done else ()
                if kind == "sarsa":
                    next_action = choose(policy.probabilities(next_state, next_actions), rng) if next_actions else None
                    target = reward + (self.gamma * q[next_state, next_action] if next_action is not None else 0)
                    q.update((state, action), target, self.alpha)
                elif kind == "q":
                    target = reward + (self.gamma * max(q[next_state, a] for a in next_actions) if next_actions else 0)
                    q.update((state, action), target, self.alpha)
                else:
                    first, second = (q, q2) if rng.random() < 0.5 else (q2, q)
                    if next_actions:
                        best = max(next_actions, key=lambda a: first[next_state, a])
                        target = reward + self.gamma * second[next_state, best]
                    else:
                        target = reward
                    first.update((state, action), target, self.alpha)
                    policy.action_values = {key: q[key] + q2[key] for s in self.env.states for a in self.env.actions(s) for key in [(s, a)]}
                if done or not next_actions:
                    break
                state = next_state
                action = next_action if kind == "sarsa" else choose(policy.probabilities(state, next_actions), rng)
            result.episode_returns.append(total)
            result.episode_lengths.append(length)
        result.action_values = {key: q[key] + q2[key] for s in self.env.states for a in self.env.actions(s) for key in [(s, a)]} if kind == "double" else q.values
        result.policy = greedy_policy(self.env, result.action_values)
        result.parameters = (q.values, q2.values) if kind == "double" else None
        return result


class Sarsa(_TDControl):
    """On-policy: Q ← Q+α[R+γQ(S',A')−Q]."""

    def train(self, episodes: int, *, max_steps: int = 1000) -> Result:
        return self._run(episodes, max_steps, "sarsa")


class QLearning(_TDControl):
    """Off-policy: Q ← Q+α[R+γ max_a Q(S',a)−Q]."""

    def train(self, episodes: int, *, max_steps: int = 1000) -> Result:
        return self._run(episodes, max_steps, "q")


class DoubleQLearning(_TDControl):
    """Use one table for argmax and the other for target evaluation."""

    def train(self, episodes: int, *, max_steps: int = 1000) -> Result:
        return self._run(episodes, max_steps, "double")
