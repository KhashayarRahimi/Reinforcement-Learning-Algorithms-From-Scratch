"""Monte Carlo prediction and epsilon-soft / importance-sampling control."""

from __future__ import annotations

import random

from rl_from_scratch.core import Environment, Result, generate_episode
from rl_from_scratch.core.types import Policy
from rl_from_scratch.policies import EpsilonGreedyPolicy
from rl_from_scratch.utils import greedy_policy


def _returns(episode, gamma: float):
    G = 0.0
    for step in reversed(episode.transitions):
        G = step.reward + gamma * G
        yield step, G


class FirstVisitPrediction:
    """Average the return following the first visit to each state."""

    def __init__(self, env: Environment, policy: Policy, *, gamma: float = 0.99, seed: int | None = None) -> None:
        self.env, self.policy, self.gamma, self.seed = env, policy, gamma, seed

    def train(self, episodes: int, *, max_steps: int = 1000) -> Result:
        if episodes < 1 or max_steps < 1 or not 0 <= self.gamma <= 1:
            raise ValueError("invalid episodes, max_steps or gamma")
        rng = random.Random(self.seed)
        counts: dict = {}
        values: dict = {}
        result = Result()
        for _ in range(episodes):
            episode = generate_episode(self.env, self.policy, rng=rng, max_steps=max_steps)
            if not episode.terminated:
                result.episode_returns.append(episode.return_)
                result.episode_lengths.append(episode.length)
                continue
            first = {}
            for i, step in enumerate(episode.transitions):
                first.setdefault(step.state, i)
            for i, (step, G) in enumerate(reversed(list(_returns(episode, self.gamma)))):
                if first[step.state] == i:
                    counts[step.state] = counts.get(step.state, 0) + 1
                    values[step.state] = values.get(step.state, 0.0) + (G - values.get(step.state, 0.0)) / counts[step.state]
            result.episode_returns.append(episode.return_)
            result.episode_lengths.append(episode.length)
        result.values = values
        return result


class OnPolicyControl:
    """First-visit MC control with epsilon-greedy policy improvement."""

    def __init__(self, env: Environment, *, gamma: float = 0.99, epsilon: float = 0.1, seed: int | None = None) -> None:
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be in [0,1]")
        self.env, self.gamma, self.epsilon, self.seed = env, gamma, epsilon, seed

    def train(self, episodes: int, *, max_steps: int = 1000) -> Result:
        if episodes < 1 or max_steps < 1 or not 0 <= self.gamma <= 1:
            raise ValueError("invalid episodes, max_steps or gamma")
        rng = random.Random(self.seed)
        q: dict = {}
        counts: dict = {}
        result = Result()
        policy = EpsilonGreedyPolicy(q, self.epsilon)
        for _ in range(episodes):
            episode = generate_episode(self.env, policy, rng=rng, max_steps=max_steps)
            if not episode.terminated:
                result.episode_returns.append(episode.return_)
                result.episode_lengths.append(episode.length)
                continue
            steps = list(episode.transitions)
            first = {}
            for i, step in enumerate(steps):
                first.setdefault((step.state, step.action), i)
            for i, (step, G) in enumerate(reversed(list(_returns(episode, self.gamma)))):
                key = step.state, step.action
                if first[key] == i:
                    counts[key] = counts.get(key, 0) + 1
                    q[key] = q.get(key, 0.0) + (G - q.get(key, 0.0)) / counts[key]
            result.episode_returns.append(episode.return_)
            result.episode_lengths.append(episode.length)
        result.action_values = q
        result.policy = greedy_policy(self.env, q)
        return result


class OffPolicyPrediction:
    """Weighted importance sampling: C+=W, Q+=W/C*(G-Q)."""

    def __init__(self, env: Environment, target: Policy, behavior: Policy, *, gamma: float = 0.99, seed: int | None = None) -> None:
        self.env, self.target, self.behavior, self.gamma, self.seed = env, target, behavior, gamma, seed

    def train(self, episodes: int, *, max_steps: int = 1000) -> Result:
        if episodes < 1 or max_steps < 1 or not 0 <= self.gamma <= 1:
            raise ValueError("invalid episodes, max_steps or gamma")
        rng = random.Random(self.seed)
        q: dict = {}
        cumulative: dict = {}
        result = Result()
        for _ in range(episodes):
            episode = generate_episode(self.env, self.behavior, rng=rng, max_steps=max_steps)
            if not episode.terminated:
                result.episode_returns.append(episode.return_)
                result.episode_lengths.append(episode.length)
                continue
            weight = 1.0
            for step, G in _returns(episode, self.gamma):
                key = step.state, step.action
                cumulative[key] = cumulative.get(key, 0.0) + weight
                q[key] = q.get(key, 0.0) + weight / cumulative[key] * (G - q.get(key, 0.0))
                target_p = self.target.probabilities(step.state, self.env.actions(step.state))[step.action]
                behavior_p = self.behavior.probabilities(step.state, self.env.actions(step.state))[step.action]
                if behavior_p <= 0:
                    raise ValueError("behavior policy lacks support")
                weight *= target_p / behavior_p
                if weight == 0:
                    break
            result.episode_returns.append(episode.return_)
            result.episode_lengths.append(episode.length)
        result.action_values = q
        return result


class OffPolicyControl:
    """Weighted-importance-sampling MC control with greedy target policy."""

    def __init__(self, env: Environment, behavior: Policy, *, gamma: float = 0.99, seed: int | None = None) -> None:
        self.env, self.behavior, self.gamma, self.seed = env, behavior, gamma, seed

    def train(self, episodes: int, *, max_steps: int = 1000) -> Result:
        if episodes < 1 or max_steps < 1 or not 0 <= self.gamma <= 1:
            raise ValueError("invalid episodes, max_steps or gamma")
        rng = random.Random(self.seed)
        q: dict = {}
        cumulative: dict = {}
        result = Result()
        for _ in range(episodes):
            episode = generate_episode(self.env, self.behavior, rng=rng, max_steps=max_steps)
            if not episode.terminated:
                result.episode_returns.append(episode.return_)
                result.episode_lengths.append(episode.length)
                continue
            weight = 1.0
            for step, G in _returns(episode, self.gamma):
                key = step.state, step.action
                cumulative[key] = cumulative.get(key, 0.0) + weight
                q[key] = q.get(key, 0.0) + weight / cumulative[key] * (G - q.get(key, 0.0))
                best = max(self.env.actions(step.state), key=lambda a: q.get((step.state, a), 0.0))
                if step.action != best:
                    break
                p = self.behavior.probabilities(step.state, self.env.actions(step.state))[step.action]
                if p <= 0:
                    raise ValueError("behavior policy lacks support")
                weight /= p
            result.episode_returns.append(episode.return_)
            result.episode_lengths.append(episode.length)
        result.action_values = q
        result.policy = greedy_policy(self.env, q)
        return result
