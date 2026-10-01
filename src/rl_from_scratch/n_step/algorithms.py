"""Episodic n-step tabular methods; updates follow each sampled trajectory."""

from __future__ import annotations

import random

from rl_from_scratch.core import Environment, Result, generate_episode
from rl_from_scratch.core.types import Policy
from rl_from_scratch.policies import EpsilonGreedyPolicy
from rl_from_scratch.utils import greedy_policy, validate_learning
from rl_from_scratch.value_functions import TabularValues, TabularActionValues


class NStepTD:
    """G_{t:t+n}=Σ_{i=0}^{n-1}γ^i R_{t+i+1}+γ^n V(S_{t+n})."""

    def __init__(self, env: Environment, policy: Policy, *, n: int = 2, alpha: float = 0.1, gamma: float = 0.99, seed: int | None = None) -> None:
        if n < 1:
            raise ValueError("n must be positive")
        self.env, self.policy, self.n, self.alpha, self.gamma, self.seed = env, policy, n, alpha, gamma, seed

    def train(self, episodes: int, *, max_steps: int = 1000) -> Result:
        validate_learning(self.alpha, self.gamma, episodes, max_steps)
        v = TabularValues()
        rng = random.Random(self.seed)
        result = Result()
        for _ in range(episodes):
            episode = generate_episode(self.env, self.policy, rng=rng, max_steps=max_steps)
            steps = episode.transitions
            for t, step in enumerate(steps):
                h = min(t + self.n, len(steps))
                G = sum(self.gamma ** (i - t) * steps[i].reward for i in range(t, h))
                if h < len(steps) or (h == len(steps) and steps and not steps[-1].terminated):
                    G += self.gamma ** (h - t) * v[steps[h - 1].next_state]
                v.update(step.state, G, self.alpha)
            result.episode_returns.append(episode.return_)
            result.episode_lengths.append(episode.length)
        result.values = v.values
        return result


class _NStepControl:
    def __init__(self, env: Environment, *, n: int = 2, alpha: float = 0.1, gamma: float = 0.99, epsilon: float = 0.1, seed: int | None = None, behavior: Policy | None = None, target: Policy | None = None, sigma: float = 1.0) -> None:
        if n < 1 or not 0 <= epsilon <= 1 or not 0 <= sigma <= 1:
            raise ValueError("invalid n, epsilon or sigma")
        self.env, self.n, self.alpha, self.gamma, self.epsilon, self.seed = env, n, alpha, gamma, epsilon, seed
        self.behavior, self.target, self.sigma = behavior, target, sigma

    def _train(self, episodes: int, max_steps: int, kind: str) -> Result:
        validate_learning(self.alpha, self.gamma, episodes, max_steps)
        rng = random.Random(self.seed)
        q = TabularActionValues()
        result = Result()
        for _ in range(episodes):
            target = self.target or EpsilonGreedyPolicy(q.values, self.epsilon)
            behavior = self.behavior or target
            episode = generate_episode(self.env, behavior, rng=rng, max_steps=max_steps)
            steps = episode.transitions
            for t, step in enumerate(steps):
                h = min(t + self.n, len(steps))
                if kind in ("sarsa", "off_policy"):
                    G = sum(self.gamma ** (i - t) * steps[i].reward for i in range(t, h))
                    if h < len(steps):
                        G += self.gamma ** (h - t) * q[steps[h].state, steps[h].action]
                    elif steps and not steps[-1].terminated:
                        final = steps[-1].next_state
                        if self.env.actions(final):
                            from rl_from_scratch.utils import choose
                            sampled = choose(behavior.probabilities(final, self.env.actions(final)), rng)
                            G += self.gamma ** (h - t) * q[final, sampled]
                    if kind == "off_policy":
                        rho = 1.0
                        for k in range(t + 1, h):
                            s, a = steps[k].state, steps[k].action
                            target_p = target.probabilities(s, self.env.actions(s))[a]
                            behavior_p = behavior.probabilities(s, self.env.actions(s))[a]
                            if behavior_p <= 0:
                                raise ValueError("behavior policy lacks support")
                            rho *= target_p / behavior_p
                        G = q[step.state, step.action] + rho * (G - q[step.state, step.action])
                else:
                    # Tree Backup (σ=0) and Q(σ) share the backward return recursion.
                    # G = R + γ[(1−σ)(V−π_a Q_a+π_a G_next)+σρ G_next].
                    if h < len(steps):
                        G = q[steps[h].state, steps[h].action]
                    else:
                        G = 0.0
                    for k in range(h - 1, t - 1, -1):
                        current = steps[k]
                        if k + 1 < len(steps):
                            following = steps[k + 1]
                            probs = target.probabilities(following.state, self.env.actions(following.state))
                            v_next = sum(p * q[following.state, a] for a, p in probs.items())
                            pi = probs[following.action]
                            b = behavior.probabilities(following.state, self.env.actions(following.state))[following.action]
                            if b <= 0:
                                raise ValueError("behavior policy lacks support")
                            rho = pi / b
                            expected_branch = v_next + pi * (G - q[following.state, following.action])
                            G = current.reward + self.gamma * ((1 - self.sigma) * expected_branch + self.sigma * rho * G)
                        else:
                            if current.terminated:
                                G = current.reward
                            else:
                                final = current.next_state
                                actions = self.env.actions(final)
                                if not actions:
                                    G = current.reward
                                else:
                                    probs = target.probabilities(final, actions)
                                    expected = sum(p * q[final, a] for a, p in probs.items())
                                    from rl_from_scratch.utils import choose
                                    sampled = choose(behavior.probabilities(final, actions), rng)
                                    b = behavior.probabilities(final, actions)[sampled]
                                    rho = probs[sampled] / b
                                    G = current.reward + self.gamma * ((1 - self.sigma) * expected + self.sigma * rho * q[final, sampled])
                q.update((step.state, step.action), G, self.alpha)
            result.episode_returns.append(episode.return_)
            result.episode_lengths.append(episode.length)
        result.action_values = q.values
        result.policy = greedy_policy(self.env, q.values)
        return result


class NStepSarsa(_NStepControl):
    def train(self, episodes: int, *, max_steps: int = 1000) -> Result:
        return self._train(episodes, max_steps, "sarsa")


class OffPolicyNStepSarsa(_NStepControl):
    def __init__(self, env: Environment, behavior: Policy, target: Policy, **kwargs: object) -> None:
        super().__init__(env, behavior=behavior, target=target, **kwargs)

    def train(self, episodes: int, *, max_steps: int = 1000) -> Result:
        return self._train(episodes, max_steps, "off_policy")


class TreeBackup(_NStepControl):
    def __init__(self, env: Environment, **kwargs: object) -> None:
        super().__init__(env, sigma=0.0, **kwargs)

    def train(self, episodes: int, *, max_steps: int = 1000) -> Result:
        return self._train(episodes, max_steps, "backup")


class QSigma(_NStepControl):
    def train(self, episodes: int, *, max_steps: int = 1000) -> Result:
        return self._train(episodes, max_steps, "sigma")
