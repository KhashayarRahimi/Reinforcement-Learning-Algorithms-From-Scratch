"""Linear softmax policy-gradient and actor-critic algorithms."""

from __future__ import annotations

import random

import numpy as np

from rl_from_scratch.core import Environment, Result, generate_episode
from rl_from_scratch.policies import LinearSoftmaxPolicy
from rl_from_scratch.utils import choose
from rl_from_scratch.value_functions import LinearValue


def _finish(env: Environment, actor: LinearSoftmaxPolicy, critic: LinearValue | None, result: Result) -> Result:
    result.policy = {s: max(env.actions(s), key=lambda a: actor.probabilities(s, env.actions(s))[a]) for s in env.states if env.actions(s)}
    if critic is not None:
        result.values = {s: critic(s) for s in env.states}
    result.parameters = {"actor": actor.weights.copy(), "critic": None if critic is None else critic.weights.copy()}
    return result


class _PolicyGradient:
    def __init__(self, env: Environment, actor: LinearSoftmaxPolicy, *, critic: LinearValue | None = None, actor_alpha: float = 0.01, critic_alpha: float = 0.1, gamma: float = 0.99, seed: int | None = None) -> None:
        if actor_alpha <= 0 or critic_alpha <= 0 or not 0 <= gamma <= 1:
            raise ValueError("invalid learning rates or gamma")
        self.env, self.actor, self.critic, self.actor_alpha, self.critic_alpha, self.gamma, self.seed = env, actor, critic, actor_alpha, critic_alpha, gamma, seed

    def _mc(self, episodes: int, max_steps: int, baseline: bool) -> Result:
        if episodes < 1 or max_steps < 1 or (baseline and self.critic is None):
            raise ValueError("positive episodes/max_steps and a critic for baseline are required")
        rng = random.Random(self.seed)
        result = Result()
        for _ in range(episodes):
            episode = generate_episode(self.env, self.actor, rng=rng, max_steps=max_steps)
            if not episode.terminated:
                result.episode_returns.append(episode.return_)
                result.episode_lengths.append(episode.length)
                continue
            steps = episode.transitions
            G = 0.0
            returns = [0.0] * len(steps)
            for t in range(len(steps) - 1, -1, -1):
                G = steps[t].reward + self.gamma * G
                returns[t] = G
            for t, (step, G) in enumerate(zip(steps, returns)):
                advantage = G
                if baseline:
                    value = self.critic(step.state)
                    advantage -= value
                    self.critic.weights += self.critic_alpha * advantage * self.critic.x(step.state)
                self.actor.weights += self.actor_alpha * self.gamma ** t * advantage * self.actor.grad_log(step.state, step.action, self.env.actions(step.state))
            result.episode_returns.append(episode.return_)
            result.episode_lengths.append(episode.length)
        return _finish(self.env, self.actor, self.critic, result)


class REINFORCE(_PolicyGradient):
    """θ ← θ+α γ^t G_t ∇log π(A_t|S_t,θ)."""

    def train(self, episodes: int, *, max_steps: int = 1000) -> Result:
        return self._mc(episodes, max_steps, False)


class REINFORCEWithBaseline(_PolicyGradient):
    def train(self, episodes: int, *, max_steps: int = 1000) -> Result:
        return self._mc(episodes, max_steps, True)


class _ActorCritic(_PolicyGradient):
    def __init__(self, env: Environment, actor: LinearSoftmaxPolicy, critic: LinearValue, *, actor_lam: float = 0.0, critic_lam: float = 0.0, average_alpha: float = 0.01, **kwargs: object) -> None:
        if not 0 <= actor_lam <= 1 or not 0 <= critic_lam <= 1 or average_alpha <= 0:
            raise ValueError("invalid trace decay or average-reward step size")
        super().__init__(env, actor, critic=critic, **kwargs)
        self.actor_lam, self.critic_lam, self.average_alpha = actor_lam, critic_lam, average_alpha

    def _train(self, episodes: int, max_steps: int, continuing: bool) -> Result:
        if episodes < 1 or max_steps < 1:
            raise ValueError("episodes and max_steps must be positive")
        rng = random.Random(self.seed)
        result = Result()
        average_reward = 0.0
        continuing_state = None
        for _ in range(episodes):
            state = continuing_state if continuing and continuing_state is not None else self.env.reset()
            actor_trace = np.zeros_like(self.actor.weights)
            critic_trace = np.zeros_like(self.critic.weights)
            discount = 1.0
            total = 0.0
            length = 0
            for length in range(1, max_steps + 1):
                actions = self.env.actions(state)
                if not actions:
                    length = 0
                    break
                action = choose(self.actor.probabilities(state, actions), rng)
                grad = self.actor.grad_log(state, action, actions)
                next_state, reward, done = self.env.step(action)
                if continuing and done:
                    raise ValueError("continuing actor-critic requires a continuing environment")
                total += reward
                value = self.critic(state)
                next_value = 0.0 if done else self.critic(next_state)
                if continuing:
                    delta = reward - average_reward + next_value - value
                    average_reward += self.average_alpha * delta
                    actor_trace = self.actor_lam * actor_trace + grad
                    critic_trace = self.critic_lam * critic_trace + self.critic.x(state)
                else:
                    delta = reward + self.gamma * next_value - value
                    actor_trace = self.gamma * self.actor_lam * actor_trace + discount * grad
                    critic_trace = self.gamma * self.critic_lam * critic_trace + self.critic.x(state)
                self.critic.weights += self.critic_alpha * delta * critic_trace
                self.actor.weights += self.actor_alpha * delta * actor_trace
                if not continuing:
                    discount *= self.gamma
                state = next_state
                if done:
                    break
            if continuing:
                continuing_state = state
            result.episode_returns.append(total)
            result.episode_lengths.append(length)
        result = _finish(self.env, self.actor, self.critic, result)
        if continuing:
            result.values["average_reward"] = average_reward
        return result


class OneStepActorCritic(_ActorCritic):
    def train(self, episodes: int, *, max_steps: int = 1000) -> Result:
        return self._train(episodes, max_steps, False)


class EpisodicTraceActorCritic(_ActorCritic):
    def train(self, episodes: int, *, max_steps: int = 1000) -> Result:
        return self._train(episodes, max_steps, False)


class ContinuingTraceActorCritic(_ActorCritic):
    """Differential actor-critic; use an environment without terminal states."""

    def train(self, episodes: int, *, max_steps: int = 1000) -> Result:
        return self._train(episodes, max_steps, True)
