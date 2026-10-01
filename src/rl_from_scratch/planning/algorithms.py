"""Tabular sample planning, Dyna-Q and prioritized sweeping."""

from __future__ import annotations

import heapq
import random

from rl_from_scratch.core import Environment, Result
from rl_from_scratch.core.types import Action, State, ModelEnvironment
from rl_from_scratch.policies import EpsilonGreedyPolicy
from rl_from_scratch.utils import choose, greedy_policy, validate_learning
from rl_from_scratch.value_functions import TabularActionValues


class DeterministicModel:
    """The most recently observed transition for each (state, action)."""

    def __init__(self) -> None:
        self.experience: dict[tuple[State, Action], tuple[State, float, bool]] = {}
        self.predecessors: dict[State, set[tuple[State, Action]]] = {}

    @classmethod
    def from_environment(cls, env: ModelEnvironment) -> "DeterministicModel":
        """Populate a planning model from a known deterministic transition table."""
        model = cls()
        for state in env.states:
            for action in env.actions(state):
                outcomes = env.transitions(state, action)
                if len(outcomes) != 1 or abs(outcomes[0][0] - 1.0) > 1e-9:
                    raise ValueError("from_environment requires deterministic transitions")
                _, next_state, reward, done = outcomes[0]
                model.observe(state, action, next_state, reward, done)
        return model

    def observe(self, state: State, action: Action, next_state: State, reward: float, done: bool) -> None:
        key = state, action
        old = self.experience.get(key)
        if old:
            self.predecessors.get(old[0], set()).discard(key)
        self.experience[key] = next_state, reward, done
        self.predecessors.setdefault(next_state, set()).add(key)

    def sample(self, rng: random.Random) -> tuple[State, Action, State, float, bool]:
        if not self.experience:
            raise ValueError("model has no observations")
        key = rng.choice(list(self.experience))
        next_state, reward, done = self.experience[key]
        return key[0], key[1], next_state, reward, done


def _target(env: Environment, q: TabularActionValues, next_state: State, reward: float, done: bool, gamma: float) -> float:
    actions = env.actions(next_state) if not done else ()
    return reward + (gamma * max(q[next_state, a] for a in actions) if actions else 0.0)


class RandomSamplePlanning:
    """Q-learning updates drawn only from an already supplied model."""

    def __init__(self, env: Environment, model: DeterministicModel, *, alpha: float = 0.1, gamma: float = 0.99, seed: int | None = None) -> None:
        self.env, self.model, self.alpha, self.gamma, self.seed = env, model, alpha, gamma, seed

    def train(self, updates: int) -> Result:
        validate_learning(self.alpha, self.gamma, updates, 1)
        q = TabularActionValues()
        rng = random.Random(self.seed)
        for _ in range(updates):
            state, action, next_state, reward, done = self.model.sample(rng)
            q.update((state, action), _target(self.env, q, next_state, reward, done, self.gamma), self.alpha)
        return Result(action_values=q.values, policy=greedy_policy(self.env, q.values))


class DynaQ:
    """One real Q-learning update followed by n_planning model updates."""

    def __init__(self, env: Environment, *, n_planning: int = 5, alpha: float = 0.1, gamma: float = 0.99, epsilon: float = 0.1, seed: int | None = None, model: DeterministicModel | None = None) -> None:
        if n_planning < 0 or not 0 <= epsilon <= 1:
            raise ValueError("n_planning must be nonnegative and epsilon in [0,1]")
        self.env, self.n_planning, self.alpha, self.gamma, self.epsilon, self.seed = env, n_planning, alpha, gamma, epsilon, seed
        self.model = model or DeterministicModel()

    def train(self, episodes: int, *, max_steps: int = 1000) -> Result:
        validate_learning(self.alpha, self.gamma, episodes, max_steps)
        rng = random.Random(self.seed)
        q = TabularActionValues()
        policy = EpsilonGreedyPolicy(q.values, self.epsilon)
        result = Result()
        for _ in range(episodes):
            state = self.env.reset()
            total = 0.0
            length = 0
            for length in range(1, max_steps + 1):
                actions = self.env.actions(state)
                if not actions:
                    length = 0
                    break
                action = choose(policy.probabilities(state, actions), rng)
                next_state, reward, done = self.env.step(action)
                total += reward
                q.update((state, action), _target(self.env, q, next_state, reward, done, self.gamma), self.alpha)
                self.model.observe(state, action, next_state, reward, done)
                for _ in range(self.n_planning):
                    s, a, ns, r, terminal = self.model.sample(rng)
                    q.update((s, a), _target(self.env, q, ns, r, terminal, self.gamma), self.alpha)
                state = next_state
                if done:
                    break
            result.episode_returns.append(total)
            result.episode_lengths.append(length)
        result.action_values = q.values
        result.policy = greedy_policy(self.env, q.values)
        return result


class PrioritizedSweeping(DynaQ):
    """Plan from the largest model TD error, then queue predecessors."""

    def __init__(self, env: Environment, *, theta: float = 1e-4, **kwargs: object) -> None:
        if theta < 0:
            raise ValueError("theta must be nonnegative")
        super().__init__(env, **kwargs)
        self.theta = theta

    def train(self, episodes: int, *, max_steps: int = 1000) -> Result:
        validate_learning(self.alpha, self.gamma, episodes, max_steps)
        rng = random.Random(self.seed)
        q = TabularActionValues()
        policy = EpsilonGreedyPolicy(q.values, self.epsilon)
        heap: list[tuple[float, int, tuple[State, Action]]] = []
        counter = 0
        result = Result()
        for _ in range(episodes):
            state = self.env.reset()
            total = 0.0
            length = 0
            for length in range(1, max_steps + 1):
                actions = self.env.actions(state)
                if not actions:
                    length = 0
                    break
                action = choose(policy.probabilities(state, actions), rng)
                next_state, reward, done = self.env.step(action)
                total += reward
                self.model.observe(state, action, next_state, reward, done)
                priority = abs(_target(self.env, q, next_state, reward, done, self.gamma) - q[state, action])
                if priority > self.theta:
                    heapq.heappush(heap, (-priority, counter, (state, action)))
                    counter += 1
                for _ in range(self.n_planning):
                    if not heap:
                        break
                    _, _, (s, a) = heapq.heappop(heap)
                    ns, r, terminal = self.model.experience[s, a]
                    q.update((s, a), _target(self.env, q, ns, r, terminal, self.gamma), self.alpha)
                    for predecessor in self.model.predecessors.get(s, ()):
                        ps, pa = predecessor
                        successor, pr, pd = self.model.experience[predecessor]
                        p = abs(_target(self.env, q, successor, pr, pd, self.gamma) - q[ps, pa])
                        if p > self.theta:
                            heapq.heappush(heap, (-p, counter, predecessor))
                            counter += 1
                state = next_state
                if done:
                    break
            result.episode_returns.append(total)
            result.episode_lengths.append(length)
        result.action_values = q.values
        result.policy = greedy_policy(self.env, q.values)
        return result
