"""Episode sampling, kept separate from update equations."""

from __future__ import annotations

import random

from .types import Environment, Episode, Policy, Transition


def generate_episode(
    env: Environment, policy: Policy, *, rng: random.Random | None = None,
    max_steps: int = 1_000, seed: int | None = None,
) -> Episode:
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    rng = rng or random.Random(seed)
    state = env.reset(seed)
    episode = Episode()
    for _ in range(max_steps):
        actions = env.actions(state)
        if not actions:
            break
        probabilities = policy.probabilities(state, actions)
        action = rng.choices(list(probabilities), weights=list(probabilities.values()))[0]
        next_state, reward, done = env.step(action)
        episode.transitions.append(Transition(state, action, reward, next_state, done))
        state = next_state
        if done:
            break
    return episode
