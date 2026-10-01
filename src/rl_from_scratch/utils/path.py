"""Inspect one sampled path under a policy without training an algorithm."""

from __future__ import annotations

import random

from rl_from_scratch.core import Environment
from rl_from_scratch.core.types import Policy, State
from .sampling import choose


def follow_policy(env: Environment, policy: Policy, *, max_steps: int = 1000, seed: int | None = None) -> list[State]:
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    rng = random.Random(seed)
    state = env.reset(seed)
    path = [state]
    for _ in range(max_steps):
        actions = env.actions(state)
        if not actions:
            break
        action = choose(policy.probabilities(state, actions), rng)
        state, _, done = env.step(action)
        path.append(state)
        if done:
            break
    return path
