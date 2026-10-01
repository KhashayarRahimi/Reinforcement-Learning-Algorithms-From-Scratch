from __future__ import annotations

import random
from collections.abc import Mapping, Sequence

from rl_from_scratch.core.types import Action, Environment, State


def choose(probabilities: Mapping[Action, float], rng: random.Random) -> Action:
    return rng.choices(list(probabilities), weights=list(probabilities.values()))[0]


def greedy_policy(env: Environment, q: Mapping[tuple[State, Action], float]) -> dict[State, Action]:
    return {s: max(env.actions(s), key=lambda a: q.get((s, a), 0.0)) for s in env.states if env.actions(s)}


def validate_learning(alpha: float, gamma: float, episodes: int, max_steps: int) -> None:
    if not 0 < alpha <= 1 or not 0 <= gamma <= 1 or episodes < 1 or max_steps < 1:
        raise ValueError("require alpha in (0,1], gamma in [0,1], and positive episodes/max_steps")
