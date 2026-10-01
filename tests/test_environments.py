import pytest

from rl_from_scratch.environments import FiniteMDP, GridWorld
from rl_from_scratch.policies import GoalDirectedPolicy
from rl_from_scratch.utils import follow_policy


def test_finite_mdp_validates_probabilities_and_samples_terminal_transition():
    with pytest.raises(ValueError):
        FiniteMDP({("s", "a"): [(0.8, "t", 1, True)]}, "s")
    env = FiniteMDP({("s", "a"): [(1, "t", 2, True)]}, "s")
    assert env.reset() == "s"
    assert env.step("a") == ("t", 2, True)
    assert env.actions("t") == ()


def test_grid_walls_goal_holes_and_slip():
    env = GridWorld(2, 2, start=(0, 0), goal=(1, 1), holes=((0, 1),), slip=0)
    assert env.transitions((0, 0), "up") == ((1.0, (0, 0), -1.0, False),)
    assert env.transitions((0, 0), "right") == ((1.0, (0, 0), -3.0, False),)
    assert env.transitions((1, 0), "right") == ((1.0, (1, 1), 10.0, True),)
    assert sum(p for p, *_ in GridWorld(2, 2, slip=0.3).transitions((0, 0), "right")) == pytest.approx(1)


def test_random_grid_is_valid_and_seeded():
    first = GridWorld.random(4, 4, 4, seed=12)
    second = GridWorld.random(4, 4, 4, seed=12)
    assert first.holes == second.holes
    assert first.start not in first.holes and first.goal not in first.holes
    random_endpoints = GridWorld.random(4, 4, 2, seed=12, random_start_goal=True)
    assert (random_endpoints.start, random_endpoints.goal, random_endpoints.holes) == (
        GridWorld.random(4, 4, 2, seed=12, random_start_goal=True).start,
        GridWorld.random(4, 4, 2, seed=12, random_start_goal=True).goal,
        GridWorld.random(4, 4, 2, seed=12, random_start_goal=True).holes,
    )
    with pytest.raises(ValueError):
        GridWorld.random(2, 2, 3)


def test_terminal_hole_and_inspectable_path():
    terminal_hole = GridWorld(2, 2, holes=((0, 1),), hole_resets=False)
    assert terminal_hole.actions((0, 1)) == ()
    assert terminal_hole.transitions((0, 0), "right")[0][-1] is True
    env = GridWorld(2, 2)
    path = follow_policy(env, GoalDirectedPolicy(env), max_steps=10)
    assert path[0] == env.start and path[-1] == env.goal
