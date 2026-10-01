"""Run with: python examples/grid_q_learning.py"""

from rl_from_scratch import GridWorld, QLearning


def main() -> None:
    env = GridWorld.random(5, 4, holes=4, seed=39)
    result = QLearning(env, alpha=0.2, gamma=0.95, epsilon=0.1, seed=39).train(2_000, max_steps=200)
    print("Start:", env.start, "goal:", env.goal, "holes:", sorted(env.holes))
    print("Greedy action at start:", result.policy[env.start])
    print("Mean return over final 100 episodes:", sum(result.episode_returns[-100:]) / 100)


if __name__ == "__main__":
    main()
