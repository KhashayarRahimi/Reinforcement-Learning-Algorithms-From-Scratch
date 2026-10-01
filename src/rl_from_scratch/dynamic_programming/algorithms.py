"""Exact Bellman expectation and greedy improvement for a known finite model."""

from __future__ import annotations

from rl_from_scratch.core.types import ModelEnvironment, Policy, Result, State, Action


def bellman_action(env: ModelEnvironment, state: State, action: Action, values: dict[State, float], gamma: float) -> float:
    """q(s,a)=Σ_{s',r} p(s',r|s,a)[r+γv(s')], with zero terminal bootstrap."""
    return sum(p * (reward + (0.0 if done else gamma * values.get(next_state, 0.0)))
               for p, next_state, reward, done in env.transitions(state, action))


class PolicyEvaluation:
    def __init__(self, env: ModelEnvironment, policy: Policy, *, gamma: float = 0.99, theta: float = 1e-8, max_sweeps: int = 100_000) -> None:
        if not 0 <= gamma < 1 or theta <= 0 or max_sweeps < 1:
            raise ValueError("require gamma in [0,1), positive theta and max_sweeps")
        self.env, self.policy, self.gamma, self.theta, self.max_sweeps = env, policy, gamma, theta, max_sweeps

    def run(self) -> Result:
        values = {s: 0.0 for s in self.env.states}
        for _ in range(self.max_sweeps):
            delta = 0.0
            for state in self.env.states:
                actions = self.env.actions(state)
                if not actions:
                    continue
                probabilities = self.policy.probabilities(state, actions)
                new = sum(probabilities[a] * bellman_action(self.env, state, a, values, self.gamma) for a in actions)
                delta = max(delta, abs(new - values[state]))
                values[state] = new
            if delta < self.theta:
                return Result(values=values)
        raise RuntimeError("policy evaluation did not converge")


class PolicyIteration:
    def __init__(self, env: ModelEnvironment, *, gamma: float = 0.99, theta: float = 1e-8, max_iterations: int = 1000) -> None:
        if not 0 <= gamma < 1 or theta <= 0 or max_iterations < 1:
            raise ValueError("invalid gamma, theta or max_iterations")
        self.env, self.gamma, self.theta, self.max_iterations = env, gamma, theta, max_iterations

    def run(self) -> Result:
        from rl_from_scratch.policies import DeterministicPolicy
        chosen = {s: self.env.actions(s)[0] for s in self.env.states if self.env.actions(s)}
        for _ in range(self.max_iterations):
            values = PolicyEvaluation(self.env, DeterministicPolicy(chosen), gamma=self.gamma, theta=self.theta).run().values
            improved = {s: max(self.env.actions(s), key=lambda a: bellman_action(self.env, s, a, values, self.gamma)) for s in chosen}
            if improved == chosen:
                return Result(values=values, policy=chosen)
            chosen = improved
        raise RuntimeError("policy iteration did not converge")
