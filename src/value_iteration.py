from __future__ import annotations

from dataclasses import dataclass

from .environment import Action, GridWorld


@dataclass(frozen=True)
class ValueIterationResult:
    V: dict[tuple[int, int], float]
    policy: dict[tuple[int, int], Action]
    deltas: list[float]


class ValueIterationSolver:
    def __init__(self, env: GridWorld, *, gamma: float = 0.99, theta: float = 1e-6) -> None:
        self.env = env
        self.gamma = float(gamma)
        self.theta = float(theta)

    def solve(self, *, max_iterations: int = 100_000) -> ValueIterationResult:
        states = list(self.env.states())
        V: dict[tuple[int, int], float] = {s: 0.0 for s in states}
        deltas: list[float] = []

        for _ in range(max_iterations):
            delta = 0.0
            V_new: dict[tuple[int, int], float] = {}

            for s in states:
                if self.env.is_terminal(s):
                    V_new[s] = 0.0
                    continue

                best = -float("inf")
                for a in self.env.actions():
                    q = 0.0
                    for t in self.env.get_transitions(s, a):
                        q += t.prob * (t.reward + self.gamma * V[t.next_state])
                    if q > best:
                        best = q
                V_new[s] = best
                delta = max(delta, abs(V_new[s] - V[s]))

            V = V_new
            deltas.append(delta)
            if delta < self.theta:
                break

        policy = self.extract_policy(V)
        return ValueIterationResult(V=V, policy=policy, deltas=deltas)

    def extract_policy(self, V: dict[tuple[int, int], float]) -> dict[tuple[int, int], Action]:
        policy: dict[tuple[int, int], Action] = {}
        for s in self.env.states():
            if self.env.is_terminal(s):
                continue

            best_a = Action.UP
            best_q = -float("inf")
            for a in self.env.actions():
                q = 0.0
                for t in self.env.get_transitions(s, a):
                    q += t.prob * (t.reward + self.gamma * V[t.next_state])
                if q > best_q:
                    best_q = q
                    best_a = a

            policy[s] = best_a
        return policy

    def evaluate_policy(
        self,
        policy: dict[tuple[int, int], Action],
        *,
        max_steps: int = 500,
        start_state: tuple[int, int] | None = None,
    ) -> tuple[list[tuple[int, int]], float]:
        s = self.env.reset() if start_state is None else start_state
        self.env.set_state(s)
        trajectory = [s]
        total_reward = 0.0
        for _ in range(max_steps):
            if self.env.is_terminal(s):
                break
            a = policy.get(s)
            if a is None:
                break
            s, r, done = self.env.step(a)
            trajectory.append(s)
            total_reward += r
            if done:
                break
        return trajectory, total_reward
