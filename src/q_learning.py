from __future__ import annotations

from dataclasses import dataclass
import abc
from typing import Any, Iterator

import numpy as np

from .environment import Action, GridWorld


@dataclass(frozen=True)
class QLearningMetrics:
    episode_rewards: list[float]
    episode_steps: list[int]
    epsilons: list[float]


@dataclass(frozen=True)
class StepEvent:
    """A single TD step emitted by `QLearningAgent.train_stream`.

    Fields let a UI draw the current state of training without needing to
    reimplement any of the Q-update logic.
    """

    episode: int
    step: int
    state: tuple[int, int]
    action: Action
    next_state: tuple[int, int]
    reward: float
    done: bool
    episode_total_reward: float
    epsilon: float
    Q: np.ndarray
    state_to_idx: dict[tuple[int, int], int]
    is_episode_end: bool


class BaseQLearningAgent(abc.ABC):
    def __init__(
        self,
        *,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
        seed: int | None = None,
        strategy: str = "eps_greedy",
    ) -> None:
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_decay = float(epsilon_decay)
        self.epsilon_min = float(epsilon_min)
        if strategy not in ("eps_greedy", "softmax"):
            raise ValueError(f"unknown strategy: {strategy!r}")
        self.strategy = strategy
        self._rng = np.random.default_rng(seed)

    def _select_action(self, Q: np.ndarray, s_idx: int) -> Action:
        if self.strategy == "softmax":
            temperature = max(self.epsilon, 1e-6)
            logits = Q[s_idx] / temperature
            logits = logits - float(np.max(logits))
            probs = np.exp(logits)
            probs /= probs.sum()
            return Action(int(self._rng.choice(4, p=probs)))

        if float(self._rng.random()) < self.epsilon:
            return Action(int(self._rng.integers(0, 4)))
        return Action(int(np.argmax(Q[s_idx])))

    @abc.abstractmethod
    def _initialize_Q(self, num_states: int) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def _q_view(self, Q: Any) -> np.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
    def _td_update(
        self,
        Q: Any,
        s_idx: int,
        a_idx: int,
        s2_idx: int,
        reward: float,
    ) -> None:
        raise NotImplementedError

    def _make_metrics(
        self,
        episode_rewards: list[float],
        episode_steps: list[int],
        epsilons: list[float],
    ) -> QLearningMetrics:
        return QLearningMetrics(episode_rewards=episode_rewards, episode_steps=episode_steps, epsilons=epsilons)

    def train_stream(
        self,
        env: GridWorld,
        *,
        num_episodes: int = 1000,
        max_steps: int = 200,
    ) -> Iterator[StepEvent]:
        states = list(env.states())
        state_to_idx = {s: i for i, s in enumerate(states)}
        Q = self._initialize_Q(len(states))

        episode_rewards: list[float] = []
        episode_steps: list[int] = []
        epsilons: list[float] = []

        for ep in range(num_episodes):
            s = env.reset()
            total = 0.0
            steps = 0

            for t in range(max_steps):
                if env.is_terminal(s):
                    break

                s_idx = state_to_idx[s]
                q_view = self._q_view(Q)
                a = self._select_action(q_view, s_idx)
                s2, r, done = env.step(a)
                total += float(r)
                steps += 1

                s2_idx = state_to_idx[s2]
                self._td_update(Q, s_idx, int(a), s2_idx, float(r))

                q_view = self._q_view(Q)
                is_episode_end = bool(done) or (t == max_steps - 1)
                yield StepEvent(
                    episode=ep,
                    step=steps,
                    state=s,
                    action=a,
                    next_state=s2,
                    reward=float(r),
                    done=bool(done),
                    episode_total_reward=total,
                    epsilon=self.epsilon,
                    Q=q_view,
                    state_to_idx=state_to_idx,
                    is_episode_end=is_episode_end,
                )

                s = s2
                if done:
                    break

            episode_rewards.append(total)
            episode_steps.append(steps)
            epsilons.append(self.epsilon)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        self._last_Q = self._q_view(Q)
        self._last_metrics = self._make_metrics(episode_rewards, episode_steps, epsilons)
        self._last_state_to_idx = state_to_idx

    def train(
        self,
        env: GridWorld,
        *,
        num_episodes: int = 1000,
        max_steps: int = 200,
    ) -> tuple[np.ndarray, QLearningMetrics, dict[tuple[int, int], int]]:
        for _ in self.train_stream(env, num_episodes=num_episodes, max_steps=max_steps):
            pass
        return self._last_Q, self._last_metrics, self._last_state_to_idx

    def evaluate_greedy(
        self,
        env: GridWorld,
        Q: np.ndarray,
        state_to_idx: dict[tuple[int, int], int],
        *,
        max_steps: int = 500,
    ) -> tuple[list[tuple[int, int]], float]:
        s = env.reset()
        trajectory = [s]
        total = 0.0
        for _ in range(max_steps):
            if env.is_terminal(s):
                break
            s_idx = state_to_idx[s]
            a = Action(int(np.argmax(Q[s_idx])))
            s, r, done = env.step(a)
            trajectory.append(s)
            total += float(r)
            if done:
                break
        return trajectory, total


    @staticmethod
    def get_policy(Q: np.ndarray, idx_to_state: list[tuple[int, int]]) -> dict[tuple[int, int], Action]:
        return {s: Action(int(np.argmax(Q[i]))) for i, s in enumerate(idx_to_state)}

    @staticmethod
    def get_value_function(Q: np.ndarray, idx_to_state: list[tuple[int, int]]) -> dict[tuple[int, int], float]:
        return {s: float(np.max(Q[i])) for i, s in enumerate(idx_to_state)}


class QLearningAgent(BaseQLearningAgent):
    def _initialize_Q(self, num_states: int) -> np.ndarray:
        return np.zeros((num_states, 4), dtype=float)

    def _q_view(self, Q: np.ndarray) -> np.ndarray:
        return Q

    def _td_update(
        self,
        Q: np.ndarray,
        s_idx: int,
        a_idx: int,
        s2_idx: int,
        reward: float,
    ) -> None:
        td_target = reward + self.gamma * float(np.max(Q[s2_idx]))
        Q[s_idx, a_idx] += self.alpha * (td_target - Q[s_idx, a_idx])


class DoubleQLearningAgent(BaseQLearningAgent):
    def _initialize_Q(self, num_states: int) -> tuple[np.ndarray, np.ndarray]:
        return (np.zeros((num_states, 4), dtype=float), np.zeros((num_states, 4), dtype=float))

    def _q_view(self, Q: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        Q1, Q2 = Q
        return Q1 + Q2

    def _td_update(
        self,
        Q: tuple[np.ndarray, np.ndarray],
        s_idx: int,
        a_idx: int,
        s2_idx: int,
        reward: float,
    ) -> None:
        Q1, Q2 = Q
        if float(self._rng.random()) < 0.5:
            best_action = int(np.argmax(Q1[s2_idx]))
            td_target = reward + self.gamma * float(Q2[s2_idx, best_action])
            Q1[s_idx, a_idx] += self.alpha * (td_target - Q1[s_idx, a_idx])
        else:
            best_action = int(np.argmax(Q2[s2_idx]))
            td_target = reward + self.gamma * float(Q1[s2_idx, best_action])
            Q2[s_idx, a_idx] += self.alpha * (td_target - Q2[s_idx, a_idx])
