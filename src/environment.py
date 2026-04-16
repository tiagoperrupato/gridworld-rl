from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterator, Sequence

import numpy as np


class Cell(IntEnum):
    EMPTY = 0
    WALL = 1
    START = 2
    GOAL = 3
    TRAP = 4


class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


_ACTION_TO_DELTA: dict[Action, tuple[int, int]] = {
    Action.UP: (-1, 0),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1),
}


def _perpendicular_actions(action: Action) -> tuple[Action, Action]:
    if action in (Action.UP, Action.DOWN):
        return (Action.LEFT, Action.RIGHT)
    return (Action.UP, Action.DOWN)


@dataclass(frozen=True)
class Transition:
    prob: float
    next_state: tuple[int, int]
    reward: float
    done: bool


class GridWorld:
    """
    Gridworld MDP.

    Grid symbols in layouts:
      - 'S': start
      - 'G': goal (terminal, reward +1)
      - 'T': trap (terminal, reward -1)
      - '#': wall (blocked)
      - '.': empty
    """

    def __init__(
        self,
        grid: np.ndarray,
        *,
        stochastic: bool = False,
        wind_prob: float = 0.8,
        step_cost: float = -0.04,
        goal_reward: float = 1.0,
        trap_reward: float = -1.0,
        seed: int | None = None,
    ) -> None:
        if grid.ndim != 2:
            raise ValueError("grid must be a 2D array")
        self.grid = grid.astype(np.int64, copy=False)
        self.h, self.w = self.grid.shape

        self.stochastic = stochastic
        self.wind_prob = float(wind_prob)
        if not (0.0 <= self.wind_prob <= 1.0):
            raise ValueError("wind_prob must be in [0, 1]")

        self.step_cost = float(step_cost)
        self.goal_reward = float(goal_reward)
        self.trap_reward = float(trap_reward)

        self._rng = np.random.default_rng(seed)

        starts = list(zip(*np.where(self.grid == Cell.START)))
        if len(starts) != 1:
            raise ValueError("grid must contain exactly one START cell")
        self.start_state = (int(starts[0][0]), int(starts[0][1]))
        self._state = self.start_state

    @staticmethod
    def from_layout(
        layout: Sequence[str],
        *,
        stochastic: bool = False,
        wind_prob: float = 0.8,
        step_cost: float = -0.04,
        goal_reward: float = 1.0,
        trap_reward: float = -1.0,
        seed: int | None = None,
    ) -> "GridWorld":
        if not layout:
            raise ValueError("layout cannot be empty")
        w = len(layout[0])
        if any(len(row) != w for row in layout):
            raise ValueError("layout rows must have equal length")

        mapping = {
            ".": Cell.EMPTY,
            "#": Cell.WALL,
            "S": Cell.START,
            "G": Cell.GOAL,
            "T": Cell.TRAP,
        }
        grid = np.zeros((len(layout), w), dtype=np.int64)
        for r, row in enumerate(layout):
            for c, ch in enumerate(row):
                if ch not in mapping:
                    raise ValueError(f"invalid layout char: {ch!r}")
                grid[r, c] = int(mapping[ch])

        return GridWorld(
            grid,
            stochastic=stochastic,
            wind_prob=wind_prob,
            step_cost=step_cost,
            goal_reward=goal_reward,
            trap_reward=trap_reward,
            seed=seed,
        )

    @staticmethod
    def default(
        *,
        stochastic: bool = False,
        wind_prob: float = 0.8,
        seed: int | None = None,
    ) -> "GridWorld":
        layout = [
            "S...#",
            ".#..T",
            ".#.#.",
            "..#..",
            "...G.",
        ]
        return GridWorld.from_layout(layout, stochastic=stochastic, wind_prob=wind_prob, seed=seed)

    def reset(self) -> tuple[int, int]:
        self._state = self.start_state
        return self._state

    def set_state(self, state: tuple[int, int]) -> None:
        if not self.in_bounds(state):
            raise ValueError("state out of bounds")
        if self.is_wall(state):
            raise ValueError("cannot set state to a wall cell")
        self._state = state

    @property
    def state(self) -> tuple[int, int]:
        return self._state

    def is_terminal(self, state: tuple[int, int]) -> bool:
        r, c = state
        cell = Cell(int(self.grid[r, c]))
        return cell in (Cell.GOAL, Cell.TRAP)

    def in_bounds(self, state: tuple[int, int]) -> bool:
        r, c = state
        return 0 <= r < self.h and 0 <= c < self.w

    def is_wall(self, state: tuple[int, int]) -> bool:
        r, c = state
        return Cell(int(self.grid[r, c])) == Cell.WALL

    def states(self) -> Iterator[tuple[int, int]]:
        for r in range(self.h):
            for c in range(self.w):
                s = (r, c)
                if not self.is_wall(s):
                    yield s

    def actions(self) -> tuple[Action, Action, Action, Action]:
        return (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT)

    def _move(self, state: tuple[int, int], action: Action) -> tuple[int, int]:
        dr, dc = _ACTION_TO_DELTA[action]
        nr, nc = state[0] + dr, state[1] + dc
        ns = (nr, nc)
        if not self.in_bounds(ns) or self.is_wall(ns):
            return state
        return ns

    def _reward_for_transition(self, next_state: tuple[int, int]) -> tuple[float, bool]:
        r, c = next_state
        cell = Cell(int(self.grid[r, c]))
        if cell == Cell.GOAL:
            return self.goal_reward, True
        if cell == Cell.TRAP:
            return self.trap_reward, True
        return self.step_cost, False

    def get_transitions(self, state: tuple[int, int], action: Action) -> list[Transition]:
        if self.is_terminal(state):
            return [Transition(prob=1.0, next_state=state, reward=0.0, done=True)]

        if not self.stochastic:
            ns = self._move(state, action)
            r, done = self._reward_for_transition(ns)
            return [Transition(prob=1.0, next_state=ns, reward=r, done=done)]

        intended = action
        perp_a, perp_b = _perpendicular_actions(action)

        outcomes: list[tuple[float, Action]] = [
            (self.wind_prob, intended),
            ((1.0 - self.wind_prob) / 2.0, perp_a),
            ((1.0 - self.wind_prob) / 2.0, perp_b),
        ]

        # Aggregate probabilities if multiple actions lead to same next_state (walls/bounds).
        agg: dict[tuple[int, int], float] = {}
        for p, a in outcomes:
            ns = self._move(state, a)
            agg[ns] = agg.get(ns, 0.0) + p

        transitions: list[Transition] = []
        for ns, p in agg.items():
            r, done = self._reward_for_transition(ns)
            transitions.append(Transition(prob=p, next_state=ns, reward=r, done=done))

        transitions.sort(key=lambda t: (t.next_state[0], t.next_state[1]))
        return transitions

    def step(self, action: Action) -> tuple[tuple[int, int], float, bool]:
        transitions = self.get_transitions(self._state, action)
        probs = np.array([t.prob for t in transitions], dtype=float)
        idx = int(self._rng.choice(len(transitions), p=probs))
        t = transitions[idx]
        self._state = t.next_state
        return t.next_state, t.reward, t.done

    def render_ascii(self, state: tuple[int, int] | None = None) -> str:
        s = state if state is not None else self._state
        inv = {Cell.EMPTY: ".", Cell.WALL: "#", Cell.START: "S", Cell.GOAL: "G", Cell.TRAP: "T"}
        rows: list[str] = []
        for r in range(self.h):
            chars: list[str] = []
            for c in range(self.w):
                if (r, c) == s and not self.is_terminal((r, c)):
                    chars.append("A")
                else:
                    chars.append(inv[Cell(int(self.grid[r, c]))])
            rows.append("".join(chars))
        return "\n".join(rows)
