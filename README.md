# Gridworld RL (Value Iteration + Q-learning)

This project implements a complete **Reinforcement Learning** pipeline for the **“Gridworld com Perigos”** assignment:

- **Environment** modeled as an MDP (states, actions, transitions, rewards, terminal states)
- **Planning** with **Value Iteration** (Bellman optimality updates)
- **Learning** with **Q-learning** (ε-greedy exploration + ε decay)
- **Experimental protocol + visualizations** (learning curves, value maps, policies, trajectories, comparisons)

All core algorithms are implemented **from scratch** (no RL libraries). Only `numpy` and `matplotlib` are used.

---

## Project layout

Each file under `src/` has a single responsibility—no duplicate implementations.

```
project2-gridworld/
  src/
    __init__.py           # Public API (re-exports main classes and plot helpers)
    environment.py        # GridWorld MDP + transition enumeration
    value_iteration.py    # Value Iteration solver (Bellman updates)
    q_learning.py         # Q-learning agent (ε-greedy, tabular Q)
    visualization.py      # Matplotlib plots (maps, policies, curves, comparisons)
    experiment.py         # Orchestrates VI + Q-learning experiments; writes output/
  main.py                 # CLI entry point
  requirements.txt        # numpy, matplotlib
  output/                 # generated figures (created at runtime)
  2026s1_IA_T2.md         # assignment specification
```

Import from the package root when convenient:

```python
from src import GridWorld, ValueIterationSolver, QLearningAgent, ExperimentRunner
```

---

## Setup

### 1) Create a virtual environment (recommended)

On macOS/Homebrew Python you may hit **PEP 668 (externally-managed environment)** if you try to install packages globally, so use a local venv:

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
```

---

## Run

### Default run (full pipeline)

```bash
./.venv/bin/python main.py
```

This runs:

- Value Iteration on the known MDP (planning)
- Q-learning training (learning)
- Gamma comparison for Q-learning
- Exploration comparison (different ε/decay setups)
- Saves all plots into `output/`

### Faster run (good for sanity checks)

```bash
./.venv/bin/python main.py --episodes 200 --seed 0
```

---

## CLI options

Run `--help` to see the full list:

```bash
./.venv/bin/python main.py --help
```

Main options:

- `--stochastic`: enables stochastic transitions (“wind”)
- `--wind-prob`: probability of executing the intended action under wind (default `0.8`)
- `--gamma`: discount factor (default `0.99`)
- `--theta`: convergence threshold for Value Iteration (default `1e-6`)
- `--episodes`: Q-learning episodes (default `1000`)
- `--max-steps`: max steps per episode/rollout (default `200`)
- `--alpha`: Q-learning learning rate (default `0.1`)
- `--epsilon`, `--epsilon-decay`, `--epsilon-min`: ε-greedy parameters
- `--seed`: RNG seed for reproducibility
- `--output-dir`: where figures are written (default `output/`)

---

## The environment (MDP)

The environment is a grid where each non-wall cell is a state \(s=(row, col)\).

### Cell types (layout symbols)

Layouts are created from a list of strings using:

- `GridWorld.from_layout([...])`
- or the built-in `GridWorld.default(...)`

Symbols:

- `S`: **start**
- `G`: **goal** (**terminal**, reward `+1.0`)
- `T`: **trap** (**terminal**, reward `-1.0`)
- `#`: **wall** (blocked cell)
- `.`: **empty**

### Actions

Four discrete actions:

- `UP`, `DOWN`, `LEFT`, `RIGHT`

### Transitions

- **Deterministic** (default): the chosen action is applied; hitting a wall/boundary leaves the agent in place.
- **Stochastic wind** (`--stochastic`): intended direction succeeds with probability `wind_prob` and **perpendicular** directions share the remaining probability equally.

Value Iteration uses `GridWorld.get_transitions(state, action)` to enumerate all \((p, s', r, done)\) outcomes.

### Rewards

Configured in `GridWorld(...)`:

- Goal: `+1.0`
- Trap: `-1.0`
- Step cost: `-0.04` (encourages shorter paths)

---

## Algorithms

### Value Iteration (planning)

Implemented in `src/value_iteration.py`.

Update rule:

\[
V(s) \leftarrow \max_a \sum_{s'} T(s,a,s')\big[R(s,a,s') + \gamma V(s')\big]
\]

Outputs:

- `V(s)` value function
- Greedy policy derived from `V`
- Convergence curve (`max |ΔV|` per iteration)

### Q-learning (model-free learning)

Implemented in `src/q_learning.py`.

Update rule:

\[
Q(s,a) \leftarrow Q(s,a) + \alpha\Big[r + \gamma \max_{a'} Q(s',a') - Q(s,a)\Big]
\]

Features:

- ε-greedy exploration
- ε decay per episode (bounded by `epsilon_min`)
- Tracks metrics:
  - reward per episode
  - steps per episode
  - epsilon over time

---

## Outputs (figures)

After running, `output/` contains PNG files such as:

- **Value Iteration**
  - `vi_value_map.png`
  - `vi_policy.png`
  - `vi_trajectory.png`
  - `vi_convergence.png`
- **Q-learning**
  - `ql_value_map.png`
  - `ql_policy.png`
  - `ql_trajectory.png`
  - `ql_learning_curve.png`
- **Comparisons**
  - `vi_vs_ql_rewards.png`
  - `ql_gamma_comparison.png`
  - `ql_exploration_comparison.png`

---

## Notes on headless execution (Matplotlib)

`main.py` forces a non-interactive backend (`Agg`) and sets cache directories under `output/` to avoid permission issues when running in restricted environments.

---

## Customizing the grid

You can edit the default layout in `GridWorld.default()` (`src/environment.py`) or create your own:

```python
from src.environment import GridWorld

env = GridWorld.from_layout(
    [
        "S..#.",
        ".#..T",
        "...#.",
        ".#...",
        "...G.",
    ],
    stochastic=True,
    wind_prob=0.8,
    seed=0,
)
```

Then run the experiment runner similarly to `main.py`.
