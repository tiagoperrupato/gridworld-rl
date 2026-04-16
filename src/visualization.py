from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .environment import Action, Cell, GridWorld, _ACTION_TO_DELTA


def plot_value_map(V, env: GridWorld, *, title="Value Function V(s)", save_path: Path | None = None) -> None:
    values = np.full((env.h, env.w), fill_value=np.nan, dtype=float)
    for (r, c), v in V.items():
        values[r, c] = float(v)

    plt.figure(figsize=(6, 5))
    plt.imshow(np.ma.array(values, mask=np.isnan(values)), cmap="viridis")
    plt.colorbar(label="V(s)")
    _annotate_cells(env)
    plt.title(title)
    plt.xticks(range(env.w))
    plt.yticks(range(env.h))
    plt.gca().invert_yaxis()
    plt.tight_layout()
    _save_or_show(save_path)


def plot_policy(policy, env: GridWorld, *, title="Policy", save_path: Path | None = None) -> None:
    # Quiver uses data coords (x=col, y=row) matching imshow. Environment deltas are (dr, dc).
    X, Y = np.meshgrid(np.arange(env.w), np.arange(env.h))
    U = np.zeros((env.h, env.w), dtype=float)
    Vv = np.zeros((env.h, env.w), dtype=float)

    for r in range(env.h):
        for c in range(env.w):
            s = (r, c)
            if env.is_wall(s) or env.is_terminal(s):
                continue
            a = policy.get(s)
            if a is None:
                continue
            dr, dc = _ACTION_TO_DELTA[Action(int(a))]
            U[r, c] = float(dc)
            Vv[r, c] = float(dr)

    plt.figure(figsize=(6, 5))
    plt.imshow(np.where(env.grid == Cell.WALL, 1, 0), cmap="gray_r", alpha=0.25)
    plt.quiver(X, Y, U, Vv, pivot="middle", color="tab:blue")
    _annotate_cells(env)
    plt.title(title)
    plt.xticks(range(env.w))
    plt.yticks(range(env.h))
    plt.gca().invert_yaxis()
    plt.tight_layout()
    _save_or_show(save_path)


def plot_trajectory(trajectory, env: GridWorld, *, title="Trajectory", save_path: Path | None = None) -> None:
    plt.figure(figsize=(6, 5))
    plt.imshow(np.where(env.grid == Cell.WALL, 1, 0), cmap="gray_r", alpha=0.25)
    if trajectory:
        xs = [c for r, c in trajectory]
        ys = [r for r, c in trajectory]
        plt.plot(xs, ys, "-o", color="tab:orange", linewidth=2, markersize=4)
    _annotate_cells(env)
    plt.title(title)
    plt.xticks(range(env.w))
    plt.yticks(range(env.h))
    plt.gca().invert_yaxis()
    plt.tight_layout()
    _save_or_show(save_path)


def plot_learning_curve(rewards, *, title="Learning curve (reward per episode)", save_path: Path | None = None) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(rewards, color="tab:green", linewidth=1)
    plt.xlabel("Episode")
    plt.ylabel("Cumulative reward")
    plt.title(title)
    plt.tight_layout()
    _save_or_show(save_path)


def plot_convergence(deltas, *, title="Value Iteration convergence (delta)", save_path: Path | None = None) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(deltas, color="tab:purple", linewidth=1)
    plt.xlabel("Iteration")
    plt.ylabel("max |ΔV|")
    plt.title(title)
    plt.yscale("log")
    plt.tight_layout()
    _save_or_show(save_path)


def plot_comparison(vi_rewards, ql_rewards, *, title="Bellman vs Q-learning", save_path: Path | None = None) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(vi_rewards, label="VI eval (per rollout)", linewidth=2)
    plt.plot(ql_rewards, label="Q-learning (per episode)", linewidth=1)
    plt.xlabel("Episode / rollout")
    plt.ylabel("Reward")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    _save_or_show(save_path)


def plot_multi_curve(
    curves: dict[object, list[float]],
    *,
    title: str,
    xlabel: str = "Episode",
    ylabel: str = "Cumulative reward",
    save_path: Path | None = None,
) -> None:
    plt.figure(figsize=(7, 4))
    for label, ys in curves.items():
        plt.plot(ys, label=str(label), linewidth=1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    _save_or_show(save_path)


def _annotate_cells(env: GridWorld) -> None:
    ax = plt.gca()
    for r in range(env.h):
        for c in range(env.w):
            cell = Cell(int(env.grid[r, c]))
            if cell == Cell.WALL:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="black"))
            elif cell == Cell.GOAL:
                plt.text(c, r, "G", ha="center", va="center", color="white", fontweight="bold")
            elif cell == Cell.TRAP:
                plt.text(c, r, "T", ha="center", va="center", color="white", fontweight="bold")
            elif cell == Cell.START:
                plt.text(c, r, "S", ha="center", va="center", color="white", fontweight="bold")


def _save_or_show(save_path: Path | None) -> None:
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()

