from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.collections import LineCollection
import numpy as np

from .environment import Action, Cell, GridWorld, _ACTION_TO_DELTA


# Colors reused for the S/G/T callouts so every grid plot is consistent.
_CELL_COLORS: dict[Cell, str] = {
    Cell.START: "#1f77b4",
    Cell.GOAL: "#2ca02c",
    Cell.TRAP: "#d62728",
}

_CELL_LETTERS: dict[Cell, str] = {
    Cell.START: "S",
    Cell.GOAL: "G",
    Cell.TRAP: "T",
}


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
    # `angles='xy', scale_units='xy', scale=1` makes V interpret in data
    # coordinates so a (-1) row-delta visually points "up" on imshow's
    # origin='upper' axis (row 0 at the top). Without this, quiver defaults to
    # display-coordinate angles and renders every UP/DOWN arrow inverted.
    plt.quiver(
        X, Y, U, Vv,
        pivot="middle", color="tab:blue", zorder=2,
        angles="xy", scale_units="xy", scale=1.6,
    )
    _annotate_cells(env)
    plt.title(title)
    plt.xticks(range(env.w))
    plt.yticks(range(env.h))
    plt.tight_layout()
    _save_or_show(save_path)


def plot_trajectory(trajectory, env: GridWorld, *, title="Trajectory", save_path: Path | None = None) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(np.where(env.grid == Cell.WALL, 1, 0), cmap="gray_r", alpha=0.25)

    if trajectory and len(trajectory) >= 2:
        xs = np.asarray([c for r, c in trajectory], dtype=float)
        ys = np.asarray([r for r, c in trajectory], dtype=float)

        # Build per-segment LineCollection colored by step index so early/late
        # parts of the rollout are visually distinct without relying on arrows.
        points = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        step_idx = np.arange(len(segments))
        lc = LineCollection(
            segments,
            cmap="plasma",
            array=step_idx,
            linewidths=2.5,
            zorder=2,
        )
        ax.add_collection(lc)
        cbar = fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Step index")

        # Sparse directional arrows every few steps for long trajectories.
        arrow_every = max(1, len(segments) // 6)
        for i in range(0, len(segments), arrow_every):
            x0, y0 = xs[i], ys[i]
            x1, y1 = xs[i + 1], ys[i + 1]
            if x0 == x1 and y0 == y1:
                continue
            ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.0, alpha=0.7),
                zorder=3,
            )

        # Start marker: hollow blue circle. End marker: star colored by outcome.
        ax.plot(
            xs[0],
            ys[0],
            marker="o",
            markersize=14,
            markerfacecolor="none",
            markeredgecolor="#1f77b4",
            markeredgewidth=2.0,
            linestyle="None",
            zorder=4,
        )
        final_cell = Cell(int(env.grid[int(ys[-1]), int(xs[-1])]))
        end_color = {
            Cell.GOAL: "#2ca02c",
            Cell.TRAP: "#d62728",
        }.get(final_cell, "#888888")
        ax.plot(
            xs[-1],
            ys[-1],
            marker="*",
            markersize=18,
            color=end_color,
            markeredgecolor="black",
            markeredgewidth=1.0,
            linestyle="None",
            zorder=4,
        )
    elif trajectory:
        # Single-state degenerate trajectory (already at terminal, etc.).
        r, c = trajectory[0]
        ax.plot(c, r, marker="o", markersize=12, color="#1f77b4", zorder=3)

    _annotate_cells(env, ax=ax)
    ax.set_title(title)
    ax.set_xticks(range(env.w))
    ax.set_yticks(range(env.h))
    fig.tight_layout()
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


def plot_multi_curve_bands(
    curves: Mapping[str, Mapping[str, np.ndarray]],
    *,
    title: str,
    xlabel: str = "Episode",
    ylabel: str = "Cumulative reward",
    save_path: Path | None = None,
    smooth: int = 20,
) -> None:
    """Plot mean line + std band for each series in `curves`.

    Input schema: ``{label: {"mean": np.ndarray, "std": np.ndarray}}``. The two
    arrays are expected to have the same length. When ``smooth > 1`` both the
    mean and std are trailing-averaged over a ``smooth``-episode window, which
    keeps multi-seed plots readable when individual returns are noisy.
    """
    plt.figure(figsize=(7, 4))
    for label, series in curves.items():
        mean = np.asarray(series["mean"], dtype=float)
        std = np.asarray(series["std"], dtype=float)
        if mean.size == 0:
            continue
        if smooth and smooth > 1:
            mean_plot = _trailing_mean(mean, smooth)
            std_plot = _trailing_mean(std, smooth)
        else:
            mean_plot = mean
            std_plot = std
        x = np.arange(mean_plot.size)
        line, = plt.plot(x, mean_plot, label=str(label), linewidth=1.5)
        plt.fill_between(
            x,
            mean_plot - std_plot,
            mean_plot + std_plot,
            color=line.get_color(),
            alpha=0.2,
            linewidth=0,
        )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    _save_or_show(save_path)


def plot_visit_heatmap(
    visit_counts: np.ndarray,
    env: GridWorld,
    *,
    title: str = "Aggregated visit heatmap",
    save_path: Path | None = None,
    log_scale: bool = True,
) -> None:
    """Render a state-visit heatmap over the grid, with walls masked.

    `visit_counts` is a ``(h, w)`` array of integer or float counts (already
    summed across seeds by the caller). When ``log_scale`` is True we plot
    ``log10(1 + counts)`` to keep frequently-visited cells from drowning out
    the rest of the grid.
    """
    counts = np.asarray(visit_counts, dtype=float)
    data = np.log10(1.0 + counts) if log_scale else counts
    mask = env.grid == Cell.WALL
    masked = np.ma.array(data, mask=mask)

    plt.figure(figsize=(6, 5))
    plt.imshow(masked, cmap="magma")
    plt.colorbar(label="log10(1 + visits)" if log_scale else "visits")
    _annotate_cells(env)
    plt.title(title)
    plt.xticks(range(env.w))
    plt.yticks(range(env.h))
    plt.tight_layout()
    _save_or_show(save_path)


def _trailing_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Trailing moving average with partial windows at the left edge.

    For index i we average x[max(0, i-window+1) .. i] so the output length
    matches the input and early values aren't artificially zero.
    """
    if window <= 1 or x.size == 0:
        return x.astype(float, copy=False)
    cumsum = np.concatenate(([0.0], np.cumsum(x)))
    out = np.empty_like(x, dtype=float)
    for i in range(x.size):
        start = max(0, i - window + 1)
        count = i - start + 1
        out[i] = (cumsum[i + 1] - cumsum[start]) / count
    return out


def _annotate_cells(env: GridWorld, *, ax=None) -> None:
    ax = ax if ax is not None else plt.gca()
    letter_effects = [path_effects.withStroke(linewidth=2.0, foreground="black")]
    for r in range(env.h):
        for c in range(env.w):
            cell = Cell(int(env.grid[r, c]))
            if cell == Cell.WALL:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="black", zorder=1))
                continue
            if cell not in _CELL_COLORS:
                continue
            ax.add_patch(
                plt.Circle(
                    (c, r),
                    0.35,
                    facecolor=_CELL_COLORS[cell],
                    edgecolor="black",
                    linewidth=1.2,
                    zorder=5,
                )
            )
            txt = ax.text(
                c,
                r,
                _CELL_LETTERS[cell],
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
                fontsize=11,
                zorder=6,
            )
            txt.set_path_effects(letter_effects)


def _save_or_show(save_path: Path | None) -> None:
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()
