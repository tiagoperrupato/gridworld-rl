from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.environment import GridWorld
from src.maps import get_map, map_keys


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Gridworld RL: Value Iteration and Q-learning")
    p.add_argument("--stochastic", action="store_true", help="Enable stochastic wind transitions")
    p.add_argument("--wind-prob", type=float, default=0.8, help="Probability of intended action under wind")
    p.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    p.add_argument("--theta", type=float, default=1e-6, help="Value Iteration convergence threshold")
    p.add_argument("--episodes", type=int, default=1000, help="Q-learning training episodes")
    p.add_argument("--max-steps", type=int, default=200, help="Max steps per episode / rollout")
    p.add_argument("--alpha", type=float, default=0.1, help="Q-learning learning rate")
    p.add_argument("--epsilon", type=float, default=1.0, help="Initial epsilon for epsilon-greedy")
    p.add_argument("--epsilon-decay", type=float, default=0.995, help="Epsilon decay per episode")
    p.add_argument("--epsilon-min", type=float, default=0.05, help="Minimum epsilon")
    p.add_argument("--seed", type=int, default=None, help="Single RNG seed (ignored when --seeds is set)")
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Multi-seed list for confidence bands (default: 0 1 2 3 4). Overrides --seed.",
    )
    p.add_argument(
        "--maps",
        nargs="+",
        default=["default"],
        choices=[*map_keys(), "all"],
        help=(
            "Which built-in maps to sweep over. Pass any combination of "
            f"{map_keys()} or 'all' for every map. Each map gets its own "
            "subdirectory inside the run folder."
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Root directory for runs (default: output/). Runs go into <output-dir>/runs/<timestamp>_<slug>/",
    )
    p.add_argument("--run-name", type=str, default="default", help="Slug appended to the run directory name")
    return p


def _resolve_maps(map_args: list[str]) -> list:
    """Expand 'all' and de-dupe while preserving order."""
    keys: list[str] = []
    for k in map_args:
        if k == "all":
            for mk in map_keys():
                if mk not in keys:
                    keys.append(mk)
        elif k not in keys:
            keys.append(k)
    return [get_map(k) for k in keys]


def main() -> None:
    args = build_arg_parser().parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str((args.output_dir / ".mplconfig").resolve()))
    os.environ.setdefault("XDG_CACHE_HOME", str((args.output_dir / ".cache").resolve()))

    from src.experiment import ExperimentConfig, ExperimentRunner, summarize_across_maps
    from src.run_dir import create_run_dir, update_latest_symlink, write_config, write_metrics

    if args.seeds is not None:
        seeds_list: list[int] = list(args.seeds)
    elif args.seed is not None:
        seeds_list = [args.seed]
    else:
        seeds_list = [0, 1, 2, 3, 4]

    selected_maps = _resolve_maps(args.maps)
    if not selected_maps:
        raise SystemExit("--maps must select at least one map")

    cfg = ExperimentConfig(
        gamma=args.gamma,
        theta=args.theta,
        episodes=args.episodes,
        max_steps=args.max_steps,
        alpha=args.alpha,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
    )

    run_dir = create_run_dir(args.output_dir, slug=args.run_name)
    runner = ExperimentRunner(run_dir=run_dir)

    per_map_metrics: dict[str, dict] = {}
    print(f"Run directory: {run_dir.resolve()}")
    print(f"Maps to sweep: {[m.name for m in selected_maps]}")
    for mc in selected_maps:
        env = GridWorld.from_layout(
            mc.layout,
            stochastic=args.stochastic,
            wind_prob=args.wind_prob,
            seed=seeds_list[0],
        )
        env_meta = {
            "layout_name": mc.name,
            "layout_slug": mc.slug,
            "stochastic": args.stochastic,
            "wind_prob": args.wind_prob,
            "shape": list(env.grid.shape),
        }
        print(f"  -> {mc.name} (shape={env_meta['shape']})  subdir={mc.slug}/")
        per_map_metrics[mc.slug] = runner.run_all(
            env,
            config=cfg,
            seed=args.seed,
            seeds=seeds_list,
            env_meta=env_meta,
            map_subdir=mc.slug,
        )

    write_config(
        run_dir,
        cfg,
        extra={
            "seed": args.seed,
            "seeds": seeds_list,
            "maps": [m.slug for m in selected_maps],
            "map_names": [m.name for m in selected_maps],
            "stochastic": args.stochastic,
            "wind_prob": args.wind_prob,
        },
    )
    write_metrics(
        run_dir,
        {
            "cross_map_summary": summarize_across_maps(per_map_metrics),
            "per_map": per_map_metrics,
        },
    )

    update_latest_symlink(run_dir)
    print(f"Run directory: {run_dir.resolve()}")
    print(f"Latest symlink: {(args.output_dir / 'latest').resolve()}")


if __name__ == "__main__":
    main()
