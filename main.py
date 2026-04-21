from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.environment import GridWorld


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
    p.add_argument("--seed", type=int, default=None, help="RNG seed")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Root directory for runs (default: output/). Runs go into <output-dir>/runs/<timestamp>_<slug>/",
    )
    p.add_argument("--run-name", type=str, default="default", help="Slug appended to the run directory name")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str((args.output_dir / ".mplconfig").resolve()))
    os.environ.setdefault("XDG_CACHE_HOME", str((args.output_dir / ".cache").resolve()))

    from src.experiment import ExperimentConfig, ExperimentRunner
    from src.run_dir import create_run_dir, update_latest_symlink

    env = GridWorld.default(stochastic=args.stochastic, wind_prob=args.wind_prob, seed=args.seed)
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
    env_meta = {
        "layout_name": "default",
        "stochastic": args.stochastic,
        "wind_prob": args.wind_prob,
        "shape": list(env.grid.shape),
    }

    runner = ExperimentRunner(run_dir=run_dir)
    runner.run_all(env, config=cfg, seed=args.seed, env_meta=env_meta)
    update_latest_symlink(run_dir)
    print(f"Run directory: {run_dir.resolve()}")
    print(f"Latest symlink: {(args.output_dir / 'latest').resolve()}")


if __name__ == "__main__":
    main()
