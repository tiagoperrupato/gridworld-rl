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
    p.add_argument("--output-dir", type=Path, default=Path("output"), help="Directory to save plots")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str((args.output_dir / ".mplconfig").resolve()))
    os.environ.setdefault("XDG_CACHE_HOME", str((args.output_dir / ".cache").resolve()))

    from src.experiment import ExperimentConfig, ExperimentRunner

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
    runner = ExperimentRunner(output_dir=args.output_dir)
    runner.run_all(env, config=cfg, seed=args.seed)
    print(f"Saved plots to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

