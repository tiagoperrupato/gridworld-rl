from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .environment import GridWorld
from .q_learning import QLearningAgent
from .run_dir import write_config, write_metrics
from .value_iteration import ValueIterationSolver
from .visualization import (
    plot_comparison,
    plot_convergence,
    plot_learning_curve,
    plot_multi_curve,
    plot_policy,
    plot_trajectory,
    plot_value_map,
)


@dataclass(frozen=True)
class ExperimentConfig:
    gamma: float = 0.99
    theta: float = 1e-6
    episodes: int = 1000
    max_steps: int = 200
    alpha: float = 0.1
    epsilon: float = 1.0
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.05


class ExperimentRunner:
    """Orchestrates VI + Q-learning experiments into a structured run directory.

    Expected layout (created by `run_dir.create_run_dir`):
        <run_dir>/
            vi/
            ql/
            comparisons/
    """

    def __init__(self, *, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.vi_dir = self.run_dir / "vi"
        self.ql_dir = self.run_dir / "ql"
        self.cmp_dir = self.run_dir / "comparisons"
        for d in (self.vi_dir, self.ql_dir, self.cmp_dir):
            d.mkdir(parents=True, exist_ok=True)

    def run_all(
        self,
        env: GridWorld,
        *,
        config: ExperimentConfig = ExperimentConfig(),
        seed: int | None = None,
        env_meta: dict[str, Any] | None = None,
    ) -> None:
        # 1) Value Iteration
        vi = ValueIterationSolver(env, gamma=config.gamma, theta=config.theta)
        vi_res = vi.solve()

        plot_value_map(vi_res.V, env, save_path=self.vi_dir / "value_map.png")
        plot_policy(vi_res.policy, env, save_path=self.vi_dir / "policy.png")
        plot_convergence(vi_res.deltas, save_path=self.vi_dir / "convergence.png")
        vi_traj, vi_return = vi.evaluate_policy(vi_res.policy, max_steps=config.max_steps)
        plot_trajectory(
            vi_traj,
            env,
            title=f"VI trajectory (return={vi_return:.3f})",
            save_path=self.vi_dir / "trajectory.png",
        )

        vi_rollout_rewards: list[float] = []
        for _ in range(min(200, config.episodes)):
            _traj, ret = vi.evaluate_policy(vi_res.policy, max_steps=config.max_steps)
            vi_rollout_rewards.append(ret)

        # 2) Q-learning
        q_agent = QLearningAgent(
            alpha=config.alpha,
            gamma=config.gamma,
            epsilon=config.epsilon,
            epsilon_decay=config.epsilon_decay,
            epsilon_min=config.epsilon_min,
            seed=seed,
        )
        Q, metrics, state_to_idx = q_agent.train(env, num_episodes=config.episodes, max_steps=config.max_steps)
        idx_to_state = [s for s, _i in sorted(state_to_idx.items(), key=lambda kv: kv[1])]
        q_V = q_agent.get_value_function(Q, idx_to_state)
        q_policy = q_agent.get_policy(Q, idx_to_state)
        q_traj, q_return = q_agent.evaluate_greedy(env, Q, state_to_idx, max_steps=config.max_steps)

        plot_value_map(q_V, env, title="Q-learning V(s)=max_a Q(s,a)", save_path=self.ql_dir / "value_map.png")
        plot_policy(q_policy, env, title="Q-learning greedy policy", save_path=self.ql_dir / "policy.png")
        plot_trajectory(
            q_traj,
            env,
            title=f"Q-learning greedy trajectory (return={q_return:.3f})",
            save_path=self.ql_dir / "trajectory.png",
        )
        plot_learning_curve(metrics.episode_rewards, save_path=self.ql_dir / "learning_curve.png")

        plot_comparison(
            vi_rollout_rewards,
            metrics.episode_rewards[: len(vi_rollout_rewards)],
            save_path=self.cmp_dir / "vi_vs_ql_rewards.png",
        )

        # 3) Gamma comparison (Q-learning)
        gammas = [0.5, 0.9, 0.99]
        gamma_rewards: dict[float, list[float]] = {}
        for g in gammas:
            agent = QLearningAgent(
                alpha=config.alpha,
                gamma=g,
                epsilon=config.epsilon,
                epsilon_decay=config.epsilon_decay,
                epsilon_min=config.epsilon_min,
                seed=seed,
            )
            _Q, m, _ = agent.train(env, num_episodes=config.episodes, max_steps=config.max_steps)
            gamma_rewards[g] = m.episode_rewards

        plot_multi_curve(
            gamma_rewards,
            title="Q-learning: gamma comparison",
            save_path=self.cmp_dir / "gamma_comparison.png",
        )

        # 4) Exploration comparison (epsilon settings)
        eps_setups = {
            "eps=1.0 decay=0.995": (1.0, 0.995),
            "eps=0.5 decay=0.997": (0.5, 0.997),
            "eps=0.2 decay=0.999": (0.2, 0.999),
        }
        exploration_rewards: dict[str, list[float]] = {}
        for label, (eps, decay) in eps_setups.items():
            agent = QLearningAgent(
                alpha=config.alpha,
                gamma=config.gamma,
                epsilon=eps,
                epsilon_decay=decay,
                epsilon_min=config.epsilon_min,
                seed=seed,
            )
            _Q, m, _ = agent.train(env, num_episodes=config.episodes, max_steps=config.max_steps)
            exploration_rewards[label] = m.episode_rewards

        plot_multi_curve(
            exploration_rewards,
            title="Q-learning: exploration comparison",
            save_path=self.cmp_dir / "exploration_comparison.png",
        )

        # 5) Serialize config + metrics for reproducibility.
        write_config(
            self.run_dir,
            config,
            extra={
                "seed": seed,
                "env": env_meta or {},
            },
        )
        write_metrics(
            self.run_dir,
            {
                "vi": {
                    "final_return": vi_return,
                    "deltas": vi_res.deltas,
                    "rollout_rewards": vi_rollout_rewards,
                    "iterations": len(vi_res.deltas),
                },
                "ql": {
                    "final_return": q_return,
                    "episode_rewards": metrics.episode_rewards,
                    "episode_steps": metrics.episode_steps,
                    "epsilons": metrics.epsilons,
                },
                "gamma_comparison": {str(k): v for k, v in gamma_rewards.items()},
                "exploration_comparison": exploration_rewards,
            },
        )
