from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .environment import GridWorld
from .q_learning import QLearningAgent
from .stats_tools import aggregate_curves, episodes_to_threshold, policy_agreement
from .value_iteration import ValueIterationSolver
from .visualization import (
    plot_comparison,
    plot_convergence,
    plot_learning_curve,
    plot_multi_curve_bands,
    plot_policy,
    plot_trajectory,
    plot_value_map,
    plot_visit_heatmap,
)

_DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)


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


@dataclass(frozen=True)
class _QLRun:
    """Result of a single Q-learning training run used by the analysis helpers."""

    episode_rewards: list[float]
    episode_steps: list[int]
    epsilons: list[float]
    Q: np.ndarray
    state_to_idx: dict[tuple[int, int], int]
    policy: dict[tuple[int, int], Any]
    value_fn: dict[tuple[int, int], float]
    greedy_return: float
    greedy_trajectory: list[tuple[int, int]]
    visit_counts: np.ndarray
    wall_time_seconds: float


class ExperimentRunner:
    """Orchestrates VI + Q-learning experiments into a structured run directory.

    Layout for a single run:
        <run_dir>/
            <map_slug>/
                vi/
                ql/
                comparisons/
                analysis/

    `run_all(...)` writes only its map subtree and returns the metrics dict for
    that map; the caller is responsible for aggregating per-map dicts into a
    top-level `metrics.json` and writing it once with `run_dir.write_metrics`.
    """

    def __init__(self, *, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # Low-level helpers
    # ---------------------------------------------------------------------

    def _build_env(self, template: GridWorld, seed: int | None) -> GridWorld:
        """Clone the environment with a fresh RNG for a specific seed.

        We keep the grid and reward/transition settings identical to `template`
        so all multi-seed runs compare apples to apples; only the RNG used for
        stochastic step sampling changes between seeds.
        """
        return GridWorld(
            grid=template.grid.copy(),
            stochastic=template.stochastic,
            wind_prob=template.wind_prob,
            step_cost=template.step_cost,
            goal_reward=template.goal_reward,
            trap_reward=template.trap_reward,
            seed=seed,
        )

    @staticmethod
    def _env_with_stochasticity(env: GridWorld, *, stochastic: bool) -> GridWorld:
        """Same MDP parameters as ``env`` but with a chosen wind (stochastic) mode."""
        return GridWorld(
            grid=env.grid.copy(),
            stochastic=stochastic,
            wind_prob=env.wind_prob,
            step_cost=env.step_cost,
            goal_reward=env.goal_reward,
            trap_reward=env.trap_reward,
            seed=None,
        )

    def _train_qlearning(
        self,
        template: GridWorld,
        *,
        seed: int,
        num_episodes: int,
        max_steps: int,
        agent_kwargs: dict[str, Any],
    ) -> _QLRun:
        """Run one Q-learning training pass and collect everything we analyze."""
        env = self._build_env(template, seed=seed)
        agent = QLearningAgent(seed=seed, **agent_kwargs)

        visit_counts = np.zeros((env.h, env.w), dtype=np.int64)
        t0 = time.perf_counter()
        for ev in agent.train_stream(env, num_episodes=num_episodes, max_steps=max_steps):
            r, c = ev.state
            visit_counts[r, c] += 1
        wall = time.perf_counter() - t0

        Q = agent._last_Q  # noqa: SLF001 - train_stream contract sets these
        metrics = agent._last_metrics  # noqa: SLF001
        state_to_idx = agent._last_state_to_idx  # noqa: SLF001
        idx_to_state = [s for s, _i in sorted(state_to_idx.items(), key=lambda kv: kv[1])]

        policy = QLearningAgent.get_policy(Q, idx_to_state)
        value_fn = QLearningAgent.get_value_function(Q, idx_to_state)

        # Greedy eval uses its own env RNG so stochastic rollouts don't touch
        # the training env's RNG (kept deterministic per seed for reproducibility).
        greedy_env = self._build_env(template, seed=seed + 10_000)
        greedy_traj, greedy_return = agent.evaluate_greedy(greedy_env, Q, state_to_idx, max_steps=max_steps)

        return _QLRun(
            episode_rewards=list(metrics.episode_rewards),
            episode_steps=list(metrics.episode_steps),
            epsilons=list(metrics.epsilons),
            Q=Q,
            state_to_idx=state_to_idx,
            policy=policy,
            value_fn=value_fn,
            greedy_return=float(greedy_return),
            greedy_trajectory=list(greedy_traj),
            visit_counts=visit_counts,
            wall_time_seconds=float(wall),
        )

    # ---------------------------------------------------------------------
    # Public entry point
    # ---------------------------------------------------------------------

    def run_all(
        self,
        env: GridWorld,
        *,
        config: ExperimentConfig = ExperimentConfig(),
        seed: int | None = None,
        seeds: list[int] | None = None,
        env_meta: dict[str, Any] | None = None,
        map_subdir: str = "default",
        compare_exploration_wind: bool = False,
    ) -> dict[str, Any]:
        """Run the full pipeline for one map and return its metrics dict.

        All artifacts are written under `<run_dir>/<map_subdir>/`. The caller
        is responsible for stitching multiple per-map dicts together into the
        top-level `metrics.json`.

        When ``compare_exploration_wind`` is True, each exploration strategy is
        trained under both deterministic dynamics and stochastic wind (same
        grid and ``wind_prob``), producing ``comparisons/exploration_det_vs_wind.png``.
        """
        # `seed` is kept for backwards compatibility (single-seed behavior).
        # `seeds` overrides it and is the preferred entry point.
        if seeds is None:
            seeds_list: list[int] = [seed] if seed is not None else list(_DEFAULT_SEEDS)
        else:
            seeds_list = list(seeds)
        if not seeds_list:
            raise ValueError("seeds must contain at least one entry")

        map_dir = self.run_dir / map_subdir
        vi_dir = map_dir / "vi"
        ql_dir = map_dir / "ql"
        cmp_dir = map_dir / "comparisons"
        analysis_dir = map_dir / "analysis"
        for d in (vi_dir, ql_dir, cmp_dir, analysis_dir):
            d.mkdir(parents=True, exist_ok=True)

        # 1) Value Iteration (deterministic given env + gamma + theta, runs once).
        vi = ValueIterationSolver(env, gamma=config.gamma, theta=config.theta)
        t0 = time.perf_counter()
        vi_res = vi.solve()
        vi_wall = time.perf_counter() - t0

        plot_value_map(vi_res.V, env, save_path=vi_dir / "value_map.png")
        plot_policy(vi_res.policy, env, save_path=vi_dir / "policy.png")
        plot_convergence(vi_res.deltas, save_path=vi_dir / "convergence.png")
        vi_traj, vi_return = vi.evaluate_policy(vi_res.policy, max_steps=config.max_steps)
        plot_trajectory(
            vi_traj,
            env,
            title=f"VI trajectory (return={vi_return:.3f})",
            save_path=vi_dir / "trajectory.png",
        )

        # VI rollout rewards across seeds: one rollout per seed so the
        # comparison against Q-learning has the same sample size.
        vi_rollout_per_seed: list[list[float]] = []
        # We still want a per-episode curve of VI returns to compare against the
        # Q-learning learning curve. Because VI is already optimal, we roll it
        # out once per episode index (it's cheap) with the same `episodes` count.
        for s in seeds_list:
            eval_env = self._build_env(env, seed=s + 50_000)
            eval_solver = ValueIterationSolver(eval_env, gamma=config.gamma, theta=config.theta)
            seed_curve: list[float] = []
            for _ in range(config.episodes):
                _t, ret = eval_solver.evaluate_policy(vi_res.policy, max_steps=config.max_steps)
                seed_curve.append(ret)
            vi_rollout_per_seed.append(seed_curve)
        vi_rollout_agg = aggregate_curves(vi_rollout_per_seed)

        # 2) Baseline Q-learning across all seeds (used for the main learning
        #    curve, VI-vs-QL comparison, convergence/stability metrics, and
        #    the aggregated visit heatmap).
        baseline_kwargs = dict(
            alpha=config.alpha,
            gamma=config.gamma,
            epsilon=config.epsilon,
            epsilon_decay=config.epsilon_decay,
            epsilon_min=config.epsilon_min,
        )
        baseline_runs: list[_QLRun] = []
        for s in seeds_list:
            baseline_runs.append(
                self._train_qlearning(
                    env,
                    seed=s,
                    num_episodes=config.episodes,
                    max_steps=config.max_steps,
                    agent_kwargs=baseline_kwargs,
                )
            )

        # Use the first seed's run for the single-instance Q-learning artifacts
        # (value map, policy, trajectory, learning curve). These are illustrative
        # figures, not the statistical story — the comparisons plots carry that.
        illustrative = baseline_runs[0]
        plot_value_map(
            illustrative.value_fn,
            env,
            title="Q-learning V(s)=max_a Q(s,a)",
            save_path=ql_dir / "value_map.png",
        )
        plot_policy(illustrative.policy, env, title="Q-learning greedy policy", save_path=ql_dir / "policy.png")
        plot_trajectory(
            illustrative.greedy_trajectory,
            env,
            title=f"Q-learning greedy trajectory (return={illustrative.greedy_return:.3f})",
            save_path=ql_dir / "trajectory.png",
        )
        plot_learning_curve(illustrative.episode_rewards, save_path=ql_dir / "learning_curve.png")

        # VI vs Q-learning: mean ± std curves on the same axes.
        baseline_agg = aggregate_curves([r.episode_rewards for r in baseline_runs])
        plot_multi_curve_bands(
            {
                "Value Iteration (rollouts)": vi_rollout_agg,
                "Q-learning (training)": baseline_agg,
            },
            title="Bellman vs Q-learning (mean ± std over seeds)",
            ylabel="Return",
            save_path=cmp_dir / "vi_vs_ql_rewards.png",
        )
        # Keep a single-seed overlay too: helps the eye when the bands overlap.
        plot_comparison(
            vi_rollout_per_seed[0],
            illustrative.episode_rewards[: len(vi_rollout_per_seed[0])],
            save_path=cmp_dir / "vi_vs_ql_rewards_single_seed.png",
        )

        # 3) Gamma comparison — multi-seed.
        gammas = [0.5, 0.9, 0.99]
        gamma_runs: dict[float, list[_QLRun]] = {g: [] for g in gammas}
        for g in gammas:
            for s in seeds_list:
                gamma_runs[g].append(
                    self._train_qlearning(
                        env,
                        seed=s,
                        num_episodes=config.episodes,
                        max_steps=config.max_steps,
                        agent_kwargs={**baseline_kwargs, "gamma": g},
                    )
                )
        gamma_bands = {
            f"γ={g}": aggregate_curves([r.episode_rewards for r in runs])
            for g, runs in gamma_runs.items()
        }
        plot_multi_curve_bands(
            gamma_bands,
            title="Q-learning: gamma comparison (mean ± std over seeds)",
            save_path=cmp_dir / "gamma_comparison.png",
        )

        # 4) Exploration strategies — three genuinely different strategies.
        strategy_specs: list[tuple[str, dict[str, Any]]] = [
            (
                "eps-greedy (decay)",
                {**baseline_kwargs, "epsilon": 1.0, "epsilon_decay": 0.995, "epsilon_min": 0.05, "strategy": "eps_greedy"},
            ),
            (
                "eps-greedy (fixed)",
                {**baseline_kwargs, "epsilon": 0.1, "epsilon_decay": 1.0, "epsilon_min": 0.1, "strategy": "eps_greedy"},
            ),
            (
                "softmax (decaying T)",
                {**baseline_kwargs, "epsilon": 1.0, "epsilon_decay": 0.995, "epsilon_min": 0.05, "strategy": "softmax"},
            ),
        ]

        def _run_strategy_sweep(template: GridWorld) -> dict[str, dict[str, np.ndarray]]:
            runs_map: dict[str, list[_QLRun]] = {label: [] for label, _ in strategy_specs}
            for label, kwargs in strategy_specs:
                for s in seeds_list:
                    runs_map[label].append(
                        self._train_qlearning(
                            template,
                            seed=s,
                            num_episodes=config.episodes,
                            max_steps=config.max_steps,
                            agent_kwargs=kwargs,
                        )
                    )
            return {label: aggregate_curves([r.episode_rewards for r in runs]) for label, runs in runs_map.items()}

        exploration_det_vs_wind_bands: dict[str, dict[str, np.ndarray]] | None = None
        if compare_exploration_wind:
            det_template = self._env_with_stochasticity(env, stochastic=False)
            wind_template = self._env_with_stochasticity(env, stochastic=True)
            strategy_bands_det = _run_strategy_sweep(det_template)
            strategy_bands_wind = _run_strategy_sweep(wind_template)
            strategy_bands = strategy_bands_wind if env.stochastic else strategy_bands_det
            exploration_det_vs_wind_bands = dict[str, dict[str, np.ndarray]]()
            for label in strategy_bands_det:
                exploration_det_vs_wind_bands[f"{label} (no wind)"] = strategy_bands_det[label]
                exploration_det_vs_wind_bands[f"{label} (stochastic wind)"] = strategy_bands_wind[label]
            plot_multi_curve_bands(
                exploration_det_vs_wind_bands,
                title=(
                    "Q-learning: exploration strategies — deterministic vs stochastic wind "
                    f"(p_intended={env.wind_prob:.2f}, mean ± std over seeds)"
                ),
                save_path=cmp_dir / "exploration_det_vs_wind.png",
                figsize=(10, 5),
            )
        else:
            strategy_bands = _run_strategy_sweep(env)

        plot_multi_curve_bands(
            strategy_bands,
            title="Q-learning: exploration strategies (mean ± std over seeds)",
            save_path=cmp_dir / "exploration_comparison.png",
        )

        # 5) Convergence & stability metrics (baseline config only — the
        #    comparison plots already show the per-sweep picture).
        convergence_threshold = 0.5
        per_seed_convergence: list[dict[str, Any]] = []
        for s, run in zip(seeds_list, baseline_runs):
            conv_ep = episodes_to_threshold(
                run.episode_rewards,
                threshold=convergence_threshold,
                window=20,
            )
            per_seed_convergence.append(
                {
                    "seed": int(s),
                    "episodes_to_threshold": None if conv_ep is None else int(conv_ep),
                    "policy_agreement_vs_vi": policy_agreement(run.policy, vi_res.policy),
                    "greedy_return": run.greedy_return,
                    "wall_time_seconds": run.wall_time_seconds,
                }
            )

        conv_values = [e["episodes_to_threshold"] for e in per_seed_convergence if e["episodes_to_threshold"] is not None]
        agreement_values = [e["policy_agreement_vs_vi"] for e in per_seed_convergence]
        greedy_values = [e["greedy_return"] for e in per_seed_convergence]
        wall_values = [e["wall_time_seconds"] for e in per_seed_convergence]

        convergence_summary: dict[str, Any] = {
            "threshold": convergence_threshold,
            "vi": {
                "iterations": len(vi_res.deltas),
                "wall_time_seconds": float(vi_wall),
                "greedy_rollout_return": float(vi_return),
            },
            "ql": {
                "per_seed": per_seed_convergence,
                "episodes_to_threshold_mean": float(np.mean(conv_values)) if conv_values else None,
                "episodes_to_threshold_std": float(np.std(conv_values, ddof=0)) if conv_values else None,
                "episodes_to_threshold_hit_rate": len(conv_values) / len(per_seed_convergence),
                "policy_agreement_mean": float(np.mean(agreement_values)),
                "policy_agreement_std": float(np.std(agreement_values, ddof=0)),
                "greedy_return_mean": float(np.mean(greedy_values)),
                "greedy_return_std": float(np.std(greedy_values, ddof=0)),
                "wall_time_mean_seconds": float(np.mean(wall_values)),
                "wall_time_std_seconds": float(np.std(wall_values, ddof=0)),
            },
        }

        # 6) Aggregated visit heatmap across seeds.
        visits_sum = np.sum([r.visit_counts for r in baseline_runs], axis=0)
        visits_mean = visits_sum / float(len(baseline_runs))
        plot_visit_heatmap(
            visits_sum,
            env,
            title=f"Q-learning state visits (sum over {len(baseline_runs)} seeds, log scale)",
            save_path=analysis_dir / "visit_heatmap_sum_log.png",
            log_scale=True,
        )
        plot_visit_heatmap(
            visits_mean,
            env,
            title=f"Q-learning state visits (mean over {len(baseline_runs)} seeds, log scale)",
            save_path=analysis_dir / "visit_heatmap_mean_log.png",
            log_scale=True,
        )

        # 7) Serialize config + metrics.
        def _bands_to_json(bands: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, list[float]]]:
            return {
                label: {
                    "mean": series["mean"].tolist(),
                    "std": series["std"].tolist(),
                    "n": int(series["n"]),
                }
                for label, series in bands.items()
            }

        metrics_result: dict[str, Any] = {
            "map_subdir": map_subdir,
            "env": env_meta or {},
            "seeds": seeds_list,
            "vi": {
                "final_return": float(vi_return),
                "deltas": vi_res.deltas,
                "rollout_rewards_per_seed": vi_rollout_per_seed,
                "iterations": len(vi_res.deltas),
                "wall_time_seconds": float(vi_wall),
            },
            "ql": {
                "baseline": {
                    "per_seed_rewards": [r.episode_rewards for r in baseline_runs],
                    "per_seed_steps": [r.episode_steps for r in baseline_runs],
                    "per_seed_epsilons": [r.epsilons for r in baseline_runs],
                    "per_seed_greedy_returns": [r.greedy_return for r in baseline_runs],
                },
                "aggregate": {
                    "mean": baseline_agg["mean"].tolist(),
                    "std": baseline_agg["std"].tolist(),
                    "n": int(baseline_agg["n"]),
                },
            },
            "gamma_comparison": _bands_to_json(gamma_bands),
            "exploration_comparison": _bands_to_json(strategy_bands),
            "convergence_and_stability": convergence_summary,
            "visit_counts_sum": visits_sum.tolist(),
        }
        if exploration_det_vs_wind_bands is not None:
            metrics_result["exploration_det_vs_wind"] = _bands_to_json(exploration_det_vs_wind_bands)
        return metrics_result


def summarize_across_maps(per_map_metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Build a compact, human-friendly cross-map comparison table.

    Given `{map_slug: <metrics dict from run_all>}`, return a summary that's
    cheap to eyeball in `metrics.json` (the bulky per-seed arrays stay nested
    inside each map). For every map we surface VI's optimal return and the
    headline Q-learning convergence/stability stats.
    """
    summary: dict[str, dict[str, Any]] = {}
    for slug, m in per_map_metrics.items():
        ql_summary = m["convergence_and_stability"]["ql"]
        vi_summary = m["convergence_and_stability"]["vi"]
        summary[slug] = {
            "env": m.get("env", {}),
            "seeds": m.get("seeds", []),
            "vi_iterations": vi_summary["iterations"],
            "vi_wall_time_seconds": vi_summary["wall_time_seconds"],
            "vi_greedy_return": vi_summary["greedy_rollout_return"],
            "ql_episodes_to_threshold_mean": ql_summary["episodes_to_threshold_mean"],
            "ql_episodes_to_threshold_std": ql_summary["episodes_to_threshold_std"],
            "ql_episodes_to_threshold_hit_rate": ql_summary["episodes_to_threshold_hit_rate"],
            "ql_policy_agreement_mean": ql_summary["policy_agreement_mean"],
            "ql_policy_agreement_std": ql_summary["policy_agreement_std"],
            "ql_greedy_return_mean": ql_summary["greedy_return_mean"],
            "ql_greedy_return_std": ql_summary["greedy_return_std"],
            "ql_wall_time_mean_seconds": ql_summary["wall_time_mean_seconds"],
        }
    return {"maps": summary, "n_maps": len(summary)}
