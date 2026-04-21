# Experiment analysis — what each figure and metric means

This note documents the richer experimental analysis added for topic 3 of
`ROADMAP.md`. It's meant as a scratch reference the final technical report can
cite — not the report itself. All figures live under
`output/runs/<timestamp>_<slug>/` and are regenerated automatically by
`main.py`.

## How to reproduce

Single map (the default 5x5):

```bash
./.venv/bin/python main.py \
    --episodes 500 \
    --seeds 0 1 2 3 4 \
    --run-name multiseed
```

Sweep across every built-in map (default, open, maze, gauntlet) in one run:

```bash
./.venv/bin/python main.py \
    --episodes 500 \
    --seeds 0 1 2 3 4 \
    --maps all \
    --run-name allmaps
```

`--maps` accepts any combination of `default`, `open`, `maze`, `gauntlet`, or
the convenience keyword `all`. Each map gets its own subdirectory inside the
run folder so artifacts never collide. Other axes still come from the old
flags (`--stochastic`, `--wind-prob`, `--gamma`, `--alpha`, ...). The
`--seeds` flag overrides `--seed` when both are given, and defaults to
`0 1 2 3 4` when neither is set. All curves below are averaged across that
seed set, on a per-map basis.

## Files produced

```
output/runs/<timestamp>_<run_name>/
    config.yaml
    metrics.json                                   (cross-map summary + per-map dicts)
    <map_slug>/                                    (one per --maps entry, e.g. default_5x5/)
        vi/
            value_map.png
            policy.png
            trajectory.png
            convergence.png
        ql/
            value_map.png                (single-seed illustrative figure)
            policy.png                   (single-seed illustrative figure)
            trajectory.png               (single-seed illustrative figure)
            learning_curve.png           (single-seed illustrative figure)
        comparisons/
            vi_vs_ql_rewards.png                 (mean ± std over seeds)
            vi_vs_ql_rewards_single_seed.png     (sanity overlay, one seed)
            gamma_comparison.png                 (mean ± std over seeds)
            exploration_comparison.png           (mean ± std over seeds)
        analysis/
            visit_heatmap_sum_log.png    (summed across seeds, log scale)
            visit_heatmap_mean_log.png   (per-seed mean, log scale)
```

## What each plot is telling us

### `comparisons/vi_vs_ql_rewards.png`

Shows the per-episode return for Q-learning (noisy, exploring) against
Value-Iteration greedy rollouts (optimal by construction on a known MDP) with
a ±1 std band. The VI band is tight — it's always rolling out the same
optimal policy, so variance only comes from the stochastic wind. The
Q-learning band is wide early and narrows as the policy stabilizes. This is
the figure for the report's "Bellman vs Q-learning" requirement.

### `comparisons/gamma_comparison.png`

Q-learning trained with γ ∈ {0.5, 0.9, 0.99} across all seeds. Small γ makes
the agent myopic (it barely values the distant goal; returns stay
dominated by step costs), large γ recovers the full goal reward. Bands keep
the comparison honest when individual seeds are noisy.

### `comparisons/exploration_comparison.png`

Three genuinely different exploration schemes, not three tunings of one:

- **ε-greedy (decay)** — ε=1.0 → 0.05, decay 0.995.
- **ε-greedy (fixed)** — ε held at 0.1.
- **softmax (decaying T)** — Boltzmann sampling with temperature decay
  reusing the `epsilon` schedule.

The decay variant typically converges highest; the fixed-ε variant plateaus
below because it never commits to the greedy action; softmax sits in between
depending on temperature schedule. This covers the "exploration strategies"
requirement in section 4 of `project_description.md`.

### `analysis/visit_heatmap_{sum,mean}_log.png`

State-visit counts accumulated during training, summed across seeds (`sum`)
and per-seed-averaged (`mean`), on a log-1p scale. Answers two questions for
the report's discussion section: which cells are consistently explored vs
chronically ignored, and whether the policy found a narrow corridor or
diffuses across the grid.

### `ql/*` illustrative plots

These still exist but are drawn from seed 0 only. They're there to give the
reader a concrete value map, policy, and trajectory — the statistical story
is in the `comparisons/` plots and in `metrics.json`.

## Metrics stored in `metrics.json`

The new `convergence_and_stability` block answers the
"convergence time" and "stability of the learned policy" bullets in
section 4:

```json
{
  "convergence_and_stability": {
    "threshold": 0.5,
    "vi": {"iterations": ..., "wall_time_seconds": ..., "greedy_rollout_return": ...},
    "ql": {
      "per_seed": [{"seed": 0, "episodes_to_threshold": ..., "policy_agreement_vs_vi": ..., "greedy_return": ..., "wall_time_seconds": ...}, ...],
      "episodes_to_threshold_mean": ...,
      "episodes_to_threshold_std": ...,
      "episodes_to_threshold_hit_rate": ...,
      "policy_agreement_mean": ...,
      "policy_agreement_std": ...,
      "greedy_return_mean": ...,
      "greedy_return_std": ...,
      "wall_time_mean_seconds": ...,
      "wall_time_std_seconds": ...
    }
  }
}
```

Definitions:

- **episodes_to_threshold**: first episode where the 20-episode trailing mean
  of per-episode return reaches 0.5 (well above random but below the ≈0.7
  optimal return on DEFAULT 5×5). Lower = faster convergence. `null` means
  the run never hit the threshold.
- **policy_agreement_vs_vi**: fraction of non-terminal states where the
  greedy Q-learning policy picks the same action as VI's optimal policy.
  1.0 means the learned policy is indistinguishable from the planner.
- **greedy_return**: one greedy rollout per seed with the trained Q-table.
  Mean ± std across seeds is the "stability of the learned policy" metric.
- **wall_time_seconds**: measured with `time.perf_counter()` around
  `train_stream` (Q-learning) and `solve()` (VI). Gives a concrete answer to
  "planning is fast but needs the model; learning is slow but model-free".

## Cross-map summary in `metrics.json`

When the run sweeps multiple maps, `metrics.json` has the shape:

```json
{
  "cross_map_summary": {
    "n_maps": 4,
    "maps": {
      "default_5x5": {
        "env": {"layout_name": "DEFAULT 5x5", "shape": [5, 5], ...},
        "vi_iterations": ...,
        "vi_greedy_return": ...,
        "ql_episodes_to_threshold_mean": ...,
        "ql_policy_agreement_mean": ...,
        "ql_greedy_return_mean": ...,
        "ql_greedy_return_std": ...,
        "ql_wall_time_mean_seconds": ...
      },
      "open_7x7":  { ... },
      "maze_9x9":  { ... },
      "gauntlet_6x10": { ... }
    }
  },
  "per_map": {
    "default_5x5": { /* full metrics dict for that map: vi, ql.baseline, gamma_comparison, ... */ },
    ...
  }
}
```

`cross_map_summary.maps[<slug>]` is the at-a-glance table for the report's
discussion of how map topology affects convergence and policy quality.
`per_map[<slug>]` keeps the same dense data the single-map runs used to
write at the top level, so notebooks/scripts that already consumed that
schema only need to read one extra key.

## Explicit non-goals

See `ROADMAP.md` topic 3 for ideas we consciously left out of this iteration:

- No significance tests (Wilcoxon, bootstrap) — n=5 is too small to be worth
  the extra machinery.
- No ablation harness — the sweep axes are hard-coded.
- No new algorithms (SARSA, Double Q-learning, Policy Iteration) — those
  belong to topic 2 and would inflate scope without helping section 4.
