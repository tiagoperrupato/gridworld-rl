# Roadmap

This file tracks ideas we've scoped for **future iterations** of the gridworld-rl project. The goal is to keep a visible record of where the project can go after the current milestone (arcade UI + structured output) so nothing gets lost between sprints.

Items are grouped by theme and ordered roughly by how much payoff they give vs. effort. Nothing here is committed — feel free to reprioritize, drop, or add items in PRs.

---

## 1. Maps and environments

- **Procedural maze generator** (DFS / Prim's / Wilson's) with configurable size, branching, and trap density. Expose as `GridWorld.random(size=(h, w), seed=..., trap_fraction=...)`.
- **Bigger worlds** (20×20, 40×40) to test scalability of tabular methods and start motivating function approximation.
- **Multiple goals** (reach any) and **ordered goals** (collect key → open door → reach exit).
- **Doors + keys + switches** as new cell types, extending the state space.
- **Directional wind zones** (per-cell stochastic bias) to break symmetry of the current uniform wind.
- **Moving hazards** (enemies on patrol) — agent must reason about adversarial co-agents.

## 2. Additional algorithms

All of these can reuse the `train_stream` pattern to plug into the arcade UI:

- **SARSA** and **Expected SARSA** — contrast on-policy vs off-policy behavior on the same map.
- **Double Q-learning** — reduce maximization bias; visualize the two Q-tables side by side.
- **Policy Iteration** — the other Bellman-family planner; show faster convergence on some maps.
- **Monte Carlo (first-visit / every-visit)** — episodic-only baseline.
- **n-step TD / Eligibility traces (TD(λ))** — bridge MC and TD.
- **Function approximation** (linear features or a tiny MLP) — path toward DQN without leaving the codebase.

## 3. Richer experimental analysis

- **Multi-seed runs with confidence bands**: run each config across N seeds, plot mean ± std on learning curves.
- **Aggregated visit heatmaps** across seeds — see which states are consistently explored.
- **Paired significance tests** (Wilcoxon / bootstrap) between configs, reported in `metrics.json`.
- **Wall-clock vs sample-efficiency plots** — planning is fast but needs the model; learning is slow but model-free. Make the tradeoff explicit.
- **Ablation harness**: config-driven sweeps over α, γ, ε, decay, step cost, trap reward.

## 4. Automated reports

- **HTML report**: one page per run dir, embedding all figures + metrics tables. Generate from `metrics.json` + `config.yaml` using Jinja2.
- **PDF export** (WeasyPrint or `matplotlib` → PDF). Drop the result directly into the final technical report.
- **Markdown digest** (`REPORT.md` per run) so diffs between runs are reviewable in GitHub.

## 5. Developer experience

- **YAML configs** as first-class: `python main.py --config configs/maze_hard.yaml`. Current CLI still works but YAML keeps reproducibility friendlier.
- **Structured logging** (`logging` module) instead of bare `print`, with levels.
- **Progress bars** via `tqdm` on the headless runner.
- **Subcommands**: `gw run`, `gw play`, `gw report` via a `click` or `argparse` subparser entry point.
- **Unit tests** for Bellman updates, transition enumeration under wind, and ε-greedy correctness.
- **Type checking** (`mypy` or `pyright`) in CI.

## 6. UI follow-ups

- **Human-playable mode**: arrow keys to move the agent yourself, compare your cumulative reward with the learned policy's.
- **Side-by-side VI vs Q-learning playback** (two grids in one window, same episode).
- **Animated value-function "blooming"**: watch V(s) propagate out from the goal over VI iterations.
- **Per-step Q-value inspector** when paused — show the four `Q(s, a)` values for the current cell.
- **Minimap + fog of war** for big maps.
- **Custom-map editor**: drop walls, traps, goals with the mouse, save to `maps/custom/*.txt`.
- **Fullscreen toggle + window scaling** (current layout assumes 1024×768-ish).
- **Particle effects** (sparkles when hitting the goal, puff when hitting a trap) for arcade juice.

## 7. Web / remote demo (alternative to desktop UI)

- **Streamlit or Gradio dashboard** for a zero-install demo that runs in the browser, with sliders for hyperparameters and live plots. Would complement the Pygame arcade UI, not replace it.
- **Export the trained policy + environment as JSON** so a static HTML/JS viewer (no Python) can replay it.

## 8. Curriculum and transfer

- **Curriculum learning**: train on a sequence of progressively harder maps, reuse Q between tasks, measure speedup.
- **Transfer evaluation**: train on map A, test greedy on map B, quantify zero-shot quality.
- **Shaped rewards**: plug in potential-based shaping and show it preserves optimal policy while speeding convergence.

