#!/usr/bin/env python3
"""Headless capture of arcade UI frames into docs/screenshots/.

Uses SDL dummy video (no display server). Run from repo root:

    ./.venv/bin/python scripts/capture_doc_screenshots.py

Requires the same dependencies as main_ui.py (pygame, numpy, imageio).
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

# Must be set before pygame is imported anywhere in the process.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DOCS_SHOTS = REPO_ROOT / "docs" / "screenshots"
os.environ.setdefault(
    "MPLCONFIGDIR", str((REPO_ROOT / "output" / ".mplconfig").resolve())
)


def _init_pygame() -> None:
    import pygame

    pygame.init()
    try:
        pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
    except pygame.error:
        pass


def _save_screen(screen, path: Path) -> None:
    import pygame

    path.parent.mkdir(parents=True, exist_ok=True)
    pygame.image.save(screen, str(path))


def _bootstrap_ctx(*, env, run_dir: Path):
    import pygame

    from src.ui import assets
    from src.ui.gif_export import GifRecorder
    from src.ui.renderer import Renderer, compute_layout
    from src.ui.scenes import Context

    layout = compute_layout(env)
    screen = pygame.display.set_mode(layout.screen_size)
    sprites = assets.load_sprites(layout.tile_px)
    fonts = assets.load_fonts()
    sounds = assets.load_sounds()
    renderer = Renderer(screen, layout, sprites, fonts)
    return Context(
        screen=screen,
        renderer=renderer,
        sprites=sprites,
        fonts=fonts,
        sounds=sounds,
        run_dir=run_dir,
        recorder=GifRecorder(fps=8),
    )


def main() -> None:
    import pygame

    from src.environment import GridWorld
    from src.q_learning import QLearningAgent
    from src.value_iteration import ValueIterationSolver
    from src.ui import assets
    from src.ui.scenes import (
        DEFAULT_MAPS,
        MenuScene,
        PlaybackScene,
        TrainScene,
        _refresh_renderer_for_env,
    )

    _init_pygame()

    if DOCS_SHOTS.exists():
        for p in DOCS_SHOTS.iterdir():
            if p.is_file():
                p.unlink()
    else:
        DOCS_SHOTS.mkdir(parents=True, exist_ok=True)

    tmp_run = Path(tempfile.mkdtemp(prefix="docshots_", dir=REPO_ROOT / "output"))

    try:
        # --- Menu (default layout matches cold start: 5x5 default env) ---
        base_env = GridWorld.default(seed=0)
        ctx = _bootstrap_ctx(env=base_env, run_dir=tmp_run)
        menu = MenuScene(ctx)
        menu.draw()
        _save_screen(ctx.screen, DOCS_SHOTS / "menu.png")

        # Menu: MAZE + WIZARD (shows map roster + character pick)
        menu.map_idx = 2
        menu.char_idx = 2
        name = assets.CHARACTER_ORDER[menu.char_idx]
        _refresh_renderer_for_env(ctx, base_env, agent_rc=assets.CHARACTERS[name])
        menu.draw()
        _save_screen(ctx.screen, DOCS_SHOTS / "menu_maze_wizard.png")

        # Menu: stochastic wind ON
        menu.stochastic = True
        menu.draw()
        _save_screen(ctx.screen, DOCS_SHOTS / "menu_stochastic_wind.png")

        # Menu: Value Iteration selected
        menu.stochastic = False
        menu.algo_idx = 1
        menu.draw()
        _save_screen(ctx.screen, DOCS_SHOTS / "menu_value_iteration.png")

        # --- Training: DEFAULT 5x5, mid-run ---
        choice_5 = DEFAULT_MAPS[0]
        env_5 = GridWorld.from_layout(
            choice_5.layout, stochastic=False, wind_prob=0.8, seed=0
        )
        _refresh_renderer_for_env(
            ctx, env_5, agent_rc=assets.CHARACTERS[assets.CHARACTER_ORDER[0]]
        )
        train = TrainScene(ctx, env=env_5, map_name=choice_5.name)
        train.speed_idx = 5
        while not train.done and train.current_episode < 35:
            train.update(16)
        train.speed_idx = 0
        train.paused = True
        train.draw()
        _save_screen(ctx.screen, DOCS_SHOTS / "train_default_5x5.png")

        # --- Training: MAZE 9x9 with overlays (matches README “overlays” story) ---
        choice_maze = DEFAULT_MAPS[2]
        env_maze = GridWorld.from_layout(
            choice_maze.layout, stochastic=False, wind_prob=0.8, seed=0
        )
        _refresh_renderer_for_env(
            ctx, env_maze, agent_rc=assets.CHARACTERS["KNIGHT"]
        )
        train_m = TrainScene(ctx, env=env_maze, map_name=choice_maze.name)
        train_m.speed_idx = 5
        while not train_m.done and train_m.current_episode < 120:
            train_m.update(16)
        train_m.speed_idx = 2
        train_m.paused = True
        train_m.overlay.value_map = True
        train_m.overlay.policy = True
        train_m.overlay.visits = True
        train_m.draw()
        _save_screen(ctx.screen, DOCS_SHOTS / "train_overlays.png")

        # --- Training: OPEN 7x7 (readable open field) ---
        choice_open = DEFAULT_MAPS[1]
        env_open = GridWorld.from_layout(
            choice_open.layout, stochastic=False, wind_prob=0.8, seed=0
        )
        _refresh_renderer_for_env(ctx, env_open, agent_rc=assets.CHARACTERS["VIKING"])
        train_o = TrainScene(ctx, env=env_open, map_name=choice_open.name)
        train_o.speed_idx = 5
        while not train_o.done and train_o.current_episode < 60:
            train_o.update(16)
        train_o.paused = True
        train_o.draw()
        _save_screen(ctx.screen, DOCS_SHOTS / "train_open_7x7.png")

        # --- Training: GAUNTLET ---
        choice_g = DEFAULT_MAPS[3]
        env_g = GridWorld.from_layout(
            choice_g.layout, stochastic=False, wind_prob=0.8, seed=0
        )
        _refresh_renderer_for_env(ctx, env_g, agent_rc=assets.CHARACTERS["RANGER"])
        train_g = TrainScene(ctx, env=env_g, map_name=choice_g.name)
        train_g.speed_idx = 5
        while not train_g.done and train_g.current_episode < 90:
            train_g.update(16)
        train_g.paused = True
        train_g.draw()
        _save_screen(ctx.screen, DOCS_SHOTS / "train_gauntlet.png")

        # --- Q-learning playback on maze (finish training once) ---
        _refresh_renderer_for_env(
            ctx, env_maze, agent_rc=assets.CHARACTERS["KNIGHT"]
        )
        train_full = TrainScene(ctx, env=env_maze, map_name=choice_maze.name)
        train_full.speed_idx = 5
        while not train_full.done:
            train_full.update(16)
        assert train_full.Q is not None and train_full.state_to_idx
        idx_to_state = [
            s for s, _ in sorted(train_full.state_to_idx.items(), key=lambda kv: kv[1])
        ]
        policy = QLearningAgent.get_policy(train_full.Q, idx_to_state)
        Vq = QLearningAgent.get_value_function(train_full.Q, idx_to_state)

        play = PlaybackScene(
            ctx,
            env=env_maze,
            policy=policy,
            V=Vq,
            title=f"Q-LEARNING GREEDY  -  {choice_maze.name}",
            auto_gif_path=None,
            show_value_by_default=True,
            record_gif=False,
        )
        for _ in range(600):
            play.update(16)
            play.draw()
            if play.steps >= 14:
                break
        if play.done:
            raise RuntimeError("Playback ended before mid-episode frame; check policy.")
        _save_screen(ctx.screen, DOCS_SHOTS / "playback_maze.png")

        while not play.done:
            play.update(16)
            play.draw()
        play.draw()
        _save_screen(ctx.screen, DOCS_SHOTS / "playback_maze_victory.png")

        # --- Episode GIF (recorded greedy maze solve) ---
        gif_run = tmp_run / "gifcap"
        gif_run.mkdir(exist_ok=True)
        play_gif = PlaybackScene(
            ctx,
            env=env_maze,
            policy=policy,
            V=Vq,
            title=f"Q-LEARNING GREEDY  -  {choice_maze.name}",
            auto_gif_path=gif_run / "maze_episode.gif",
            show_value_by_default=False,
            record_gif=True,
        )
        safety = 0
        while not play_gif.done and safety < 100_000:
            play_gif.update(16)
            play_gif.draw()
            safety += 1
        if not (gif_run / "maze_episode.gif").exists():
            raise RuntimeError("GIF was not written; playback may not have terminated.")
        shutil.copy(gif_run / "maze_episode.gif", DOCS_SHOTS / "maze_episode.gif")

        # --- Value Iteration → playback (default map) ---
        env_vi = GridWorld.from_layout(
            choice_5.layout, stochastic=False, wind_prob=0.8, seed=0
        )
        _refresh_renderer_for_env(
            ctx, env_vi, agent_rc=assets.CHARACTERS["WIZARD"]
        )
        vi_res = ValueIterationSolver(env_vi, gamma=0.99, theta=1e-6).solve()
        vi_play = PlaybackScene(
            ctx,
            env=env_vi,
            policy=vi_res.policy,
            V=vi_res.V,
            title=f"VALUE ITERATION  -  {choice_5.name}",
            auto_gif_path=None,
            show_value_by_default=True,
            record_gif=False,
        )
        for _ in range(400):
            vi_play.update(16)
            vi_play.draw()
            if vi_play.steps >= 6:
                break
        _save_screen(ctx.screen, DOCS_SHOTS / "playback_value_iteration.png")

        print(f"Wrote screenshots to {DOCS_SHOTS.resolve()}")
    finally:
        shutil.rmtree(tmp_run, ignore_errors=True)
        pygame.quit()


if __name__ == "__main__":
    main()
