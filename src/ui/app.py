"""Pygame app bootstrap for the gridworld-rl arcade UI."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from ..environment import GridWorld
from ..run_dir import create_run_dir, update_latest_symlink
from . import assets
from .gif_export import GifRecorder
from .renderer import Renderer, compute_layout
from .scenes import Context, MenuScene, Scene, SceneTransition, TrainScene, _refresh_renderer_for_env


def run_arcade(
    *,
    output_dir: Path,
    run_name: str = "ui",
    initial_env: Optional[GridWorld] = None,
    window_title: str = "Gridworld RL - Arcade",
) -> Path:
    """Start the arcade UI event loop. Returns the run directory used."""
    # Quiet the macOS Apple Silicon SDL warning about fullscreen.
    os.environ.setdefault("SDL_VIDEO_CENTERED", "1")

    import pygame

    pygame.init()
    pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
    pygame.display.set_caption(window_title)

    env = initial_env or GridWorld.default(seed=0)
    layout = compute_layout(env)
    screen = pygame.display.set_mode(layout.screen_size)
    sprites = assets.load_sprites(layout.tile_px)
    fonts = assets.load_fonts()
    sounds = assets.load_sounds()

    run_dir = create_run_dir(output_dir, slug=run_name)
    renderer = Renderer(screen, layout, sprites, fonts)
    ctx = Context(
        screen=screen,
        renderer=renderer,
        sprites=sprites,
        fonts=fonts,
        sounds=sounds,
        run_dir=run_dir,
        recorder=GifRecorder(fps=8),
    )

    # Start with a menu. Callers can preset a TrainScene by passing an env
    # and slug, but that's a future nice-to-have.
    scene: Scene = MenuScene(ctx)

    clock = pygame.time.Clock()
    running = True
    while running:
        dt = clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            transition = scene.handle_event(event)
            if transition is not None:
                if transition.next_scene is None:
                    running = False
                    break
                scene = transition.next_scene(ctx)

        if not running:
            break

        transition = scene.update(dt)
        if transition is not None:
            if transition.next_scene is None:
                running = False
                break
            scene = transition.next_scene(ctx)
            continue

        scene.draw()
        pygame.display.flip()

    # Save any pending recording on exit.
    if ctx.recorder.is_recording or ctx.recorder.frame_count > 0:
        path = ctx.recorder.save(ctx.run_dir / "session.gif")
        if path is not None:
            print(f"[UI] Saved session GIF: {path}")

    update_latest_symlink(run_dir)
    pygame.quit()
    print(f"[UI] Run directory: {run_dir.resolve()}")
    return run_dir
