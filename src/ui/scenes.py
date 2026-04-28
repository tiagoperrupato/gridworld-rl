"""Arcade scenes: Menu, Train, Playback.

Each scene exposes a minimal interface:
    scene.handle_event(event) -> Optional[SceneTransition]
    scene.update(dt) -> Optional[SceneTransition]
    scene.draw()

The App runs the scene loop, swapping scenes when `SceneTransition` is
returned. Scenes share assets/fonts/sounds/renderer through the `Context`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from ..environment import _ACTION_TO_DELTA, Action, Cell, GridWorld
from ..maps import DEFAULT_MAPS, MapChoice
from ..q_learning import BaseQLearningAgent, QLearningAgent, DoubleQLearningAgent, StepEvent
from ..value_iteration import ValueIterationSolver
from . import assets
from .gif_export import GifRecorder
from .renderer import OverlayState, Renderer, compute_layout

__all__ = ["DEFAULT_MAPS", "MapChoice"]

BUMP_FRAMES = 6  # how many draws a wall-bump wiggle lasts

# Speed ladder (steps per rendered frame). "turbo" skips rendering inside
# long bursts to finish training quickly.
SPEED_LADDER: list[tuple[str, int, bool]] = [
    ("1x", 1, True),
    ("2x", 2, True),
    ("4x", 4, True),
    ("8x", 8, True),
    ("16x", 16, True),
    ("TURBO", 200, False),
]


@dataclass
class TrainConfig:
    episodes: int = 500
    max_steps: int = 200
    alpha: float = 0.1
    gamma: float = 0.99
    epsilon: float = 1.0
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.05
    seed: int | None = 0


@dataclass
class Context:
    screen: "any"  # pygame.Surface
    renderer: Renderer
    sprites: assets.Sprites
    fonts: assets.Fonts
    sounds: assets.Sounds
    run_dir: Path
    recorder: GifRecorder = field(default_factory=GifRecorder)


@dataclass
class SceneTransition:
    next_scene: Optional[Callable[[Context], "Scene"]]


class Scene:
    def __init__(self, ctx: Context) -> None:
        self.ctx = ctx

    def handle_event(self, event) -> Optional[SceneTransition]:  # noqa: ANN001 - pygame.Event
        return None

    def update(self, dt_ms: int) -> Optional[SceneTransition]:
        return None

    def draw(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Menu
# ---------------------------------------------------------------------------


class MenuScene(Scene):
    """Arcade title screen: pick algorithm + map + stochastic toggle."""

    ALGOS: list[tuple[str, str]] = [
        ("Q-LEARNING", "ql"),
        ("DOUBLE-Q-LEARNING", "dql"),
        ("VALUE ITERATION", "vi"),
    ]

    # Menu rows. Each index is a focusable line.
    _ALGO = 0
    _MAP = 1
    _WIND = 2
    _CHAR = 3
    _START = 4
    _NUM_ROWS = 5

    def __init__(self, ctx: Context) -> None:
        super().__init__(ctx)
        self.algo_idx = 0
        self.map_idx = 0
        self.stochastic = False
        self.char_idx = 0
        self.focus = 0
        self._frame = 0

    def handle_event(self, event) -> Optional[SceneTransition]:
        import pygame

        if event.type != pygame.KEYDOWN:
            return None

        key = event.key
        if key in (pygame.K_UP, pygame.K_w):
            self.focus = (self.focus - 1) % self._NUM_ROWS
            self.ctx.sounds.play("menu")
        elif key in (pygame.K_DOWN, pygame.K_s):
            self.focus = (self.focus + 1) % self._NUM_ROWS
            self.ctx.sounds.play("menu")
        elif key in (pygame.K_LEFT, pygame.K_a):
            self._adjust(-1)
        elif key in (pygame.K_RIGHT, pygame.K_d):
            self._adjust(1)
        elif key in (pygame.K_RETURN, pygame.K_SPACE, pygame.K_KP_ENTER):
            return self._confirm()
        elif key == pygame.K_ESCAPE:
            return SceneTransition(next_scene=None)
        return None

    def _adjust(self, delta: int) -> None:
        self.ctx.sounds.play("menu")
        if self.focus == self._ALGO:
            self.algo_idx = (self.algo_idx + delta) % len(self.ALGOS)
        elif self.focus == self._MAP:
            self.map_idx = (self.map_idx + delta) % len(DEFAULT_MAPS)
        elif self.focus == self._WIND:
            self.stochastic = not self.stochastic
        elif self.focus == self._CHAR:
            self.char_idx = (self.char_idx + delta) % len(assets.CHARACTER_ORDER)
            # Hot-swap the sprite band to preview the pick immediately.
            name = assets.CHARACTER_ORDER[self.char_idx]
            _reload_character_sprites(self.ctx, assets.CHARACTERS[name])

    def _confirm(self) -> Optional[SceneTransition]:
        self.ctx.sounds.play("select")
        choice = DEFAULT_MAPS[self.map_idx]
        env = GridWorld.from_layout(
            choice.layout,
            stochastic=self.stochastic,
            wind_prob=0.8,
            seed=0,
        )
        algo = self.ALGOS[self.algo_idx][1]
        char_name = assets.CHARACTER_ORDER[self.char_idx]
        char_rc = assets.CHARACTERS[char_name]

        # Recompute layout + sprites for the new grid dimensions AND keep
        # the picked character.
        _refresh_renderer_for_env(self.ctx, env, agent_rc=char_rc)

        if algo == "ql" or algo == "dql":
            def factory(ctx: Context) -> Scene:
                return TrainScene(
                    ctx, 
                    env=env, 
                    map_name=choice.name, 
                    AgentClass=QLearningAgent if algo == "ql" else DoubleQLearningAgent
                )

            return SceneTransition(next_scene=factory)
        else:
            def vi_factory(ctx: Context) -> Scene:
                return ValueIterationScene(ctx, env=env, map_name=choice.name)

            return SceneTransition(next_scene=vi_factory)

    def update(self, dt_ms: int) -> None:
        self._frame += 1
        return None

    def draw(self) -> None:
        import pygame

        self.ctx.screen.fill(assets.BG_DARK)
        w, h = self.ctx.screen.get_size()

        title = self.ctx.fonts.title.render("GRIDWORLD  RL", False, assets.FG_GOLD)
        sub = self.ctx.fonts.hud.render("ARCADE EDITION", False, assets.FG_WHITE)
        self.ctx.screen.blit(title, ((w - title.get_width()) // 2, 60))
        self.ctx.screen.blit(sub, ((w - sub.get_width()) // 2, 100))

        # Decorative sprite band (fixed position high above the menu rows).
        self._draw_sprite_band(y_top=140)

        char_name = assets.CHARACTER_ORDER[self.char_idx]
        rows = [
            ("ALGORITHM", self.ALGOS[self.algo_idx][0]),
            ("MAP      ", DEFAULT_MAPS[self.map_idx].name),
            ("WIND     ", "ON" if self.stochastic else "OFF"),
            ("CHARACTER", char_name),
            ("START    ", "PRESS ENTER"),
        ]

        y = h // 2 - 48
        for i, (label, value) in enumerate(rows):
            selected = i == self.focus
            color = assets.FG_GOLD if selected else assets.FG_WHITE
            prefix = "> " if selected else "  "
            is_toggle_row = i < self._START
            text = (
                f"{prefix}{label}  <  {value}  >"
                if is_toggle_row
                else f"{prefix}{value}"
            )
            surf = self.ctx.fonts.menu.render(text, False, color)
            x = (w - surf.get_width()) // 2
            self.ctx.screen.blit(surf, (x, y + i * 36))
            if selected and (self._frame // 15) % 2 == 0:
                pygame.draw.rect(
                    self.ctx.screen,
                    assets.FG_GOLD,
                    (x - 8, y + i * 36 - 2, surf.get_width() + 16, surf.get_height() + 4),
                    1,
                )

        hint = self.ctx.fonts.small.render(
            "ARROWS to change    ENTER to confirm    ESC to quit",
            False,
            assets.FG_BLUE,
        )
        self.ctx.screen.blit(hint, ((w - hint.get_width()) // 2, h - 40))

    def _draw_sprite_band(self, *, y_top: int) -> None:
        """Decorative centered row: agent walk frames + goal + trap.

        Uses a fixed icon size (independent of `tile_px`) so the band stays
        compact no matter how large the gameplay grid tiles become.
        """
        import pygame

        sprites = self.ctx.sprites
        w, _ = self.ctx.screen.get_size()
        icon_px = 64
        gap_frame = 6
        gap_group = 28

        items: list[pygame.Surface] = [
            *(pygame.transform.scale(f, (icon_px, icon_px)) for f in sprites.agent_frames),
            pygame.transform.scale(sprites.goal, (icon_px, icon_px)),
            pygame.transform.scale(sprites.trap, (icon_px, icon_px)),
        ]
        # Total width with extra spacing between agent band and goal/trap.
        n_agent = len(sprites.agent_frames)
        total_w = (
            n_agent * icon_px
            + (n_agent - 1) * gap_frame
            + gap_group
            + icon_px
            + gap_group
            + icon_px
        )
        x = (w - total_w) // 2
        for i, surf in enumerate(items):
            self.ctx.screen.blit(surf, (x, y_top))
            if i == n_agent - 1 or i == n_agent:
                x += icon_px + gap_group
            else:
                x += icon_px + gap_frame


# ---------------------------------------------------------------------------
# Training (Q-learning and Double Q-learning share the same scene since the only difference is the agent class).
# ---------------------------------------------------------------------------


class TrainScene(Scene):
    def __init__(self, ctx: Context, *, env: GridWorld, map_name: str, AgentClass: BaseQLearningAgent) -> None:
        super().__init__(ctx)
        self.env = env
        self.map_name = map_name
        self.cfg = TrainConfig()
        self.agent = AgentClass(
            alpha=self.cfg.alpha,
            gamma=self.cfg.gamma,
            epsilon=self.cfg.epsilon,
            epsilon_decay=self.cfg.epsilon_decay,
            epsilon_min=self.cfg.epsilon_min,
            seed=self.cfg.seed,
        )
        self.stream = self.agent.train_stream(
            env, num_episodes=self.cfg.episodes, max_steps=self.cfg.max_steps
        )
        self.speed_idx = 2  # 4x default
        self.paused = False
        self.single_step = False
        self.overlay = OverlayState()
        self.last_event: Optional[StepEvent] = None
        self.done = False
        self.visits = np.zeros((env.h, env.w), dtype=np.int64)
        self.visits[env.start_state] = 1
        self.frame_tick = 0
        self.walk_anim_idx = 0
        # For HUD we remember the most recent running values.
        self.current_state = env.start_state
        self.current_episode = 0
        self.current_steps = 0
        self.current_total_reward = 0.0
        self.current_epsilon = self.cfg.epsilon
        self.Q: Optional[np.ndarray] = None
        self.state_to_idx: dict[tuple[int, int], int] = {}
        self._bump_remaining = 0
        self._bump_action: Optional[Action] = None

    # -- Input --------------------------------------------------------------

    def handle_event(self, event) -> Optional[SceneTransition]:
        import pygame

        if event.type != pygame.KEYDOWN:
            return None

        key = event.key
        if key == pygame.K_ESCAPE:
            return SceneTransition(next_scene=_menu_factory)
        if key == pygame.K_SPACE:
            self.paused = not self.paused
            self.ctx.sounds.play("menu")
        elif key == pygame.K_RIGHT and self.paused:
            self.single_step = True
        elif key in (pygame.K_PLUS, pygame.K_EQUALS):
            self.speed_idx = min(len(SPEED_LADDER) - 1, self.speed_idx + 1)
            self.ctx.sounds.play("menu")
        elif key == pygame.K_MINUS:
            self.speed_idx = max(0, self.speed_idx - 1)
            self.ctx.sounds.play("menu")
        elif key == pygame.K_v:
            self.overlay.value_map = not self.overlay.value_map
            self.ctx.sounds.play("menu")
        elif key == pygame.K_p:
            self.overlay.policy = not self.overlay.policy
            self.ctx.sounds.play("menu")
        elif key == pygame.K_h:
            self.overlay.visits = not self.overlay.visits
            self.ctx.sounds.play("menu")
        elif key == pygame.K_r:
            self._toggle_recording()
        elif key == pygame.K_RETURN and self.done:
            return self._to_playback()
        return None

    def _toggle_recording(self) -> None:
        rec = self.ctx.recorder
        if rec.is_recording:
            rec.stop()
            path = rec.save(self.ctx.run_dir / "manual_recording.gif")
            if path is not None:
                print(f"[UI] Saved GIF: {path}")
        else:
            rec.start()
        self.ctx.sounds.play("select")

    # -- Step logic ---------------------------------------------------------

    def update(self, dt_ms: int) -> Optional[SceneTransition]:
        if self.done:
            return None

        _label, n_steps, should_render = SPEED_LADDER[self.speed_idx]

        if self.paused:
            if self.single_step:
                self._advance_one_step()
                self.single_step = False
            return None

        for _ in range(n_steps):
            if self.done:
                break
            self._advance_one_step()

        self.frame_tick += 1
        if self.frame_tick % 6 == 0:
            self.walk_anim_idx = (self.walk_anim_idx + 1) % len(self.ctx.sprites.agent_frames)
        return None

    def _advance_one_step(self) -> None:
        try:
            ev = next(self.stream)
        except StopIteration:
            self.done = True
            self._on_training_done()
            return

        self.last_event = ev
        self.current_state = ev.next_state
        self.current_episode = ev.episode + 1
        self.current_steps = ev.step
        self.current_total_reward = ev.episode_total_reward
        self.current_epsilon = ev.epsilon
        self.Q = ev.Q
        self.state_to_idx = ev.state_to_idx

        r, c = ev.next_state
        if 0 <= r < self.visits.shape[0] and 0 <= c < self.visits.shape[1]:
            self.visits[r, c] += 1

        # Detect wall/boundary bump: action did not change the state.
        if ev.state == ev.next_state and not ev.done:
            self._bump_remaining = BUMP_FRAMES
            self._bump_action = ev.action
            if self.speed_idx <= 1:
                self.ctx.sounds.play("menu", volume=0.3)

        if ev.done:
            cell = Cell(int(self.env.grid[r, c]))
            if cell == Cell.GOAL:
                self.ctx.sounds.play("goal", volume=0.6)
            elif cell == Cell.TRAP:
                self.ctx.sounds.play("trap", volume=0.6)
        elif self.speed_idx <= 1:
            self.ctx.sounds.play("step", volume=0.3)

    def _on_training_done(self) -> None:
        print("[UI] Training complete; press ENTER for playback.")

    def _current_bump_offset(self) -> tuple[int, int]:
        """Pixel offset to draw when the agent just bumped into a wall.

        Peaks toward the wall then bounces back; fades to zero over
        `BUMP_FRAMES` draws.
        """
        return _bump_offset(
            self._bump_remaining,
            self._bump_action,
            self.ctx.renderer.layout.tile_px,
        )

    def _to_playback(self) -> SceneTransition:
        assert self.Q is not None
        idx_to_state = [s for s, _ in sorted(self.state_to_idx.items(), key=lambda kv: kv[1])]
        policy = BaseQLearningAgent.get_policy(self.Q, idx_to_state)
        V = BaseQLearningAgent.get_value_function(self.Q, idx_to_state)
        label = "Q-LEARNING" if isinstance(self.agent, QLearningAgent) else "DOUBLE-Q-LEARNING"
        def factory(ctx: Context) -> Scene:
            return PlaybackScene(
                ctx,
                env=self.env,
                policy=policy,
                V=V,
                title=f"{label}  GREEDY -  {self.map_name}",
                auto_gif_path=self.ctx.run_dir / "episode.gif",
            )

        return SceneTransition(next_scene=factory)

    # -- Draw ---------------------------------------------------------------

    def draw(self) -> None:
        _label, _n, should_render = SPEED_LADDER[self.speed_idx]
        if not should_render and not self.paused and not self.done:
            # In TURBO we only redraw every ~Nth update to save time.
            if self.frame_tick % 15 != 0:
                return

        self.ctx.renderer.clear()
        self.ctx.renderer.draw_grid(self.env)

        if self.overlay.visits:
            self.ctx.renderer.draw_visits_overlay(self.visits, self.env)
        if self.overlay.value_map and self.Q is not None and self.state_to_idx:
            idx_to_state = [s for s, _ in sorted(self.state_to_idx.items(), key=lambda kv: kv[1])]
            V = BaseQLearningAgent.get_value_function(self.Q, idx_to_state)
            self.ctx.renderer.draw_value_overlay(V, self.env)
        if self.overlay.policy and self.Q is not None and self.state_to_idx:
            idx_to_state = [s for s, _ in sorted(self.state_to_idx.items(), key=lambda kv: kv[1])]
            policy = BaseQLearningAgent.get_policy(self.Q, idx_to_state)
            self.ctx.renderer.draw_policy_overlay(policy, self.env)

        bump_offset = self._current_bump_offset()
        self.ctx.renderer.draw_agent(
            self.current_state, self.walk_anim_idx, bump_offset=bump_offset
        )
        if self._bump_remaining > 0:
            self._bump_remaining -= 1

        speed_label = SPEED_LADDER[self.speed_idx][0]
        if self.paused:
            speed_label = "PAUSE"
        label = "Q-LEARNING" if isinstance(self.agent, QLearningAgent) else "DOUBLE-Q-LEARNING"
        title = f"{label}  -  {self.map_name}"
        if self.done:
            title += "   [DONE - ENTER for playback]"

        self.ctx.renderer.draw_hud(
            title=title,
            episode=self.current_episode,
            total_episodes=self.cfg.episodes,
            score=self.current_total_reward,
            steps=self.current_steps,
            epsilon=self.current_epsilon,
            speed_label=speed_label,
            recording=self.ctx.recorder.is_recording,
        )
        self.ctx.renderer.draw_footer(
            [
                "SPACE=pause",
                "RIGHT=step",
                "+/-=speed",
                "V=value",
                "P=policy",
                "H=visits",
                "R=record",
                "ESC=menu",
            ]
        )

        self.ctx.recorder.capture(self.ctx.screen)


# ---------------------------------------------------------------------------
# Value Iteration scene — solve headless, then go straight to playback.
# ---------------------------------------------------------------------------


class ValueIterationScene(Scene):
    def __init__(self, ctx: Context, *, env: GridWorld, map_name: str) -> None:
        super().__init__(ctx)
        self.env = env
        self.map_name = map_name
        self.solver = ValueIterationSolver(env, gamma=0.99, theta=1e-6)
        self.result = None
        self.progress_msg = "PLANNING..."

    def update(self, dt_ms: int) -> Optional[SceneTransition]:
        if self.result is None:
            self.result = self.solver.solve()

            def factory(ctx: Context) -> Scene:
                return PlaybackScene(
                    ctx,
                    env=self.env,
                    policy=self.result.policy,
                    V=self.result.V,
                    title=f"VALUE ITERATION  -  {self.map_name}",
                    auto_gif_path=self.ctx.run_dir / "episode.gif",
                    show_value_by_default=True,
                )

            return SceneTransition(next_scene=factory)
        return None

    def draw(self) -> None:
        self.ctx.screen.fill(assets.BG_DARK)
        w, h = self.ctx.screen.get_size()
        surf = self.ctx.fonts.title.render(self.progress_msg, False, assets.FG_GOLD)
        self.ctx.screen.blit(surf, ((w - surf.get_width()) // 2, h // 2 - 16))


# ---------------------------------------------------------------------------
# Playback — replay the greedy episode under a fixed policy, record GIF.
# ---------------------------------------------------------------------------


class PlaybackScene(Scene):
    def __init__(
        self,
        ctx: Context,
        *,
        env: GridWorld,
        policy: dict[tuple[int, int], Action],
        V: dict[tuple[int, int], float],
        title: str,
        auto_gif_path: Path | None = None,
        show_value_by_default: bool = False,
        record_gif: bool = True,
    ) -> None:
        super().__init__(ctx)
        self.env = env
        self.policy = policy
        self.V = V
        self.title = title
        self.state = env.reset()
        self.trajectory = [self.state]
        self.total_reward = 0.0
        self.steps = 0
        self.max_steps = 300
        self.done = False
        self.overlay = OverlayState(value_map=show_value_by_default, policy=True)
        self.step_every_frames = 6
        self._frame = 0
        self.walk_anim_idx = 0
        self.auto_gif_path = auto_gif_path
        self.record_gif = record_gif
        self._bump_remaining = 0
        self._bump_action: Optional[Action] = None
        # Auto-record this playback so we get a nice episode.gif.
        if record_gif:
            ctx.recorder.start()

    def handle_event(self, event) -> Optional[SceneTransition]:
        import pygame

        if event.type != pygame.KEYDOWN:
            return None
        if event.key == pygame.K_ESCAPE or event.key == pygame.K_RETURN:
            return self._finish()
        if event.key == pygame.K_SPACE and self.done:
            self._replay()
            return None
        if event.key == pygame.K_v:
            self.overlay.value_map = not self.overlay.value_map
        elif event.key == pygame.K_p:
            self.overlay.policy = not self.overlay.policy
        elif event.key == pygame.K_r:
            rec = self.ctx.recorder
            if rec.is_recording:
                rec.stop()
                path = rec.save(self.ctx.run_dir / "manual_recording.gif")
                if path is not None:
                    print(f"[UI] Saved GIF: {path}")
            else:
                rec.start()
        return None

    def _replay(self) -> None:
        """Re-run the same greedy episode without returning to the menu."""
        self.state = self.env.reset()
        self.trajectory = [self.state]
        self.total_reward = 0.0
        self.steps = 0
        self.done = False
        self._frame = 0
        self._bump_remaining = 0
        self._bump_action = None
        self.ctx.sounds.play("select")

    def update(self, dt_ms: int) -> Optional[SceneTransition]:
        self._frame += 1
        if self._frame % 6 == 0:
            self.walk_anim_idx = (self.walk_anim_idx + 1) % len(self.ctx.sprites.agent_frames)

        if self.done:
            return None

        if self._frame % self.step_every_frames != 0:
            return None

        if self.env.is_terminal(self.state) or self.steps >= self.max_steps:
            self._on_done()
            return None

        a = self.policy.get(self.state)
        if a is None:
            self._on_done()
            return None
        self.env.set_state(self.state)
        prev_state = self.state
        s2, r, done = self.env.step(a)
        self.state = s2
        self.trajectory.append(s2)
        self.total_reward += r
        self.steps += 1
        if prev_state == s2 and not done:
            self._bump_remaining = BUMP_FRAMES
            self._bump_action = a
            self.ctx.sounds.play("menu", volume=0.3)
        if done:
            cell = Cell(int(self.env.grid[s2[0], s2[1]]))
            if cell == Cell.GOAL:
                self.ctx.sounds.play("goal", volume=0.7)
            elif cell == Cell.TRAP:
                self.ctx.sounds.play("trap", volume=0.7)
            self._on_done()
        else:
            self.ctx.sounds.play("step", volume=0.2)
        return None

    def _on_done(self) -> None:
        if self.done:
            return
        self.done = True
        rec = self.ctx.recorder
        if self.record_gif:
            # Capture a few final frames so the GIF shows the end state clearly.
            for _ in range(8):
                rec.capture(self.ctx.screen)
            if self.auto_gif_path is not None:
                path = rec.save(self.auto_gif_path)
                if path is not None:
                    print(f"[UI] Saved episode GIF: {path}")

    def _finish(self) -> SceneTransition:
        # If recorder still has unsaved frames, drop them (user asked to leave).
        self.ctx.recorder.stop()
        return SceneTransition(next_scene=_menu_factory)

    def draw(self) -> None:
        self.ctx.renderer.clear()
        self.ctx.renderer.draw_grid(self.env)
        if self.overlay.value_map:
            self.ctx.renderer.draw_value_overlay(self.V, self.env)
        if self.overlay.policy:
            self.ctx.renderer.draw_policy_overlay(self.policy, self.env)

        # Draw trajectory as a faint breadcrumb trail.
        import pygame

        tpx = self.ctx.renderer.layout.tile_px
        ox, oy = self.ctx.renderer.layout.grid_origin
        pts = [
            (ox + c * tpx + tpx // 2, oy + r * tpx + tpx // 2)
            for r, c in self.trajectory
        ]
        if len(pts) >= 2:
            pygame.draw.lines(self.ctx.screen, assets.FG_GOLD, False, pts, 2)

        bump_offset = _bump_offset(
            self._bump_remaining,
            self._bump_action,
            self.ctx.renderer.layout.tile_px,
        )
        self.ctx.renderer.draw_agent(
            self.state, self.walk_anim_idx, bump_offset=bump_offset
        )
        if self._bump_remaining > 0:
            self._bump_remaining -= 1

        self.ctx.renderer.draw_hud(
            title=self.title,
            episode=1,
            total_episodes=1,
            score=self.total_reward,
            steps=self.steps,
            epsilon=0.0,
            speed_label="GREEDY",
            recording=self.ctx.recorder.is_recording,
        )
        if self.done:
            self._draw_end_banner()
            self.ctx.renderer.draw_footer(
                ["SPACE=replay", "ENTER=menu", "ESC=menu", "V=value", "P=policy"]
            )
        else:
            self.ctx.renderer.draw_footer(
                ["V=value", "P=policy", "R=toggle record", "ENTER/ESC=menu"]
            )
        if self.record_gif:
            self.ctx.recorder.capture(self.ctx.screen)

    def _draw_end_banner(self) -> None:
        """Render a centered arcade-style banner after the episode ends."""
        import pygame

        cell = Cell(int(self.env.grid[self.state[0], self.state[1]]))
        if cell == Cell.GOAL:
            title = "VICTORY!"
            color = assets.FG_GOLD
        elif cell == Cell.TRAP:
            title = "GAME OVER"
            color = assets.FG_RED
        else:
            title = "EPISODE END"
            color = assets.FG_WHITE

        screen = self.ctx.screen
        layout = self.ctx.renderer.layout
        big = self.ctx.fonts.title
        small = self.ctx.fonts.small

        title_surf = big.render(title, True, color)
        score_surf = small.render(
            f"SCORE {self.total_reward:+.2f}   STEPS {self.steps}", True, assets.FG_WHITE
        )
        hint_surf = small.render(
            "SPACE: REPLAY    ENTER: MENU", True, assets.FG_BLUE
        )

        padding = 18
        box_w = max(
            title_surf.get_width(), score_surf.get_width(), hint_surf.get_width()
        ) + padding * 2
        box_h = (
            title_surf.get_height()
            + score_surf.get_height()
            + hint_surf.get_height()
            + padding * 3
        )
        gx, gy = layout.grid_origin
        grid_w, grid_h = layout.grid_px
        box_x = gx + (grid_w - box_w) // 2
        box_y = gy + (grid_h - box_h) // 2

        overlay = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 210))
        pygame.draw.rect(overlay, color, overlay.get_rect(), width=2)
        screen.blit(overlay, (box_x, box_y))

        cursor_y = box_y + padding
        screen.blit(title_surf, (box_x + (box_w - title_surf.get_width()) // 2, cursor_y))
        cursor_y += title_surf.get_height() + padding
        screen.blit(
            score_surf, (box_x + (box_w - score_surf.get_width()) // 2, cursor_y)
        )
        cursor_y += score_surf.get_height() + padding
        screen.blit(hint_surf, (box_x + (box_w - hint_surf.get_width()) // 2, cursor_y))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _menu_factory(ctx: Context) -> Scene:
    return MenuScene(ctx)


def _bump_offset(
    remaining: int, action: Optional[Action], tile_px: int
) -> tuple[int, int]:
    """Compute the pixel offset for a wall-bump wiggle.

    Returns an offset that peaks toward the blocked direction then decays.
    Returns (0, 0) when no bump is active.
    """
    if remaining <= 0 or action is None:
        return (0, 0)
    dr, dc = _ACTION_TO_DELTA[action]
    # Triangle envelope: peaks at mid-animation, fades at start and end.
    progress = 1.0 - (remaining / BUMP_FRAMES)
    amplitude = int(tile_px * 0.22)
    if progress < 0.5:
        factor = progress * 2.0
    else:
        factor = (1.0 - progress) * 2.0
    offset_px = int(amplitude * factor)
    return (dc * offset_px, dr * offset_px)


def _refresh_renderer_for_env(
    ctx: Context, env: GridWorld, *, agent_rc: tuple[int, int] | None = None
) -> None:
    """When a new map is chosen, recompute layout + resize screen + reload sprites.

    Pass `agent_rc` to also swap the character sprite (defaults to the
    loader's built-in ranger when omitted).
    """
    import pygame

    layout = compute_layout(env)
    if ctx.screen.get_size() != layout.screen_size:
        ctx.screen = pygame.display.set_mode(layout.screen_size)
    if agent_rc is None:
        sprites = assets.load_sprites(layout.tile_px)
    else:
        sprites = assets.load_sprites(layout.tile_px, agent_rc=agent_rc)
    ctx.sprites = sprites
    ctx.renderer.screen = ctx.screen
    ctx.renderer.update_layout(layout, sprites)


def _reload_character_sprites(ctx: Context, agent_rc: tuple[int, int]) -> None:
    """Hot-swap just the agent character without touching layout/screen size.

    Used in the menu so the decorative sprite band updates live as the user
    cycles through the CHARACTER option.
    """
    sprites = assets.load_sprites(ctx.sprites.tile_px, agent_rc=agent_rc)
    ctx.sprites = sprites
    ctx.renderer.sprites = sprites
