"""Rendering helpers for the arcade UI.

All drawing goes through `Renderer` so scenes stay declarative.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

from ..environment import Action, Cell, GridWorld
from . import assets

if TYPE_CHECKING:
    import pygame


@dataclass
class Layout:
    """Pixel layout of the arcade screen.

    `grid_origin` is the top-left (x, y) of the grid in screen coords. HUD
    sits above the grid; a footer strip with controls sits below.
    """

    tile_px: int
    grid_origin: tuple[int, int]
    grid_px: tuple[int, int]
    screen_size: tuple[int, int]
    hud_rect: tuple[int, int, int, int]
    footer_rect: tuple[int, int, int, int]


def compute_layout(env: GridWorld, *, max_screen: tuple[int, int] = (1280, 800)) -> Layout:
    """Pick a tile size that fits the grid into the screen with HUD+footer.

    HUD takes the top 64 px, footer the bottom 36 px, grid fills the middle,
    centered. We enforce a minimum screen width wide enough to fit the worst
    case HUD stat line (e.g. "SPD PAUSE" + REC badge) without truncation.
    """
    hud_h = 64
    footer_h = 36
    side_pad = 16
    # Worst-case width needed for the HUD stat line. We cannot measure the
    # fonts here (they're not loaded yet), so pick a conservative pixel budget
    # based on the `hud` font at 16 px / 8x8 chars * ~10 chars per field * 5
    # fields + REC badge + paddings. Empirically 960 px is comfortable.
    min_screen_w = 960
    avail_w = max_screen[0] - 2 * side_pad
    avail_h = max_screen[1] - hud_h - footer_h - 2 * side_pad

    max_tile_w = avail_w // env.w
    max_tile_h = avail_h // env.h
    tile_px = max(16, min(max_tile_w, max_tile_h))
    # Snap to multiples of 16 for crisp pixel art when possible.
    tile_px = (tile_px // 16) * 16 if tile_px >= 32 else tile_px

    grid_w = env.w * tile_px
    grid_h = env.h * tile_px
    screen_w = max(grid_w + 2 * side_pad, min_screen_w)
    screen_h = hud_h + grid_h + footer_h + 2 * side_pad
    origin_x = (screen_w - grid_w) // 2
    origin_y = hud_h + side_pad

    return Layout(
        tile_px=tile_px,
        grid_origin=(origin_x, origin_y),
        grid_px=(grid_w, grid_h),
        screen_size=(screen_w, screen_h),
        hud_rect=(0, 0, screen_w, hud_h),
        footer_rect=(0, screen_h - footer_h, screen_w, footer_h),
    )


@dataclass
class OverlayState:
    value_map: bool = False
    policy: bool = False
    visits: bool = False


class Renderer:
    def __init__(
        self,
        screen: "pygame.Surface",
        layout: Layout,
        sprites: assets.Sprites,
        fonts: assets.Fonts,
    ) -> None:
        self.screen = screen
        self.layout = layout
        self.sprites = sprites
        self.fonts = fonts

    def update_layout(self, layout: Layout, sprites: assets.Sprites) -> None:
        self.layout = layout
        self.sprites = sprites

    # -- Primitive drawing --------------------------------------------------

    def tile_rect(self, row: int, col: int) -> tuple[int, int, int, int]:
        tpx = self.layout.tile_px
        ox, oy = self.layout.grid_origin
        return (ox + col * tpx, oy + row * tpx, tpx, tpx)

    def clear(self) -> None:
        self.screen.fill(assets.BG_DARK)

    def draw_grid(self, env: GridWorld) -> None:
        """Draw floor / walls / start / goal / trap tiles."""
        import pygame

        for r in range(env.h):
            for c in range(env.w):
                cell = Cell(int(env.grid[r, c]))
                rect = self.tile_rect(r, c)
                if cell == Cell.WALL:
                    self.screen.blit(self.sprites.wall, rect[:2])
                else:
                    self.screen.blit(self.sprites.floor, rect[:2])
                    if cell == Cell.GOAL:
                        self.screen.blit(self.sprites.goal, rect[:2])
                    elif cell == Cell.TRAP:
                        self.screen.blit(self.sprites.trap, rect[:2])
                    elif cell == Cell.START:
                        self.screen.blit(self.sprites.start_mark, rect[:2])

        # Draw grid lines only between non-wall cells so adjacent walls stay
        # visually coherent (tiles that touch look like a solid structure).
        ox, oy = self.layout.grid_origin
        tpx = self.layout.tile_px
        for r in range(env.h):
            for c in range(env.w):
                # Bottom edge of (r, c) — drawn only if (r, c) or (r+1, c)
                # is not a wall.
                if r < env.h - 1:
                    top = Cell(int(env.grid[r, c])) == Cell.WALL
                    bot = Cell(int(env.grid[r + 1, c])) == Cell.WALL
                    if not (top and bot):
                        y = oy + (r + 1) * tpx
                        pygame.draw.line(
                            self.screen,
                            assets.GRID_LINE,
                            (ox + c * tpx, y),
                            (ox + (c + 1) * tpx, y),
                            1,
                        )
                # Right edge of (r, c)
                if c < env.w - 1:
                    left = Cell(int(env.grid[r, c])) == Cell.WALL
                    right = Cell(int(env.grid[r, c + 1])) == Cell.WALL
                    if not (left and right):
                        x = ox + (c + 1) * tpx
                        pygame.draw.line(
                            self.screen,
                            assets.GRID_LINE,
                            (x, oy + r * tpx),
                            (x, oy + (r + 1) * tpx),
                            1,
                        )
        # Outer border
        gw, gh = self.layout.grid_px
        pygame.draw.rect(self.screen, assets.GRID_LINE, (ox, oy, gw, gh), 1)

    def draw_agent(
        self,
        state: tuple[int, int],
        frame_idx: int,
        *,
        bump_offset: tuple[int, int] = (0, 0),
    ) -> None:
        """Draw the agent sprite at `state`.

        `bump_offset` shifts the sprite in pixels (dx, dy); used to visualize
        the agent bouncing off a wall after a blocked action.
        """
        rect = self.tile_rect(*state)
        frame = self.sprites.agent_frames[frame_idx % len(self.sprites.agent_frames)]
        self.screen.blit(frame, (rect[0] + bump_offset[0], rect[1] + bump_offset[1]))

    # -- Overlays -----------------------------------------------------------

    def draw_value_overlay(self, V: dict[tuple[int, int], float], env: GridWorld) -> None:
        """Semi-transparent heatmap of V(s)."""
        import pygame

        if not V:
            return
        vmin = min(V.values())
        vmax = max(V.values())
        span = max(1e-9, vmax - vmin)
        tpx = self.layout.tile_px
        for (r, c), v in V.items():
            if env.is_wall((r, c)):
                continue
            t = (v - vmin) / span
            color = _viridis(t)
            surf = pygame.Surface((tpx, tpx), pygame.SRCALPHA)
            surf.fill((*color, 140))
            self.screen.blit(surf, self.tile_rect(r, c)[:2])

    def draw_policy_overlay(
        self, policy: dict[tuple[int, int], Action], env: GridWorld
    ) -> None:
        for s, a in policy.items():
            if env.is_wall(s) or env.is_terminal(s):
                continue
            arrow = {
                Action.UP: self.sprites.arrow_up,
                Action.DOWN: self.sprites.arrow_down,
                Action.LEFT: self.sprites.arrow_left,
                Action.RIGHT: self.sprites.arrow_right,
            }[a]
            self.screen.blit(arrow, self.tile_rect(*s)[:2])

    def draw_visits_overlay(self, visits: np.ndarray, env: GridWorld) -> None:
        """Red-ish heatmap of how many times each cell was visited."""
        import pygame

        if visits.size == 0:
            return
        mx = float(visits.max())
        if mx <= 0:
            return
        tpx = self.layout.tile_px
        for r in range(env.h):
            for c in range(env.w):
                if env.is_wall((r, c)):
                    continue
                t = float(visits[r, c]) / mx
                if t <= 0:
                    continue
                alpha = int(40 + 160 * t)
                color = (255, int(120 * (1 - t)), int(60 * (1 - t)))
                surf = pygame.Surface((tpx, tpx), pygame.SRCALPHA)
                surf.fill((*color, alpha))
                self.screen.blit(surf, self.tile_rect(r, c)[:2])

    # -- HUD / footer -------------------------------------------------------

    def draw_hud(
        self,
        *,
        title: str,
        episode: int,
        total_episodes: int,
        score: float,
        steps: int,
        epsilon: float,
        speed_label: str,
        recording: bool,
    ) -> None:
        import pygame

        hud = pygame.Rect(self.layout.hud_rect)
        pygame.draw.rect(self.screen, (14, 15, 28), hud)
        pygame.draw.line(
            self.screen, assets.FG_WHITE, (0, hud.bottom - 1), (hud.right, hud.bottom - 1), 1
        )

        title_surf = self.fonts.menu.render(title, False, assets.FG_GOLD)
        self.screen.blit(title_surf, (12, 8))

        # Stat line
        score_color = assets.FG_RED if score < 0 else assets.FG_WHITE
        stats = [
            (f"EP {episode:04d}/{total_episodes:04d}", assets.FG_WHITE),
            (f"SCORE {score:+.2f}", score_color),
            (f"STEPS {steps:03d}", assets.FG_WHITE),
            (f"e {epsilon:.2f}", assets.FG_BLUE),
            (f"SPD {speed_label}", assets.FG_GOLD),
        ]
        # Reserve space on the right so the REC badge never overlaps stats.
        rec_reserve = 88 if recording else 16
        stats_right_limit = hud.right - rec_reserve
        x = 12
        y = 38
        for text, color in stats:
            surf = self.fonts.hud.render(text, False, color)
            if x + surf.get_width() > stats_right_limit:
                break
            self.screen.blit(surf, (x, y))
            x += surf.get_width() + 18

        if recording:
            rec_surf = self.fonts.hud.render("\u25CF REC", False, assets.FG_RED)
            self.screen.blit(rec_surf, (hud.right - rec_surf.get_width() - 12, 8))

        # Progress bar
        bar_w = hud.width - 24
        bar_h = 4
        bx = 12
        by = hud.bottom - 8
        pygame.draw.rect(self.screen, (40, 44, 66), (bx, by, bar_w, bar_h))
        frac = 0 if total_episodes <= 0 else min(1.0, episode / max(1, total_episodes))
        pygame.draw.rect(self.screen, assets.FG_GOLD, (bx, by, int(bar_w * frac), bar_h))

    def draw_footer(self, lines: list[str]) -> None:
        import pygame

        footer = pygame.Rect(self.layout.footer_rect)
        pygame.draw.rect(self.screen, (14, 15, 28), footer)
        pygame.draw.line(self.screen, assets.GRID_LINE, (0, footer.top), (footer.right, footer.top), 1)
        text = "   ".join(lines)
        surf = self.fonts.small.render(text, False, assets.FG_WHITE)
        self.screen.blit(surf, (12, footer.top + (footer.height - surf.get_height()) // 2))


def _viridis(t: float) -> tuple[int, int, int]:
    """Cheap approximation of matplotlib's viridis colormap.

    Linearly interpolates between five keypoints. Good enough for the overlay.
    """
    t = max(0.0, min(1.0, t))
    stops = [
        (0.0, (68, 1, 84)),
        (0.25, (59, 82, 139)),
        (0.5, (33, 145, 140)),
        (0.75, (94, 201, 98)),
        (1.0, (253, 231, 37)),
    ]
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if t <= t1:
            u = (t - t0) / (t1 - t0)
            return tuple(int(c0[k] + u * (c1[k] - c0[k])) for k in range(3))
    return stops[-1][1]
