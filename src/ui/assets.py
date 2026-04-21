"""Asset loader for the Pygame arcade UI.

Responsibilities:
- Load the Kenney Tiny Dungeon tilesheet (CC0, 16x16, 12x11 tiles) and slice
  out the tiles we use for floor/wall/goal/trap and the agent character.
- Expose a `Sprites` dataclass with pre-scaled pygame Surfaces for each
  semantic tile (floor, wall, goal, trap, agent frames, arrows, start mark).
- Load the Press Start 2P TTF font (OFL) at a few arcade-friendly sizes.
- Load generated 8-bit sound effects (goal/trap/step/menu/select).

Tile coordinates are picked once, visually (see `TINY_DUNGEON_*_RC`). They
can be tweaked without touching the renderer / scenes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import pygame

ASSETS_ROOT = Path(__file__).resolve().parents[2] / "assets"

TILEMAP_PATH = ASSETS_ROOT / "sprites" / "tiny_dungeon.png"
FONT_PATH = ASSETS_ROOT / "fonts" / "PressStart2P-Regular.ttf"
SOUNDS_DIR = ASSETS_ROOT / "sounds"

BASE_TILE_PX = 16

# Tile coordinates on the Tiny Dungeon packed sheet (row, col), 0-indexed.
# The sheet is 12 columns x 11 rows of 16x16 tiles.
TD_FLOOR_RC = (4, 1)    # sandy floor with subtle variation
TD_WALL_RC = (3, 4)     # stone brick wall (seamless horizontally)
TD_GOAL_RC = (7, 5)     # closed treasure chest
TD_TRAP_RC = (9, 0)     # friendly teal ghost (readable on dark bg)
TD_AGENT_RC = (9, 4)    # default: green-cloaked ranger ("Link")

# Player character roster. Each entry maps a short id to a (row, col) on the
# packed Tiny Dungeon sheet. Visually distinct archetypes so the menu choice
# feels meaningful.
CHARACTERS: dict[str, tuple[int, int]] = {
    "RANGER": (9, 4),  # green cloak, headband (default)
    "KNIGHT": (8, 0),  # silver armor
    "WIZARD": (7, 0),  # purple robe, white beard
    "VIKING": (7, 3),  # redhead with horned helm
}
CHARACTER_ORDER: list[str] = ["RANGER", "KNIGHT", "WIZARD", "VIKING"]

# UI palette (used by HUD/overlays/renderer). Sprites themselves are already
# colored in the source PNG — no tinting needed, keeping the Kenney look.
FG_WHITE = (235, 235, 235)
FG_GOLD = (250, 204, 92)
FG_RED = (230, 72, 72)
FG_BLUE = (100, 170, 255)
FG_GREEN = (140, 220, 140)
BG_DARK = (20, 22, 38)
BG_WALL = (10, 10, 18)
GRID_LINE = (48, 50, 72)


@dataclass
class Sprites:
    """Cache of pygame Surfaces at the current tile size.

    All surfaces are square of `tile_px` side.
    """

    tile_px: int
    floor: "pygame.Surface"
    wall: "pygame.Surface"
    goal: "pygame.Surface"
    trap: "pygame.Surface"
    agent_frames: list["pygame.Surface"]
    arrow_up: "pygame.Surface"
    arrow_down: "pygame.Surface"
    arrow_left: "pygame.Surface"
    arrow_right: "pygame.Surface"
    start_mark: "pygame.Surface"


@dataclass
class Fonts:
    hud: "pygame.font.Font"
    title: "pygame.font.Font"
    menu: "pygame.font.Font"
    small: "pygame.font.Font"


@dataclass
class Sounds:
    goal: Optional["pygame.mixer.Sound"]
    trap: Optional["pygame.mixer.Sound"]
    step: Optional["pygame.mixer.Sound"]
    menu: Optional["pygame.mixer.Sound"]
    select: Optional["pygame.mixer.Sound"]

    def play(self, name: str, volume: float = 1.0) -> None:
        snd = getattr(self, name, None)
        if snd is None:
            return
        snd.set_volume(volume)
        snd.play()


def _load_tile(sheet: "pygame.Surface", row: int, col: int) -> "pygame.Surface":
    rect = (col * BASE_TILE_PX, row * BASE_TILE_PX, BASE_TILE_PX, BASE_TILE_PX)
    return sheet.subsurface(rect).copy()


def _scale(surface: "pygame.Surface", target_px: int) -> "pygame.Surface":
    import pygame

    if surface.get_width() == target_px:
        return surface
    # Nearest-neighbor keeps the pixel-art crisp at any zoom.
    return pygame.transform.scale(surface, (target_px, target_px))


def _make_agent_frames(
    base: "pygame.Surface", tile_px: int, n_frames: int = 4
) -> list["pygame.Surface"]:
    """Build a simple idle/walk animation from a single-frame sprite.

    The Tiny Dungeon pack ships one pose per character. We fake a bounce by
    shifting the sprite 1 base-pixel up/down across `n_frames` frames. This
    looks convincingly animated once played back at ~10 fps.
    """
    import pygame

    frames: list[pygame.Surface] = []
    scaled = _scale(base, tile_px)
    step = max(1, tile_px // BASE_TILE_PX)  # 1 "base pixel" in screen pixels
    # Sequence: 0, +1, 0, -1 — tiny vertical bobbing.
    bobs = [0, -step, 0, step]
    for i in range(n_frames):
        frame = pygame.Surface((tile_px, tile_px), pygame.SRCALPHA)
        frame.blit(scaled, (0, bobs[i % len(bobs)]))
        frames.append(frame)
    return frames


def _draw_arrow(tile_px: int, direction: tuple[int, int]) -> "pygame.Surface":
    """Draw a chunky pixel-art arrow pointing in `direction` (dr, dc).

    Used for policy overlays. Style: filled triangle + tail.
    """
    import pygame

    dr, dc = direction
    surf = pygame.Surface((tile_px, tile_px), pygame.SRCALPHA)
    cx = tile_px // 2
    cy = tile_px // 2
    size = tile_px // 3
    if (dr, dc) == (-1, 0):  # UP
        pts = [(cx, cy - size), (cx - size, cy + size // 2), (cx + size, cy + size // 2)]
        tail = ((cx - size // 4, cy), (cx + size // 4, cy + size))
    elif (dr, dc) == (1, 0):  # DOWN
        pts = [(cx, cy + size), (cx - size, cy - size // 2), (cx + size, cy - size // 2)]
        tail = ((cx - size // 4, cy - size), (cx + size // 4, cy))
    elif (dr, dc) == (0, -1):  # LEFT
        pts = [(cx - size, cy), (cx + size // 2, cy - size), (cx + size // 2, cy + size)]
        tail = ((cx, cy - size // 4), (cx + size, cy + size // 4))
    else:  # RIGHT
        pts = [(cx + size, cy), (cx - size // 2, cy - size), (cx - size // 2, cy + size)]
        tail = ((cx - size, cy - size // 4), (cx, cy + size // 4))
    pygame.draw.polygon(surf, FG_BLUE, pts)
    pygame.draw.rect(
        surf,
        FG_BLUE,
        pygame.Rect(tail[0], (tail[1][0] - tail[0][0], tail[1][1] - tail[0][1])),
    )
    return surf


def _draw_start_mark(tile_px: int) -> "pygame.Surface":
    """Hollow diamond overlay highlighting the start cell."""
    import pygame

    surf = pygame.Surface((tile_px, tile_px), pygame.SRCALPHA)
    cx = tile_px // 2
    cy = tile_px // 2
    r = tile_px // 3
    pts = [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)]
    pygame.draw.polygon(surf, FG_BLUE, pts, max(1, tile_px // 10))
    return surf


def load_sprites(
    tile_px: int, *, agent_rc: tuple[int, int] = TD_AGENT_RC
) -> Sprites:
    """Load and scale all sprites to `tile_px`.

    `agent_rc` selects which character tile to use (see `CHARACTERS`). All
    other tiles stay fixed. Called once per resize / at startup.
    """
    import pygame

    sheet = pygame.image.load(str(TILEMAP_PATH)).convert_alpha()

    floor_raw = _load_tile(sheet, *TD_FLOOR_RC)
    wall_raw = _load_tile(sheet, *TD_WALL_RC)
    goal_raw = _load_tile(sheet, *TD_GOAL_RC)
    trap_raw = _load_tile(sheet, *TD_TRAP_RC)
    agent_raw = _load_tile(sheet, *agent_rc)

    # Floor is fully opaque in the source, so the scaled version is ready
    # to use as-is. No background compositing needed.
    floor = _scale(floor_raw, tile_px)
    wall = _scale(wall_raw, tile_px)

    # Goal and trap carry alpha (transparent borders). Composite onto the
    # floor so they blend with the rest of the scene.
    def _over_floor(sprite: pygame.Surface) -> pygame.Surface:
        base = floor.copy()
        scaled_sprite = _scale(sprite, tile_px)
        base.blit(scaled_sprite, (0, 0))
        return base

    goal = _over_floor(goal_raw)
    trap = _over_floor(trap_raw)

    agent_frames = _make_agent_frames(agent_raw, tile_px)

    return Sprites(
        tile_px=tile_px,
        floor=floor,
        wall=wall,
        goal=goal,
        trap=trap,
        agent_frames=agent_frames,
        arrow_up=_draw_arrow(tile_px, (-1, 0)),
        arrow_down=_draw_arrow(tile_px, (1, 0)),
        arrow_left=_draw_arrow(tile_px, (0, -1)),
        arrow_right=_draw_arrow(tile_px, (0, 1)),
        start_mark=_draw_start_mark(tile_px),
    )


def load_fonts() -> Fonts:
    import pygame

    path = str(FONT_PATH)
    return Fonts(
        hud=pygame.font.Font(path, 12),
        title=pygame.font.Font(path, 28),
        menu=pygame.font.Font(path, 16),
        small=pygame.font.Font(path, 9),
    )


def load_sounds() -> Sounds:
    import pygame

    def _try(name: str) -> Optional[pygame.mixer.Sound]:
        p = SOUNDS_DIR / name
        if not p.exists():
            return None
        try:
            return pygame.mixer.Sound(str(p))
        except pygame.error:
            return None

    return Sounds(
        goal=_try("goal.wav"),
        trap=_try("trap.wav"),
        step=_try("step.wav"),
        menu=_try("menu.wav"),
        select=_try("select.wav"),
    )
