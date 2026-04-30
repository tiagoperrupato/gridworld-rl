"""Built-in map layouts shared between the headless pipeline and the arcade UI.

The layouts used to live in `src/ui/scenes.py`, which made them inaccessible
to anything that didn't import pygame. Keeping them here lets the headless
runner sweep over multiple maps without dragging in UI dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MapChoice:
    name: str
    layout: tuple[str, ...]

    @property
    def slug(self) -> str:
        """Filesystem-safe slug derived from `name` (e.g. "DEFAULT 5x5" -> "default_5x5")."""
        return _slugify(self.name)


def _slugify(value: str) -> str:
    out: list[str] = []
    prev_dash = False
    for ch in value.strip().lower():
        if ch.isalnum():
            out.append(ch)
            prev_dash = False
        elif not prev_dash:
            out.append("_")
            prev_dash = True
    return "".join(out).strip("_") or "map"


DEFAULT_MAPS: list[MapChoice] = [
    MapChoice(
        "DEFAULT 5x5",
        (
            "S...#",
            ".#..T",
            ".#.#.",
            "..#..",
            "...G.",
        ),
    ),
    MapChoice(
        "OPEN 7x7",
        (
            "S......",
            ".##..T.",
            ".......",
            "..##...",
            "...T...",
            ".##....",
            "......G",
        ),
    ),
    MapChoice(
        "MAZE 9x9",
        (
            "S........",
            ".#######.",
            ".#.....#.",
            ".#.###.#.",
            ".#.#G#.#.",
            ".#.#.#.#.",
            ".#...#.#.",
            ".#####.#.",
            "........T",
        ),
    ),
    MapChoice(
        "GAUNTLET 6x10",
        (
            "S.T..T....",
            "..........",
            ".####.####",
            "..........",
            "####.####.",
            "....T....G",
        ),
    ),
    MapChoice(
        "CORRIDORS 10x4",
        (
            "S...",
            ".##.",
            ".T#.",
            ".T#.",
            ".T#.",
            ".T#.",
            ".T#.",
            ".T#.",
            ".##.",
            "G...",  )
    ),
]


_BY_SLUG: dict[str, MapChoice] = {m.slug: m for m in DEFAULT_MAPS}
# Friendly short keys for the CLI: "default", "open", "maze", "gauntlet".
_BY_SHORT: dict[str, MapChoice] = {m.name.split()[0].lower(): m for m in DEFAULT_MAPS}


def get_map(key: str) -> MapChoice:
    """Look up a map by short key ('default'), slug ('default_5x5'), or full name ('DEFAULT 5x5').

    Raises `KeyError` with a helpful message when the key doesn't match any
    built-in map.
    """
    k = key.strip().lower()
    if k in _BY_SHORT:
        return _BY_SHORT[k]
    if k in _BY_SLUG:
        return _BY_SLUG[k]
    full_match = _BY_SLUG.get(_slugify(key))
    if full_match is not None:
        return full_match
    options = ", ".join(sorted(_BY_SHORT))
    raise KeyError(f"unknown map {key!r}. Available: {options}")


def map_keys() -> list[str]:
    """All short keys in declaration order — useful for argparse `choices`."""
    return [m.name.split()[0].lower() for m in DEFAULT_MAPS]

def generate_random_map(rows, cols, start, goal, num_traps, wall_prob = 0.3, name=None, rng=None) -> MapChoice:
    """Generate a random map with the given dimensions and number of traps.
    Walls and traps are added randomly"""
    import random

    if rng is None:
        rng = random.Random()
    layout = [["." for _ in range(cols)] for _ in range(rows)]
    layout[start[0]][start[1]] = "S"
    layout[goal[0]][goal[1]] = "G"

    # Place traps randomly, avoiding start and goal.
    empty_cells = [(r, c) for r in range(rows) for c in range(cols) if (r, c) not in [start, goal]]
    rng.shuffle(empty_cells)
    for r, c in empty_cells[:num_traps]:
        layout[r][c] = "T"

    # Add walls randomly.
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in [start, goal] and layout[r][c] == ".":
                if rng.random() < wall_prob:
                    layout[r][c] = "#"

    return MapChoice(name or f"RANDOM {rows}x{cols} ({num_traps} traps)", tuple("".join(row) for row in layout))


def is_solvable(map_choice: MapChoice) -> bool:
    """Check if the map is solvable using a simple breadth-first search."""
    from collections import deque

    layout = map_choice.layout
    rows, cols = len(layout), len(layout[0])
    start = next((r, c) for r in range(rows) for c in range(cols) if layout[r][c] == "S")
    goal = next((r, c) for r in range(rows) for c in range(cols) if layout[r][c] == "G")

    queue = deque([start])
    visited = set()
    while queue:
        r, c = queue.popleft()
        if (r, c) == goal:
            return True
        if (r, c) in visited:
            continue
        visited.add((r, c))
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and layout[nr][nc] not in ("#", "T") and (nr, nc) not in visited:
                queue.append((nr, nc))
    return False

def generate_solvable_maps(num_maps: int, side : int, seed=None, max_attempts: int = 1000) -> list[MapChoice]:
    """Generate a list of random maps with the given parameters."""
    import random
    rng = random.Random(seed) if seed is not None else random.Random()
    rows = cols = side
    num_traps = max(1, (rows * cols)//16)  # Heuristic: 1 trap per 16 cells
    wall_prob = 0.2
    maps = []
    for i in range(num_maps):
        print(f"Generating map {i + 1}/{num_maps}...")
        map_name = f"RANDOM {rows}x{cols} ({num_traps} traps) #{i:03d}"
        for attempt in range(max_attempts):
            map_choice = generate_random_map(rows, cols, (0, 0), (rows - 1, cols - 1), num_traps, wall_prob, map_name, rng)
            if is_solvable(map_choice):
                maps.append(map_choice)
                break
        else:
            print(f"Failed to generate a solvable map after {max_attempts} attempts.")
    return maps
