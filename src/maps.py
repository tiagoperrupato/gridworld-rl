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
