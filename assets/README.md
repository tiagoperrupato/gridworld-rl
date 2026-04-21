# Assets

All assets in this folder are either in the **public domain (CC0)** or the **SIL Open Font License (OFL)**. No attribution is required, but we list sources for transparency.

## Sprites (`sprites/`)

- `tiny_dungeon.png` — Kenney's **Tiny Dungeon** pack (CC0), a 192×176 packed tilesheet of 16×16 tiles. Original at <https://kenney.nl/assets/tiny-dungeon>. The bundled `tiny_dungeon_LICENSE.txt` is the upstream license text (CC0 1.0 Universal).

We use a small subset of the sheet (see `src/ui/assets.py::TD_*_RC`):

| Tile | Coord (row, col) | Use |
|------|------------------|-----|
| Sandy floor | (4, 1) | empty grid cells |
| Stone wall  | (4, 10) | obstacle cells |
| Closed treasure chest | (7, 5) | goal tile |
| Teal ghost | (9, 0) | trap tile |
| Ranger / Knight / Wizard / Viking | (9, 4) / (8, 0) / (7, 0) / (7, 3) | player characters (see `CHARACTERS` in `src/ui/assets.py`) |

The sprites are pre-colored in the source PNG, so no runtime tinting is needed (unlike the earlier 1-Bit Pack setup). A 1-pixel vertical bobbing across four frames is synthesized programmatically to fake a walk-cycle.

## Fonts (`fonts/`)

- `PressStart2P-Regular.ttf` — [Press Start 2P](https://fonts.google.com/specimen/Press+Start+2P) by CodeMan38, distributed under the SIL Open Font License 1.1.

## Sounds (`sounds/`)

- `goal.wav`, `trap.wav`, `step.wav`, `menu.wav`, `select.wav` — generated synthetically from square/sweep waves in NumPy (see the script embedded in the git history). Placed in the public domain alongside the rest of the project.
