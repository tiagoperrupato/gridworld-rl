"""Entry point for the arcade Pygame UI.

Run:
    ./.venv/bin/python main_ui.py
    ./.venv/bin/python main_ui.py --output-dir output --run-name demo
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Gridworld RL arcade UI (Pygame). See the title menu for controls.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Root directory for runs (default: output/). GIFs go into the created run dir.",
    )
    p.add_argument(
        "--run-name",
        type=str,
        default="ui",
        help="Slug appended to the run directory name (default: ui).",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str((args.output_dir / ".mplconfig").resolve()))
    os.environ.setdefault("XDG_CACHE_HOME", str((args.output_dir / ".cache").resolve()))

    from src.ui.app import run_arcade

    run_arcade(output_dir=args.output_dir, run_name=args.run_name)


if __name__ == "__main__":
    main()
