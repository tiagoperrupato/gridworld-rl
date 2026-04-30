"""Streamlit UI for running Gridworld RL experiments.

Thin wrapper around `main.py`: collect params via widgets, build the
equivalent CLI invocation with a smart auto-generated `--run-name`,
stream stdout live, and show the resulting figures.
"""

from __future__ import annotations

import re
import shlex
import subprocess
import sys
from pathlib import Path

import streamlit as st

from src.maps import map_keys

REPO_ROOT = Path(__file__).resolve().parent

DEFAULTS = {
    "stochastic": False,
    "wind_prob": 0.8,
    "compare_wind_exploration": False,
    "gamma": 0.99,
    "theta": 1e-6,
    "episodes": 1000,
    "max_steps": 200,
    "alpha": 0.1,
    "epsilon": 1.0,
    "epsilon_decay": 0.995,
    "epsilon_min": 0.05,
    "maps": ["default"],
    "seeds": [0, 1, 2, 3, 4],
    "output_dir": "output",
}


def _fmt_num(x: float) -> str:
    s = f"{x:g}"
    return s.replace(".", "p")


def build_run_slug(params: dict) -> str:
    parts: list[str] = []
    maps = params["maps"]
    if maps != DEFAULTS["maps"]:
        if "all" in maps:
            parts.append("allmaps")
        elif len(maps) > 0:
            parts.append("-".join(maps))
    if params["number_random_maps"] > 0:
        parts.append(f"rand{params['number_random_maps']}x{params['random_map_size']}")
    if params["stochastic"] != DEFAULTS["stochastic"]:
        parts.append(f"wind{int(params['wind_prob'] * 100)}")
    elif params["wind_prob"] != DEFAULTS["wind_prob"]:
        parts.append(f"wp{int(params['wind_prob'] * 100)}")
    if params["compare_wind_exploration"]:
        parts.append("expwind")
    if params["gamma"] != DEFAULTS["gamma"]:
        parts.append(f"g{_fmt_num(params['gamma'])}")
    if params["theta"] != DEFAULTS["theta"]:
        parts.append(f"th{_fmt_num(params['theta'])}")
    if params["episodes"] != DEFAULTS["episodes"]:
        parts.append(f"ep{params['episodes']}")
    if params["max_steps"] != DEFAULTS["max_steps"]:
        parts.append(f"ms{params['max_steps']}")
    if params["alpha"] != DEFAULTS["alpha"]:
        parts.append(f"a{_fmt_num(params['alpha'])}")
    if params["epsilon"] != DEFAULTS["epsilon"]:
        parts.append(f"e{_fmt_num(params['epsilon'])}")
    if params["epsilon_decay"] != DEFAULTS["epsilon_decay"]:
        parts.append(f"edec{_fmt_num(params['epsilon_decay'])}")
    if params["epsilon_min"] != DEFAULTS["epsilon_min"]:
        parts.append(f"emin{_fmt_num(params['epsilon_min'])}")
    seeds = params["seeds"]
    if seeds != DEFAULTS["seeds"]:
        parts.append("s" + "-".join(map(str, seeds)))
    return "-".join(parts) if parts else "default"


def build_argv(params: dict, run_name: str) -> list[str]:
    argv: list[str] = [sys.executable, "main.py"]
    argv += ["--maps", *params["maps"]]
    argv += ["--number-random-maps", str(params["number_random_maps"])]
    argv += ["--random-map-size", str(params["random_map_size"])]
    argv += ["--seeds", *map(str, params["seeds"])]
    argv += ["--run-name", run_name]
    argv += ["--output-dir", params["output_dir"]]

    if params["stochastic"]:
        argv.append("--stochastic")
    if params["wind_prob"] != DEFAULTS["wind_prob"]:
        argv += ["--wind-prob", str(params["wind_prob"])]
    if params["compare_wind_exploration"]:
        argv.append("--compare-wind-exploration")
    if params["gamma"] != DEFAULTS["gamma"]:
        argv += ["--gamma", str(params["gamma"])]
    if params["theta"] != DEFAULTS["theta"]:
        argv += ["--theta", str(params["theta"])]
    if params["episodes"] != DEFAULTS["episodes"]:
        argv += ["--episodes", str(params["episodes"])]
    if params["max_steps"] != DEFAULTS["max_steps"]:
        argv += ["--max-steps", str(params["max_steps"])]
    if params["alpha"] != DEFAULTS["alpha"]:
        argv += ["--alpha", str(params["alpha"])]
    if params["epsilon"] != DEFAULTS["epsilon"]:
        argv += ["--epsilon", str(params["epsilon"])]
    if params["epsilon_decay"] != DEFAULTS["epsilon_decay"]:
        argv += ["--epsilon-decay", str(params["epsilon_decay"])]
    if params["epsilon_min"] != DEFAULTS["epsilon_min"]:
        argv += ["--epsilon-min", str(params["epsilon_min"])]
    return argv


def parse_seeds(text: str) -> list[int]:
    cleaned = text.replace(",", " ").split()
    return [int(x) for x in cleaned]


def stream_process(argv: list[str], log_area) -> tuple[int, str]:
    buffer: list[str] = []
    proc = subprocess.Popen(
        argv,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        buffer.append(line.rstrip())
        # Only render the last ~200 lines to keep the UI responsive.
        tail = "\n".join(buffer[-200:])
        log_area.code(tail, language="text")
    proc.wait()
    return proc.returncode, "\n".join(buffer)


def extract_run_dir(log: str, output_dir: Path) -> Path | None:
    matches = re.findall(r"Run directory:\s*(.+)", log)
    if matches:
        return Path(matches[-1].strip())
    latest = output_dir / "latest"
    if latest.exists():
        return latest.resolve()
    return None


def render_gallery(run_dir: Path, selected_maps: list[str]) -> None:
    # Resolve concrete map slugs. `map_keys()` returns short keys which match
    # the subdir names used by `main.py` via `MapChoice.slug`? Actually the
    # subdir is `mc.slug` = slugified full name. Walk the dir instead.
    subdirs = [p for p in sorted(run_dir.iterdir()) if p.is_dir() and p.name not in {".cache", ".mplconfig"}]
    if not subdirs:
        st.info("No map subdirectories found in the run directory yet.")
        return
    for map_dir in subdirs:
        st.subheader(f"Map: {map_dir.name}")
        for section in ("vi", "ql", "comparisons", "analysis"):
            section_dir = map_dir / section
            if not section_dir.exists():
                continue
            pngs = sorted(section_dir.glob("*.png"))
            if not pngs:
                continue
            st.markdown(f"**{section}**")
            cols = st.columns(min(3, len(pngs)))
            for i, png in enumerate(pngs):
                with cols[i % len(cols)]:
                    st.image(str(png), caption=png.name, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Gridworld RL Runner", layout="wide")
    st.title("Gridworld RL — Experiment Runner")
    st.caption("Pick your params, preview the command, launch a run, and browse the results.")

    with st.sidebar:
        st.header("Maps")
        map_options = [*map_keys(), "all"]
        maps = st.multiselect("Maps to sweep", options=map_options, default=DEFAULTS["maps"])
        random_maps = st.number_input("Number of random maps to generate", min_value=0, max_value=100, value=0, step=1)
        if (random_maps > 0):
            random_map_size = st.number_input("Size of random maps (NxN)", min_value=5, max_value=50, value=12, step=1)
        else:
            random_map_size = 0

        st.header("Environment")
        stochastic = st.checkbox("Stochastic wind", value=DEFAULTS["stochastic"])
        wind_prob = st.slider(
            "Wind probability (intended action)",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULTS["wind_prob"],
            step=0.05,
        )
        compare_wind_exploration = st.checkbox(
            "Compare exploration strategies: deterministic vs wind",
            value=DEFAULTS["compare_wind_exploration"],
        )

        st.header("Core hyperparameters")
        gamma = st.number_input("gamma", min_value=0.0, max_value=1.0, value=DEFAULTS["gamma"], step=0.01, format="%.4f")
        theta = st.number_input(
            "theta (VI threshold)",
            min_value=1e-12,
            max_value=1.0,
            value=DEFAULTS["theta"],
            step=1e-6,
            format="%.2e",
        )
        episodes = st.number_input("episodes", min_value=1, max_value=100000, value=DEFAULTS["episodes"], step=50)
        max_steps = st.number_input("max_steps", min_value=1, max_value=10000, value=DEFAULTS["max_steps"], step=10)
        alpha = st.number_input("alpha", min_value=0.0, max_value=1.0, value=DEFAULTS["alpha"], step=0.01, format="%.4f")

        st.header("Exploration")
        epsilon = st.number_input("epsilon", min_value=0.0, max_value=1.0, value=DEFAULTS["epsilon"], step=0.05, format="%.4f")
        epsilon_decay = st.number_input(
            "epsilon_decay",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULTS["epsilon_decay"],
            step=0.001,
            format="%.4f",
        )
        epsilon_min = st.number_input(
            "epsilon_min",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULTS["epsilon_min"],
            step=0.01,
            format="%.4f",
        )

        st.header("Seeds")
        seed_mode = st.radio("Seed mode", ["Multi-seed list", "Single seed"], horizontal=True)
        if seed_mode == "Single seed":
            single_seed = st.number_input("seed", min_value=0, max_value=10_000_000, value=0, step=1)
            seeds = [int(single_seed)]
        else:
            seeds_text = st.text_input("seeds (comma or space separated)", value="0, 1, 2, 3, 4")
            try:
                seeds = parse_seeds(seeds_text)
                if not seeds:
                    raise ValueError("empty")
            except ValueError:
                st.error("Seeds must be integers separated by commas or spaces.")
                seeds = DEFAULTS["seeds"]

        st.header("Output")
        output_dir = st.text_input("output directory", value=DEFAULTS["output_dir"])
        run_name_override = st.text_input("run name (leave blank for auto)", value="")

    if not maps and random_maps == 0:
        st.warning("Select at least one map to continue.")
        return

    params = {
        "maps": maps,
        "stochastic": stochastic,
        "wind_prob": float(wind_prob),
        "compare_wind_exploration": compare_wind_exploration,
        "gamma": float(gamma),
        "theta": float(theta),
        "episodes": int(episodes),
        "max_steps": int(max_steps),
        "alpha": float(alpha),
        "epsilon": float(epsilon),
        "epsilon_decay": float(epsilon_decay),
        "epsilon_min": float(epsilon_min),
        "seeds": list(seeds),
        "output_dir": output_dir or DEFAULTS["output_dir"],
        "number_random_maps": int(random_maps),
        "random_map_size": int(random_map_size),
    }

    auto_slug = build_run_slug(params)
    run_name = run_name_override.strip() or auto_slug
    argv = build_argv(params, run_name)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Run name")
        st.code(run_name, language="text")
        if not run_name_override.strip():
            st.caption("Auto-generated from non-default choices. Override in the sidebar if you prefer.")
    with col2:
        st.subheader("Command")
        st.code(" ".join(shlex.quote(a) for a in argv), language="bash")

    run_clicked = st.button("Run experiment", type="primary", use_container_width=True)

    if run_clicked:
        st.subheader("Live output")
        log_placeholder = st.empty()
        with st.status(f"Running {run_name}...", expanded=True) as status:
            try:
                returncode, full_log = stream_process(argv, log_placeholder)
            except FileNotFoundError as exc:
                status.update(label=f"Failed to launch: {exc}", state="error")
                return
            if returncode == 0:
                status.update(label=f"Finished: {run_name}", state="complete")
            else:
                status.update(label=f"Exited with code {returncode}", state="error")

        run_dir = extract_run_dir(full_log, REPO_ROOT / params["output_dir"])
        if run_dir is None:
            st.warning("Could not locate the run directory from the logs.")
            return

        st.success(f"Run directory: `{run_dir}`")
        st.subheader("Generated figures")
        try:
            render_gallery(run_dir, params["maps"])
        except FileNotFoundError:
            st.info("Run directory not found on disk — did the process fail early?")


if __name__ == "__main__":
    main()
