from __future__ import annotations

import json
import re
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

_SLUG_RE = re.compile(r"[^a-zA-Z0-9_-]+")


def _slugify(value: str) -> str:
    slug = _SLUG_RE.sub("-", value.strip()).strip("-")
    return slug or "run"


def create_run_dir(
    output_root: Path | str,
    *,
    slug: str | None = None,
    timestamp: datetime | None = None,
) -> Path:
    """Create a fresh timestamped run directory with the standard subfolders.

    Layout:
        <output_root>/runs/<YYYY-MM-DD_HH-MM-SS>_<slug>/
            vi/
            ql/
            comparisons/
    """
    output_root = Path(output_root)
    ts = (timestamp or datetime.now()).strftime("%Y-%m-%d_%H-%M-%S")
    name = f"{ts}_{_slugify(slug or 'default')}"

    run_dir = output_root / "runs" / name
    for sub in ("vi", "ql", "comparisons"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir


def update_latest_symlink(run_dir: Path) -> None:
    """Point <output_root>/latest at the given run_dir.

    On platforms that don't support symlinks (or if we lack permissions),
    silently write a small text pointer file instead.
    """
    run_dir = Path(run_dir).resolve()
    output_root = run_dir.parent.parent
    latest = output_root / "latest"

    try:
        if latest.is_symlink() or latest.exists():
            if latest.is_symlink() or latest.is_file():
                latest.unlink()
            else:
                # Directory — don't blow it away; fall back to pointer file.
                raise OSError("latest exists and is a directory")
        latest.symlink_to(run_dir, target_is_directory=True)
    except (OSError, NotImplementedError):
        pointer = output_root / "latest.txt"
        pointer.write_text(str(run_dir) + "\n", encoding="utf-8")


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {k: _to_jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def write_config(run_dir: Path, config: Any, extra: dict[str, Any] | None = None) -> Path:
    """Serialize the experiment config + extras to run_dir/config.yaml.

    Uses PyYAML if available; otherwise falls back to a minimal YAML writer so
    the project still works without the optional dependency.
    """
    payload: dict[str, Any] = {"config": _to_jsonable(config)}
    if extra:
        payload.update({k: _to_jsonable(v) for k, v in extra.items()})

    path = run_dir / "config.yaml"
    try:
        import yaml  # type: ignore[import-not-found]

        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
    except ImportError:
        path.write_text(_minimal_yaml(payload), encoding="utf-8")
    return path


def write_metrics(run_dir: Path, metrics: dict[str, Any]) -> Path:
    path = run_dir / "metrics.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(metrics), f, indent=2)
    return path


def _minimal_yaml(obj: Any, indent: int = 0) -> str:
    """Tiny YAML emitter used only when PyYAML isn't installed.

    Handles dicts, lists, and scalars (str/int/float/bool/None). Good enough
    for our config/metrics shapes; not a general YAML implementation.
    """
    pad = "  " * indent
    if isinstance(obj, dict):
        if not obj:
            return f"{pad}{{}}\n"
        out: list[str] = []
        for k, v in obj.items():
            if isinstance(v, (dict, list)) and v:
                out.append(f"{pad}{k}:\n{_minimal_yaml(v, indent + 1)}")
            else:
                out.append(f"{pad}{k}: {_scalar(v)}\n")
        return "".join(out)
    if isinstance(obj, list):
        if not obj:
            return f"{pad}[]\n"
        out = []
        for item in obj:
            if isinstance(item, (dict, list)) and item:
                rendered = _minimal_yaml(item, indent + 1).lstrip()
                out.append(f"{pad}- {rendered}")
            else:
                out.append(f"{pad}- {_scalar(item)}\n")
        return "".join(out)
    return f"{pad}{_scalar(obj)}\n"


def _scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    s = str(value)
    if s == "" or any(ch in s for ch in ":#\n\"'") or s.strip() != s:
        return json.dumps(s, ensure_ascii=False)
    return s
