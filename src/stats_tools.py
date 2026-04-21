"""Small statistics helpers for multi-seed experiments.

Kept intentionally tiny and numpy-only so the headless pipeline doesn't gain
extra runtime dependencies for topic 3 of the roadmap.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .environment import Action


def aggregate_curves(curves: Sequence[Sequence[float]]) -> dict:
    """Aggregate a list of per-seed learning curves into mean / std arrays.

    All curves are truncated to the shortest length so the output is rectangular
    (Q-learning runs with the same `num_episodes` always have equal length, but
    being defensive here keeps the helper reusable).

    Returns a dict with keys ``mean`` (np.ndarray), ``std`` (np.ndarray), and
    ``n`` (int, number of input curves). For an empty input, ``mean`` and
    ``std`` are zero-length arrays and ``n`` is 0.
    """
    if not curves:
        return {"mean": np.zeros(0, dtype=float), "std": np.zeros(0, dtype=float), "n": 0}

    min_len = min(len(c) for c in curves)
    if min_len == 0:
        return {"mean": np.zeros(0, dtype=float), "std": np.zeros(0, dtype=float), "n": len(curves)}

    stacked = np.asarray([list(c)[:min_len] for c in curves], dtype=float)
    return {
        "mean": stacked.mean(axis=0),
        "std": stacked.std(axis=0, ddof=0),
        "n": stacked.shape[0],
    }


def episodes_to_threshold(
    rewards: Sequence[float],
    threshold: float,
    *,
    window: int = 20,
) -> int | None:
    """First episode at which the `window`-moving-average of `rewards` hits `threshold`.

    Returns ``None`` if the threshold is never met. We use a trailing moving
    average (episode index i uses rewards[i-window+1 .. i]); the return value is
    the 0-indexed episode where that average first meets or exceeds `threshold`.
    For episodes with fewer than `window` prior samples, the average is computed
    over whatever is available so we don't have to wait `window` episodes before
    we can possibly register convergence on easy configs.
    """
    arr = np.asarray(list(rewards), dtype=float)
    if arr.size == 0 or window <= 0:
        return None

    cumsum = np.concatenate(([0.0], np.cumsum(arr)))
    for i in range(arr.size):
        start = max(0, i - window + 1)
        count = i - start + 1
        avg = (cumsum[i + 1] - cumsum[start]) / count
        if avg >= threshold:
            return i
    return None


def policy_agreement(
    q_policy: dict[tuple[int, int], Action],
    reference_policy: dict[tuple[int, int], Action],
) -> float:
    """Fraction of states where `q_policy` picks the same action as `reference_policy`.

    Only states present in both dicts are scored. Terminal/wall cells are
    typically absent from both and therefore ignored automatically. Returns
    0.0 when there are no comparable states.
    """
    shared = [s for s in q_policy if s in reference_policy]
    if not shared:
        return 0.0
    matches = sum(1 for s in shared if int(q_policy[s]) == int(reference_policy[s]))
    return matches / len(shared)
