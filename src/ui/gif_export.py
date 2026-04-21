"""GIF recording utility using imageio + pygame surfaces.

The recorder buffers RGB frames in memory while recording is active and
flushes them to disk with `save()`. Frames are intentionally small (the
current pygame surface size), so 500-frame episodes fit comfortably.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pygame


class GifRecorder:
    def __init__(self, *, fps: int = 8) -> None:
        self.fps = fps
        self._frames: list[np.ndarray] = []
        self._recording = False

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    def start(self) -> None:
        self._frames = []
        self._recording = True

    def stop(self) -> None:
        self._recording = False

    def toggle(self) -> bool:
        if self._recording:
            self.stop()
        else:
            self.start()
        return self._recording

    def capture(self, surface: "pygame.Surface") -> None:
        if not self._recording:
            return
        import pygame

        arr = pygame.surfarray.array3d(surface)
        # pygame gives (W, H, 3); imageio expects (H, W, 3).
        arr = np.transpose(arr, (1, 0, 2))
        self._frames.append(arr)

    def save(self, path: Path) -> Path | None:
        """Flush captured frames to `path` as a GIF. Returns the path or None."""
        if not self._frames:
            return None
        import imageio.v2 as imageio

        path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(
            str(path),
            self._frames,
            format="GIF",
            duration=1.0 / max(1, self.fps),
            loop=0,
        )
        out = path
        self._frames = []
        self._recording = False
        return out
