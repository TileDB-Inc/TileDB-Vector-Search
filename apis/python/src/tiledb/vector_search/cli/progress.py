"""CLI progress bar (stderr), shared by ``vs build`` and ``vs eval``."""

from __future__ import annotations

import sys
import time

BAR_WIDTH = 30


def progress_bar(current: int, total: int, label: str, start_time: float) -> None:
    """Write a single-line progress bar to stderr."""
    elapsed = time.monotonic() - start_time
    frac = current / total if total else 1.0
    filled = int(BAR_WIDTH * frac)
    bar = "█" * filled + "░" * (BAR_WIDTH - filled)
    pct = int(frac * 100)
    sys.stderr.write(f"\r  {label} [{current}/{total}] {bar} {pct}%  {elapsed:.0f}s")
    sys.stderr.flush()


def progress_done(current: int, total: int, label: str, start_time: float) -> None:
    elapsed = time.monotonic() - start_time
    bar = "█" * BAR_WIDTH
    sys.stderr.write(f"\r  ✓ {label} [{total}/{total}] {bar} 100%  {elapsed:.1f}s\n")
    sys.stderr.flush()
