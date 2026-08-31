from __future__ import annotations

import time


class ProgressTracker:
    def __init__(self, total_steps: int) -> None:
        self.total_steps = max(1, total_steps)
        self.start_time = time.time()
        self.last_time = self.start_time

    def update(self, step: int, prefix: str) -> str:
        now = time.time()
        elapsed = now - self.start_time
        avg_per_step = elapsed / max(step, 1)
        remaining_steps = max(self.total_steps - step, 0)
        eta_seconds = remaining_steps * avg_per_step
        self.last_time = now
        return (
            f"{prefix} step={step}/{self.total_steps} "
            f"elapsed={elapsed:.1f}s eta={eta_seconds:.1f}s avg_step={avg_per_step:.3f}s"
        )
