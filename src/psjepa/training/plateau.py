"""Stop when the curve flattens instead of when a step counter runs out.

A fixed step budget answers a question nobody asked. The region arms were given
3125 steps and every one of them was still descending when the counter expired —
`blocks_1` peaked at 1.33 around step 1250, then fell to 1.05 by 3125 with the
unseen half going 0.744 -> 0.647 and rank flat at 75 for the last third. Reading
a final number off that is reading a curve mid-fall.

Two details make this harder than a textbook early stop.

*The metric is noisy.* Consecutive validation points on the same arm came in at
1.109, 0.916, 1.141, 0.982, 1.046 — a spread of 0.23 on a mean near 1.0, which is
larger than the improvement expected over several hundred steps. Testing raw
values would fire on noise, so the test runs on the mean of a trailing window.

*A falling loss is not always progress.* This objective's loss also falls when
the teacher collapses, which is why `collapsed` gates checkpoint selection.
Collapsed points are refused here for the same reason: a collapse would otherwise
look like the best value ever seen and freeze the plateau counter forever.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class PlateauStopper:
    """Decides, at each validation point, whether to continue, decay, or stop.

    `patience` counts validation points that fail to improve the windowed mean by
    `min_delta` relative. On running out of patience the learning rate is decayed
    `lr_reductions` times before the run is allowed to stop, so a curve that has
    only flattened for the current step size gets a chance to move again at a
    smaller one — the plateau that matters is the one that survives the decay.
    """

    patience: int = 6
    min_delta: float = 0.01
    min_steps: int = 0
    window: int = 3
    lr_reductions: int = 2
    lr_factor: float = 0.3

    best: float = float("inf")
    waited: int = 0
    reductions: int = 0
    level: float = 1.0
    history: deque = field(default_factory=deque)

    def __post_init__(self) -> None:
        if self.window < 1:
            raise ValueError("window must be at least 1")
        if self.patience < 1:
            raise ValueError("patience must be at least 1")
        self.history = deque(maxlen=self.window)

    def update(self, step: int, value: float, is_collapsed: bool = False) -> str:
        """One of ``continue``, ``reduce``, ``stop``.

        ``reduce`` means the caller multiplies its peak learning rate by
        ``self.level``, which this method has already updated.
        """
        if is_collapsed:
            # Not evidence about the plateau either way: the loss is low because
            # the representation went flat, so counting it would let a collapse
            # end the run early or reset the patience it should not touch.
            return "continue"
        self.history.append(float(value))
        if len(self.history) < self.window:
            return "continue"
        current = sum(self.history) / len(self.history)
        if current < self.best * (1.0 - self.min_delta):
            self.best = current
            self.waited = 0
            return "continue"
        # A window that is worse than the best still counts toward patience, but
        # the best is lowered anyway when it improves at all, so a long slow
        # descent below the min_delta threshold cannot be mistaken for progress.
        self.best = min(self.best, current)
        self.waited += 1
        if self.waited < self.patience or step < self.min_steps:
            return "continue"
        self.waited = 0
        if self.reductions < self.lr_reductions:
            self.reductions += 1
            self.level *= self.lr_factor


            self.history.clear()
            self.best = float("inf")
            return "reduce"
        return "stop"

    def state(self) -> dict[str, float | int]:
        return {
            "plateau_best": self.best if self.best != float("inf") else None,
            "plateau_waited": self.waited,
            "plateau_reductions": self.reductions,
            "lr_level": self.level,
        }
