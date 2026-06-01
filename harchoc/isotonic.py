from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class IsotonicModel:
    """
    Monotone non-decreasing piecewise-constant function fitted by PAVA.

    `x` is strictly increasing (breakpoints) and `y` are fitted values at those points.
    """

    x: tuple[float, ...]
    y: tuple[float, ...]

    def predict_one(self, x: float) -> float:
        if not self.x:
            return 0.0
        # clamp
        if x <= self.x[0]:
            return float(self.y[0])
        if x >= self.x[-1]:
            return float(self.y[-1])
        # binary search
        lo, hi = 0, len(self.x) - 1
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if x < self.x[mid]:
                hi = mid
            else:
                lo = mid
        return float(self.y[lo])

    def predict(self, xs: Iterable[float]) -> list[float]:
        return [self.predict_one(float(x)) for x in xs]


def fit_isotonic_pava(*, scores: Iterable[float], targets: Iterable[float]) -> IsotonicModel:
    """
    Fit isotonic regression with the pool-adjacent-violators algorithm (PAVA).

    - `scores`: x values (need not be sorted)
    - `targets`: y values (e.g. 0/1)
    Returns a monotone non-decreasing piecewise-constant model.
    """
    pairs = [(float(x), float(y)) for x, y in zip(scores, targets, strict=False)]
    if not pairs:
        return IsotonicModel(x=(), y=())
    pairs.sort(key=lambda t: t[0])

    # Start with each point as its own block.
    block_x: list[float] = []
    block_y: list[float] = []
    block_w: list[float] = []

    for x, y in pairs:
        block_x.append(x)
        block_y.append(y)
        block_w.append(1.0)

        # Merge backwards while violating monotonicity.
        while len(block_y) >= 2 and block_y[-2] > block_y[-1]:
            w1, w2 = block_w[-2], block_w[-1]
            y1, y2 = block_y[-2], block_y[-1]
            # weighted average
            y_new = (w1 * y1 + w2 * y2) / (w1 + w2)
            x_new = block_x[-1]  # keep rightmost x for the merged block
            block_w[-2] = w1 + w2
            block_y[-2] = y_new
            block_x[-2] = x_new
            block_w.pop()
            block_y.pop()
            block_x.pop()

    # Expand to a compact set of breakpoints (strictly increasing x).
    xs: list[float] = []
    ys: list[float] = []
    for x, y in zip(block_x, block_y, strict=False):
        if xs and x <= xs[-1]:
            # ensure strictly increasing breakpoints
            continue
        xs.append(float(x))
        ys.append(float(y))
    return IsotonicModel(x=tuple(xs), y=tuple(ys))

