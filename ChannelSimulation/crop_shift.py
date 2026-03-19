from __future__ import annotations

import numpy as np


def apply_random_crop_and_time_shift(
    x: np.ndarray,
    min_keep_ratio: float = 0.7,
    max_shift: int = 300,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """随机裁剪 + 时间偏移，并保持输出长度不变（零填充）。"""
    if rng is None:
        rng = np.random.default_rng()

    n = len(x)
    keep_ratio = float(rng.uniform(min_keep_ratio, 1.0))
    keep_len = max(16, int(n * keep_ratio))

    start = int(rng.integers(0, max(1, n - keep_len + 1)))
    seg = x[start : start + keep_len]

    y = np.zeros_like(x)
    y[:keep_len] = seg

    shift = int(rng.integers(-max_shift, max_shift + 1))
    y = np.roll(y, shift)

    if shift > 0:
        y[:shift] = 0
    elif shift < 0:
        y[shift:] = 0

    return y
