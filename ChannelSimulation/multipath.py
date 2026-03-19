from __future__ import annotations

import numpy as np

try:
    from .common import EPS, normalize_power
except ImportError:  # 兼容直接脚本方式
    from common import EPS, normalize_power


def apply_multipath(
    x: np.ndarray,
    max_delay_samp: int = 20,
    n_paths: int = 4,
    exp_decay: float = 0.35,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """整数采样多径信道（随机时延 + 指数衰减 + 随机相位）。"""
    if rng is None:
        rng = np.random.default_rng()

    if n_paths < 1:
        return x

    delays = np.sort(rng.integers(0, max_delay_samp + 1, size=n_paths))
    rel = np.exp(-exp_decay * delays)
    rel = rel / (np.linalg.norm(rel) + EPS)

    phases = rng.uniform(0.0, 2.0 * np.pi, size=n_paths)
    taps = rel * np.exp(1j * phases)

    h = np.zeros(max_delay_samp + 1, dtype=np.complex128)
    for d, t in zip(delays, taps):
        h[d] += t

    y_full = np.convolve(x, h, mode="full")
    y = y_full[: len(x)]
    return normalize_power(y)
