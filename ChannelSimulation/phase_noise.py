from __future__ import annotations

import numpy as np


def apply_phase_noise(
    x: np.ndarray,
    phase_std_per_sample: float = 2e-3,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """相位噪声：离散 Wiener 过程（随机游走）。"""
    if rng is None:
        rng = np.random.default_rng()

    dphi = rng.normal(0.0, phase_std_per_sample, size=len(x))
    phi = np.cumsum(dphi)
    return x * np.exp(1j * phi)
