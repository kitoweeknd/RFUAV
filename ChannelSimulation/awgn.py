from __future__ import annotations

import numpy as np

try:
    from .common import EPS
except ImportError:  # 兼容直接脚本方式
    from common import EPS


def apply_awgn(x: np.ndarray, snr_db: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """按目标 SNR 加复高斯白噪声。"""
    if rng is None:
        rng = np.random.default_rng()

    sig_power = float(np.mean(np.abs(x) ** 2))
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = sig_power / max(snr_linear, EPS)

    noise = (rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x))) * np.sqrt(noise_power / 2.0)
    return x + noise
