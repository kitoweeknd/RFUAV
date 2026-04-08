from __future__ import annotations

import numpy as np

try:
    from .common import normalize_power
except ImportError:  # Compatible with direct script execution
    from common import normalize_power


def apply_rayleigh_or_rician_fading(
    x: np.ndarray,
    fading_type: str = "rayleigh",
    k_factor: float = 4.0,
    block_size: int = 256,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Flat fading: supports Rayleigh / Rician (block fading approximation)."""
    if rng is None:
        rng = np.random.default_rng()

    n = len(x)
    n_blocks = int(np.ceil(n / block_size))

    h_nlos = (rng.standard_normal(n_blocks) + 1j * rng.standard_normal(n_blocks)) / np.sqrt(2.0)

    if fading_type.lower() == "rician":
        los_phase = rng.uniform(0.0, 2.0 * np.pi, size=n_blocks)
        h_los = np.exp(1j * los_phase)
        k = max(k_factor, 0.0)
        h_blocks = np.sqrt(k / (k + 1.0)) * h_los + np.sqrt(1.0 / (k + 1.0)) * h_nlos
    elif fading_type.lower() == "rayleigh":
        h_blocks = h_nlos
    else:
        raise ValueError("fading_type only supports 'rayleigh' or 'rician'")

    h = np.repeat(h_blocks, block_size)[:n]
    y = x * h
    return normalize_power(y)
