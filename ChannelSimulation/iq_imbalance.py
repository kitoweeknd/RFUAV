from __future__ import annotations

import numpy as np

try:
    from .common import normalize_power
except ImportError:  # Compatible with direct script execution
    from common import normalize_power


def apply_iq_imbalance(
    x: np.ndarray,
    gain_imbalance_db: float = 1.0,
    phase_imbalance_deg: float = 3.0,
) -> np.ndarray:
    """Generalized IQ imbalance model: y = alpha*x + beta*conj(x)."""
    g = 10 ** (gain_imbalance_db / 20.0)
    phi = np.deg2rad(phase_imbalance_deg)

    alpha = 0.5 * (1.0 + g * np.exp(-1j * phi))
    beta = 0.5 * (1.0 - g * np.exp(1j * phi))

    y = alpha * x + beta * np.conj(x)
    return normalize_power(y)
