from __future__ import annotations

import numpy as np

EPS = 1e-12


def normalize_power(
    x: np.ndarray,
    target_power: float = 1.0,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Normalise complex IQ signal to target average power.

    Supports in-place operation via the `out` argument to avoid allocating
    a full copy when called repeatedly across all pipeline stages.
    """
    p = float(np.mean(np.abs(x) ** 2))
    if p < EPS:
        return x
    scale = np.sqrt(target_power / p)
    if out is None:
        return x * scale
    np.multiply(x, scale, out=out)
    return out
