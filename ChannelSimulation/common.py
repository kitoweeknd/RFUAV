from __future__ import annotations

import numpy as np

EPS = 1e-12


def normalize_power(x: np.ndarray, target_power: float = 1.0) -> np.ndarray:
    p = float(np.mean(np.abs(x) ** 2))
    if p < EPS:
        return x
    return x * np.sqrt(target_power / p)
