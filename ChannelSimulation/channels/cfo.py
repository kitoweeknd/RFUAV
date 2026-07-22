from __future__ import annotations

import numpy as np

# Process per-sample rotation in chunks. Doing it whole-array at N=100M meant
# ~5 separate 800MB float64 temporaries (arange/phase/cos/sin) — ~4GB that tips
# the machine into memory compression and turned a ~2s op into ~30s. Chunking
# keeps temporaries at a few MB (cache-resident) so the cost is linear in N.
_CHUNK = 1 << 21  # ~2.1M samples


def apply_cfo(
    x: np.ndarray,
    fs: float,
    cfo_hz: float,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Carrier Frequency Offset (CFO).

    Applies a phase ramp exp(j·2π·cfo·t/fs) in cache-sized chunks. The output
    keeps the input's complex precision (complex64 in the pipeline).
    """
    cdt = x.dtype if np.issubdtype(x.dtype, np.complexfloating) else np.complex64
    n = len(x)
    y = np.empty(n, dtype=cdt) if out is None else np.ascontiguousarray(out, dtype=cdt)

    w = (2.0 * np.pi / fs) * cfo_hz
    for s in range(0, n, _CHUNK):
        e = min(s + _CHUNK, n)
        # Phase stays float64 for precision (t can reach tens of millions).
        ph = np.arange(s, e, dtype=np.float64) * w
        rot = np.empty(e - s, dtype=cdt)
        rot.real = np.cos(ph)
        rot.imag = np.sin(ph)
        y[s:e] = x[s:e] * rot
    return y
