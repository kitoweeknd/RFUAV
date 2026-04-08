from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import stft

try:
    from .common import EPS
except ImportError:  # Compatible with direct script execution
    try:
    from .common import EPS
except ImportError:  # Compatible with direct script execution
    from common import EPS


def plot_time_frequency(
    x: np.ndarray,
    fs: float,
    stage_name: str,
    out_dir: str | Path,
    nperseg: int = 256,
    noverlap: int = 192,
) -> None:
    """Save time-frequency plot for each stage (including time-domain envelope + spectrogram)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(x)
    t = np.arange(n) / fs

    f, tt, zxx = stft(
        x,
        fs=fs,
        nperseg=min(nperseg, n),
        noverlap=min(noverlap, max(0, min(nperseg, n) - 1)),
        return_onesided=False,
        boundary=None,
        padded=False,
    )

    spec_db = 20.0 * np.log10(np.abs(zxx) + EPS)
    f = np.fft.fftshift(f)
    spec_db = np.fft.fftshift(spec_db, axes=0)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)

    axes[0].plot(t, np.abs(x), linewidth=0.9)
    axes[0].set_title(f"{stage_name} - Time Domain Envelope")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("|x(t)|")
    axes[0].grid(alpha=0.3)

    im = axes[1].pcolormesh(tt, f / 1e6, spec_db, shading="auto", cmap="turbo")
    axes[1].set_title(f"{stage_name} - Time-Frequency Plot (STFT)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Frequency (MHz)")
    fig.colorbar(im, ax=axes[1], label="Magnitude (dB)")

    save_name = stage_name.replace(" ", "_").replace("/", "_") + ".png"
    fig.savefig(out_dir / save_name, dpi=150)
    plt.close(fig)
