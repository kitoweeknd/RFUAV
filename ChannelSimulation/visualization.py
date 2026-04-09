from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import stft

try:
    from .common import EPS
except ImportError:
    from common import EPS


MAX_PLOT_POINTS = 2_000_000  # max points in time-domain trace
MAX_STFT_COLS = 3_000        # max columns (time frames) in spectrogram


def _decimate(x: np.ndarray, max_points: int) -> np.ndarray:
    """Simple decimation: keep every nth element so result fits in max_points."""
    n = len(x)
    if n <= max_points:
        return x
    step = max(1, n // max_points)
    return x[::step]


def _decimate_2d(z: np.ndarray, max_cols: int) -> np.ndarray:
    """Decimate a 2D array along axis 1 (time axis of STFT output)."""
    if z.shape[1] <= max_cols:
        return z
    step = max(1, z.shape[1] // max_cols)
    return z[:, ::step]


def plot_time_frequency(
    x: np.ndarray,
    fs: float,
    stage_name: str,
    out_dir: str | Path,
    nperseg: int = 256,
    noverlap: int = 192,
) -> None:
    """Save time-frequency plot for each stage.

    Both the time-domain envelope and the STFT spectrogram are decimated
    before rendering to keep matplotlib memory usage bounded.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(x)

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
    f_shifted = np.fft.fftshift(f)
    spec_db_shifted = np.fft.fftshift(spec_db, axes=0)

    # Decimate spectrogram columns before pcolormesh (this is the memory culprit)
    spec_plot = _decimate_2d(spec_db_shifted, MAX_STFT_COLS)
    tt_plot = _decimate(tt, MAX_STFT_COLS)

    # Decimate time-domain trace (not a memory issue but avoids slow rendering)
    x_plot = _decimate(np.abs(x), MAX_PLOT_POINTS)
    t_plot = np.arange(len(x_plot)) / fs

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)

    axes[0].plot(t_plot, x_plot, linewidth=0.9)
    axes[0].set_title(f"{stage_name} - Time Domain Envelope")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("|x(t)|")
    axes[0].grid(alpha=0.3)

    im = axes[1].pcolormesh(tt_plot, f_shifted / 1e6, spec_plot, shading="auto", cmap="turbo")
    axes[1].set_title(f"{stage_name} - Time-Frequency Plot (STFT)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Frequency (MHz)")
    fig.colorbar(im, ax=axes[1], label="Magnitude (dB)")

    save_name = stage_name.replace(" ", "_").replace("/", "_") + ".png"
    fig.savefig(out_dir / save_name, dpi=150)
    plt.close(fig)
