"""
Export Utilities
~~~~~~~~~~~~~~~~

Export IQ data and spectrum PNG files.
Supports exporting IQ data from intermediate stages.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


class ExportUtils:
    """Export utilities class"""

    @staticmethod
    def export_iq(
        signal: np.ndarray,
        file_path: str | Path,
        format: str = "float32",
    ) -> None:
        """Export IQ data

        Args:
            signal: IQ signal (complex128 or complex64)
            file_path: Target file path
            format: Output format (float32 / int16)
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to interleaved format
        if signal.dtype == np.complex64:
            real = np.real(signal).astype(np.float32)
            imag = np.imag(signal).astype(np.float32)
        elif signal.dtype == np.complex128:
            real = np.real(signal).astype(np.float32)
            imag = np.imag(signal).astype(np.float32)
        else:
            real = np.real(signal).astype(np.float32)
            imag = np.imag(signal).astype(np.float32)

        # Interleaved storage
        interleaved = np.empty(len(signal) * 2, dtype=np.float32)
        interleaved[0::2] = real
        interleaved[1::2] = imag

        # Write to file
        interleaved.tofile(file_path)

    @staticmethod
    def export_figure(
        fig,
        file_path: str | Path,
        dpi: int = 300,
        transparent: bool = False,
    ) -> None:
        """Export Matplotlib figure

        Args:
            fig: Matplotlib Figure object
            file_path: Target file path
            dpi: DPI setting
            transparent: Whether to use transparent background
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(
            file_path,
            dpi=dpi,
            bbox_inches="tight",
            transparent=transparent,
        )

    @staticmethod
    def export_spectrum_png(
        image_data: np.ndarray,
        file_path: str | Path,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        extent: list | None = None,
        cmap: str = "turbo",
        dpi: int = 300,
    ) -> None:
        """Export spectrum PNG

        Args:
            image_data: Image data (2D array)
            file_path: Target file path
            title: Title
            xlabel: X-axis label
            ylabel: Y-axis label
            extent: Coordinate range [xmin, xmax, ymin, ymax]
            cmap: Colormap name
            dpi: DPI setting
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[ERROR] matplotlib is not installed, cannot export PNG")
            return

        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(
            image_data,
            aspect="auto",
            origin="lower",
            cmap=cmap,
            extent=extent,
        )

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        fig.colorbar(im, ax=ax, label="Magnitude (dB)")
        fig.tight_layout()

        fig.savefig(file_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def export_batch(
        stage_signals: dict[str, np.ndarray],
        output_dir: str | Path,
        format: str = "float32",
    ) -> dict[str, str]:
        """Batch export IQ data for multiple stages

        Args:
            stage_signals: Dictionary mapping stage names to signals
            output_dir: Output directory
            format: Output format

        Returns:
            Dictionary mapping stage names to file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_files = {}

        for stage_name, signal in stage_signals.items():
            file_path = output_dir / f"{stage_name}.iq"
            ExportUtils.export_iq(signal, file_path, format)
            output_files[stage_name] = str(file_path)

        return output_files


# ============================================================================
# File Format Auto-Detection
# ============================================================================


def detect_and_load(file_path: str | Path) -> np.ndarray:
    """Auto-detect file format and load

    Args:
        file_path: File path

    Returns:
        IQ data array

    Raises:
        ValueError: Unsupported file format
    """
    file_path = Path(file_path)
    ext = file_path.suffix.lower()

    if ext in (".iq", ".cf32"):
        return load_iq_float32(file_path)

    elif ext in (".raw", ".bin", ".cfile"):
        return load_raw_binary(file_path)

    elif ext in (".iq16", ".s16"):
        return load_int16(file_path)

    elif ext == ".npy":
        return load_numpy(file_path)

    elif ext == ".npz":
        return load_numpy_zip(file_path)

    elif ext == ".mat":
        return load_matlab(file_path)

    else:
        # Try float32 as fallback
        return load_iq_float32(file_path)


def load_iq_float32(file_path: Path) -> np.ndarray:
    """Load float32 interleaved IQ file"""
    raw = np.fromfile(file_path, dtype=np.float32)
    if raw.size % 2 == 1:
        raw = raw[:-1]
    return raw[0::2] + 1j * raw[1::2]


def load_raw_binary(file_path: Path, dtype=np.float32) -> np.ndarray:
    """Load raw binary file"""
    raw = np.fromfile(file_path, dtype=dtype)
    if raw.size % 2 == 1:
        raw = raw[:-1]
    return raw[0::2] + 1j * raw[1::2]


def load_int16(file_path: Path) -> np.ndarray:
    """Load int16 interleaved IQ file"""
    raw = np.fromfile(file_path, dtype=np.int16)
    if raw.size % 2 == 1:
        raw = raw[:-1]
    # Convert to float and normalize
    real = raw[0::2].astype(np.float32) / 32768.0
    imag = raw[1::2].astype(np.float32) / 32768.0
    return real + 1j * imag


def load_numpy(file_path: Path) -> np.ndarray:
    """Load NumPy .npy file"""
    data = np.load(file_path)
    # If real-valued, assume interleaved format
    if data.dtype in (np.float32, np.float64, np.int16, np.int32):
        if data.ndim == 1 and data.size % 2 == 0:
            return data[0::2] + 1j * data[1::2]
    # If complex
    if data.dtype in (np.complex64, np.complex128):
        return data
    raise ValueError(f"Unsupported NumPy data type: {data.dtype}")


def load_numpy_zip(file_path: Path) -> np.ndarray:
    """Load NumPy .npz file"""
    data = np.load(file_path)
    # Get the first array
    if len(data.files) == 0:
        raise ValueError("npz file is empty")
    key = data.files[0]
    return data[key]


def load_matlab(file_path: Path) -> np.ndarray:
    """Load MATLAB .mat file"""
    try:
        from scipy.io import loadmat
    except ImportError:
        raise ImportError("scipy is not installed, cannot load .mat file")

    data = loadmat(file_path)
    # Search for complex signal array
    for key in ["signal", "iq", "x", "data"]:
        if key in data and isinstance(data[key], np.ndarray):
            arr = data[key].flatten()
            if arr.dtype in (np.complex64, np.complex128):
                return arr
            elif arr.dtype in (np.float32, np.float64) and len(arr) % 2 == 0:
                return arr[0::2] + 1j * arr[1::2]

    raise ValueError("Could not find IQ signal data in .mat file")
