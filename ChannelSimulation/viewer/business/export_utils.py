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
