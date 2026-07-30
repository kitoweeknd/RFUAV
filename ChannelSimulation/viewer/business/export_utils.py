"""
Export Utilities
~~~~~~~~~~~~~~~~

Export IQ data and spectrum PNG files.
Supports exporting IQ data from intermediate stages.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class ExportUtils:
    """Export utilities class"""

    @staticmethod
    def export_iq(
        signal: np.ndarray,
        file_path: str | Path,
        format: str = "float32",
    ) -> None:
        """Export IQ data as interleaved samples (I, Q, I, Q, …).

        Args:
            signal: IQ signal (complex64 or complex128)
            file_path: Target file path
            format: Output sample format — "float32" (default) writes raw
                interleaved float32; "int16" clips to [-1, 1] and scales to
                full-scale signed 16-bit (rounded to nearest).
        """
        if format not in ("float32", "int16"):
            raise ValueError(
                f"Unsupported IQ export format: {format!r} "
                "(expected 'float32' or 'int16')"
            )

        file_path = Path(file_path)

        if format == "float32":
            # Delegate to the canonical on-disk writer, so the GUI export and
            # the backend pipeline can never drift apart in file layout.
            from ChannelSimulation.core.io_utils import save_interleaved_iq

            save_interleaved_iq(file_path, signal)
            return

        # int16: full-scale signed-16-bit convention. Build the interleaved
        # float32 buffer once (complex64 and complex128 inputs both narrow to
        # float32 components on assignment), then sanitise/clip/scale/round it
        # in place — no per-component temporaries — before the final int16
        # cast. NaNs are zeroed first: they pass np.clip unchanged and would
        # otherwise cast to platform-dependent garbage. Rounding to nearest
        # (instead of the cast's truncation toward zero) avoids a systematic
        # up-to-1-LSB magnitude bias in the exported samples.
        file_path.parent.mkdir(parents=True, exist_ok=True)
        interleaved = np.empty(len(signal) * 2, dtype=np.float32)
        interleaved[0::2] = np.real(signal)
        interleaved[1::2] = np.imag(signal)
        np.nan_to_num(interleaved, copy=False, nan=0.0)
        np.clip(interleaved, -1.0, 1.0, out=interleaved)
        interleaved *= np.float32(32767.0)
        np.round(interleaved, out=interleaved)
        interleaved.astype(np.int16).tofile(file_path)

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
