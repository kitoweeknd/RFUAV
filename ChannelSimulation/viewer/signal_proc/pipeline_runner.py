"""
Pipeline Runner
~~~~~~~~~~~~~~

Wraps the backend channel processing module, providing a single pipeline entry point.
Supports step-by-step output of intermediate signals by stage, and memmap reading for large files.

Memory optimisation notes
~~~~~~~~~~~~~~~~~~~~~~~~~
- Internal precision is float32 (complex64) to halve memory vs complex128.
- The pool stores zero-filled reusable scratch buffers; callers receive views.
- Stage signals are stored as views into the primary array wherever possible
  to avoid making extra copies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# Pool & memory management
from .array_pool import get_pool

try:
    from ChannelSimulation.channel_pipeline import STAGE_ORDER as _STAGE_ORDER
    from ChannelSimulation.core.io_utils import read_interleaved_iq
    from ChannelSimulation.core.common import normalize_power
except ImportError:
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).parent.parent.parent))
    from ChannelSimulation.channel_pipeline import STAGE_ORDER as _STAGE_ORDER
    from ChannelSimulation.core.io_utils import read_interleaved_iq
    from ChannelSimulation.core.common import normalize_power


# Internal working dtype — float32 complex is sufficient for signal processing.
# This halves the memory footprint of intermediate results vs complex128.
_WORKING_DTYPE = np.complex64


# Stage order (consistent with backend)
STAGE_ORDER = [
    "original",
    "fading",
    "doppler",
    "multipath",
    "cfo",
    "phase_noise",
    "iq_imbalance",
    "awgn",
    "crop_time_shift",
]

# Default stage configurations
DEFAULT_STAGE_CONFIGS: dict[str, dict[str, Any]] = {
    "fading": {
        "enabled": False,
        "method": "rayleigh",
        "params": {
            "k_factor": 5.0,
            "block_size": 256,
        },
    },
    "doppler": {
        "enabled": False,
        "method": "jakes",
        "params": {
            "doppler_shift_hz": 0.0,
            "doppler_spread_hz": 100.0,
        },
    },
    "multipath": {
        "enabled": False,
        "method": "tdl",
        "params": {
            "max_delay_samp": 20,
            "n_paths": 4,
            "exp_decay": 0.35,
        },
    },
    "cfo": {
        "enabled": False,
        "method": "constant",
        "params": {
            "cfo_hz": 0.0,
        },
    },
    "phase_noise": {
        "enabled": False,
        "method": "wiener",
        "params": {
            "phase_std_per_sample": 0.002,
        },
    },
    "iq_imbalance": {
        "enabled": False,
        "method": "widely_linear",
        "params": {
            "gain_imbalance_db": 1.0,
            "phase_imbalance_deg": 3.0,
        },
    },
    "awgn": {
        "enabled": False,
        "method": "gaussian",
        "params": {
            "snr_db": 15.0,
        },
    },
    "crop_time_shift": {
        "enabled": False,
        "method": "basic",
        "params": {
            "min_keep_ratio": 0.7,
            "max_shift": 300,
        },
    },
}

# Preprocessing default values
DEFAULT_PREPROCESS_CONFIG: dict[str, dict[str, Any]] = {
    "dc_removal": {"enabled": True},
    "digital_filter": {
        "enabled": False,
        "type": "lowpass",
        "filter_class": "butterworth",
        "cutoff_hz": 50_000_000,
        "order": 8,
    },
    "resampling": {
        "enabled": False,
        "target_fs": 200_000_000,
        "filter_type": "firls",
    },
    "gain_control": {
        "enabled": False,
        "gain_db": 0.0,
    },
}


class PipelineRunner:
    """Pipeline Runner

    Wraps the backend channel processing module, providing:
    - Step-by-step output of intermediate signals by stage
    - Memmap reading for large files
    - Configuration validation and default value filling
    """

    def __init__(
        self,
        iq_data: np.ndarray | None = None,
        fs: float = 100_000_000,
        fc: float = 0.0,
        seed: int = 42,
    ):
        self._iq_data: np.ndarray | None = iq_data
        self._fs = fs
        self._fc = fc
        self._seed = seed

        # Intermediate signal storage
        self._stage_signals: dict[str, np.ndarray] = {}
        self._preprocess_signal: np.ndarray | None = None
        self._preprocess_power: float | None = None

        # Stage configuration
        self._stage_configs: dict[str, dict[str, Any]] = {}
        self._preprocess_config: dict[str, dict[str, Any]] = {}

        # Initialize default configuration
        self.reset_config()

    # =========================================================================
    # Configuration Management
    # =========================================================================

    def reset_config(self) -> None:
        """Reset configuration to default values"""
        self._stage_configs = {k: dict(v) for k, v in DEFAULT_STAGE_CONFIGS.items()}
        self._preprocess_config = {k: dict(v) for k, v in DEFAULT_PREPROCESS_CONFIG.items()}

    def set_config(self, config: dict[str, Any]) -> None:
        """Set complete configuration"""
        # Parse stage configuration
        if "stage_configs" in config:
            for stage_name, stage_cfg in config["stage_configs"].items():
                if stage_name in self._stage_configs:
                    self._stage_configs[stage_name] = self._merge_stage_config(
                        self._stage_configs[stage_name], stage_cfg
                    )

        # Parse preprocessing configuration
        if "preprocess" in config:
            self._preprocess_config = config["preprocess"]

        # Parse basic information
        if "fs" in config:
            self._fs = float(config["fs"])
        if "fc" in config:
            self._fc = float(config["fc"])
        if "seed" in config:
            self._seed = int(config["seed"])

    def get_config(self) -> dict[str, Any]:
        """Get current configuration"""
        return {
            "stage_configs": dict(self._stage_configs),
            "preprocess": dict(self._preprocess_config),
            "fs": self._fs,
            "fc": self._fc,
            "seed": self._seed,
        }

    def _merge_stage_config(
        self,
        base: dict[str, Any],
        update: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge stage configuration"""
        result = dict(base)

        # Update enabled
        if "enabled" in update:
            result["enabled"] = bool(update["enabled"])

        # Update method
        if "method" in update:
            result["method"] = str(update["method"])

        # Update params
        if "params" in update:
            result["params"] = {**result.get("params", {}), **update["params"]}

        return result

    # =========================================================================
    # Data Loading
    # =========================================================================

    def load_iq_file(
        self,
        file_path: str | Path,
        dtype: type = np.float32,
    ) -> int:
        """Load IQ file

        Args:
            file_path: File path
            dtype: Data type

        Returns:
            Number of samples
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_size = file_path.stat().st_size
        if file_size == 0:
            raise ValueError(f"File is empty: {file_path}")

        # float32: 8 bytes per sample (2 x 4)
        sample_size = 8
        n_samples = file_size // sample_size

        if n_samples == 0:
            raise ValueError(f"File contains no valid samples: {file_path}")

        # For large files (>100MB), use memory mapping
        if file_size > 100 * 1024 * 1024:
            self._iq_data = self._load_memmap(file_path, dtype)
        else:
            self._iq_data = read_interleaved_iq(file_path, dtype)

        if len(self._iq_data) == 0:
            raise ValueError(f"Failed to load IQ data from file: {file_path}")

        return n_samples

    def _load_memmap(
        self,
        file_path: Path,
        dtype: type,
    ) -> np.ndarray:
        """Memory-map large file, converting to float32 complex directly.

        Uses zero-copy strided views: real and imag are extracted as slices
        (no new array is allocated for the raw interleaved data).
        """
        # Map the raw interleaved data.
        raw: np.ndarray = np.memmap(file_path, dtype=dtype, mode="r")
        n = raw.size & ~1  # round down to even (ignore last sample if odd)
        if n == 0:
            return np.array([], dtype=_WORKING_DTYPE)
        if raw.size != n:
            raw = raw[:n]

        # Expose real/imag as views (strided slices share the same buffer).
        real_v = raw[0::2].view(np.float32)
        imag_v = raw[1::2].view(np.float32)

        # Assemble into complex64 directly in a pre-allocated pool buffer.
        out = get_pool().acquire((n // 2,), _WORKING_DTYPE, zero=False)
        out[:len(real_v)] = real_v + 1j * imag_v
        return out  # already complex64

    def set_iq_data(self, iq_data: np.ndarray) -> None:
        """Set IQ data directly"""
        self._iq_data = iq_data

    # =========================================================================
    # Signal Processing
    # =========================================================================

    def run(
        self,
        return_all_stages: bool = False,
    ) -> dict[str, np.ndarray]:
        """Run pipeline

        Args:
            return_all_stages: Whether to return signals from all intermediate stages

        Returns:
            Dictionary containing signals for each stage
        """
        if self._iq_data is None:
            raise ValueError("IQ data not set. Please load a file or set data first.")

        results: dict[str, np.ndarray] = {}
        rng = np.random.default_rng(self._seed)

        # ── Load & convert to float32 complex ──────────────────────────────────
        # Complex64 conversion: if source is already complex64 the view is returned
        # without any copy; otherwise a new array is allocated.
        x = self._iq_data.astype(_WORKING_DTYPE)
        x = normalize_power(x)
        results["original"] = x

        # ── Preprocessing (DC removal in-place) ─────────────────────────────────
        x = self._apply_preprocess(x)
        results["preprocess"] = x
        self._preprocess_signal = x
        # Record preprocessed signal power as the reference for all AWGN stages.
        # Each AWGN stage uses this reference to compute noise power, so that
        # setting SNR=15dB always degrades to 15dB relative to the original
        # signal (after preprocessing) rather than accumulating noise.
        self._preprocess_power = float(np.mean(np.abs(x) ** 2))

        # ── Pipeline stages (float32 throughout) ────────────────────────────────
        for stage_name in STAGE_ORDER[1:]:
            if stage_name not in self._stage_configs:
                continue

            cfg = self._stage_configs[stage_name]
            if not cfg.get("enabled", False):
                continue

            x = self._apply_stage(stage_name, x, rng)
            # Store as view — no extra copy; PipelineRunner holds the backing array.
            self._stage_signals[stage_name] = x
            # When requested, also surface each intermediate stage in the returned
            # dict so callers (the viewer's per-stage inspector / spectrum precompute)
            # can see every enabled stage — not just preprocess/final.
            if return_all_stages:
                results[stage_name] = x

        results["final"] = x
        return results

    def _apply_preprocess(self, x: np.ndarray) -> np.ndarray:
        """Apply preprocessing"""
        # DC removal
        if self._preprocess_config.get("dc_removal", {}).get("enabled", True):
            x = self._remove_dc(x)

        # Gain control
        gain_cfg = self._preprocess_config.get("gain_control", {})
        if gain_cfg.get("enabled", False):
            gain_db = gain_cfg.get("gain_db", 0.0)
            gain_linear = 10 ** (gain_db / 20)
            x = x * gain_linear

        # Digital filter (lazy import scipy)
        filter_cfg = self._preprocess_config.get("digital_filter", {})
        if filter_cfg.get("enabled", False):
            x = self._apply_digital_filter(x, filter_cfg)

        # Resampling (lazy import scipy)
        resample_cfg = self._preprocess_config.get("resampling", {})
        if resample_cfg.get("enabled", False):
            x = self._apply_resampling(x, resample_cfg)

        return x

    def _remove_dc(self, x: np.ndarray) -> np.ndarray:
        """Remove DC component in-place (no new array)."""
        # Compute mean in-place equivalent using subtract-assign.
        # np.mean always creates a scalar; we avoid any extra allocation.
        mean_val = np.mean(x)
        x -= mean_val
        return x

    def _apply_digital_filter(
        self,
        x: np.ndarray,
        config: dict[str, Any],
    ) -> np.ndarray:
        """Apply digital filter"""
        try:
            from scipy.signal import butter, cheby1, lfilter

            filter_type = config.get("type", "lowpass")
            filter_class = config.get("filter_class", "butterworth")
            cutoff = config.get("cutoff_hz", 50_000_000)
            order = config.get("order", 8)

            # Calculate normalized cutoff frequency
            nyquist = self._fs / 2
            normalized_cutoff = min(cutoff / nyquist, 0.99)

            # Design filter
            if filter_class.lower() == "butterworth":
                b, a = butter(order, normalized_cutoff, btype=filter_type)
            elif filter_class.lower() == "chebyshev":
                b, a = cheby1(order, 0.5, normalized_cutoff, btype=filter_type)
            else:
                b, a = butter(order, normalized_cutoff, btype=filter_type)

            # Apply filter
            return lfilter(b, a, x)

        except ImportError:
            print("[WARNING] scipy not installed, skipping digital filter")
            return x

    def _apply_resampling(
        self,
        x: np.ndarray,
        config: dict[str, Any],
    ) -> np.ndarray:
        """Apply resampling"""
        try:
            from scipy.signal import resample_poly

            target_fs = config.get("target_fs", 200_000_000)
            gcd = np.gcd(int(target_fs), int(self._fs))
            up = int(target_fs) // gcd
            down = int(self._fs) // gcd

            resampled = resample_poly(x, up, down)
            self._fs = target_fs  # Update sampling rate
            return resampled

        except ImportError:
            print("[WARNING] scipy not installed, skipping resampling")
            return x

    def _apply_stage(
        self,
        stage_name: str,
        x: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Apply a single stage"""
        cfg = self._stage_configs.get(stage_name, {})
        method = cfg.get("method", "")
        params = cfg.get("params", {})

        # Lazy import backend modules
        try:
            if stage_name == "fading":
                from ChannelSimulation.channels.fading import apply_rayleigh_or_rician_fading

                fading_type = "rician" if method.lower() == "rician" else "rayleigh"
                return apply_rayleigh_or_rician_fading(
                    x,
                    fading_type=fading_type,
                    k_factor=float(params.get("k_factor", 5.0)),
                    block_size=int(params.get("block_size", 256)),
                    rng=rng,
                )

            elif stage_name == "doppler":
                from ChannelSimulation.channels.doppler import apply_doppler

                return apply_doppler(
                    x,
                    fs=self._fs,
                    doppler_shift_hz=float(params.get("doppler_shift_hz", 0.0)),
                    doppler_spread_hz=float(params.get("doppler_spread_hz", 100.0)),
                    rng=rng,
                )

            elif stage_name == "multipath":
                from ChannelSimulation.channels.multipath import apply_multipath

                return apply_multipath(
                    x,
                    max_delay_samp=int(params.get("max_delay_samp", 20)),
                    n_paths=int(params.get("n_paths", 4)),
                    exp_decay=float(params.get("exp_decay", 0.35)),
                    rng=rng,
                )

            elif stage_name == "cfo":
                from ChannelSimulation.channels.cfo import apply_cfo

                return apply_cfo(
                    x,
                    fs=self._fs,
                    cfo_hz=float(params.get("cfo_hz", 0.0)),
                )

            elif stage_name == "phase_noise":
                from ChannelSimulation.channels.phase_noise import apply_phase_noise

                return apply_phase_noise(
                    x,
                    phase_std_per_sample=float(params.get("phase_std_per_sample", 0.002)),
                    rng=rng,
                )

            elif stage_name == "iq_imbalance":
                from ChannelSimulation.channels.iq_imbalance import apply_iq_imbalance

                return apply_iq_imbalance(
                    x,
                    gain_imbalance_db=float(params.get("gain_imbalance_db", 1.0)),
                    phase_imbalance_deg=float(params.get("phase_imbalance_deg", 3.0)),
                )

            elif stage_name == "awgn":
                from ChannelSimulation.channels.awgn import apply_awgn

                return apply_awgn(
                    x,
                    snr_db=float(params.get("snr_db", 15.0)),
                    rng=rng,
                    ref_power=self._preprocess_power,
                )

            elif stage_name == "crop_time_shift":
                from ChannelSimulation.channels.crop_shift import apply_random_crop_and_time_shift

                return apply_random_crop_and_time_shift(
                    x,
                    min_keep_ratio=float(params.get("min_keep_ratio", 0.7)),
                    max_shift=int(params.get("max_shift", 300)),
                    rng=rng,
                )

            else:
                return x

        except ImportError as e:
            print(f"[WARNING] Cannot import {stage_name} module: {e}")
            return x

    # =========================================================================
    # Intermediate Signal Access
    # =========================================================================

    def get_stage_signal(self, stage_name: str) -> np.ndarray | None:
        """Get signal for a specified stage"""
        return self._stage_signals.get(stage_name)

    def get_all_stage_signals(self) -> dict[str, np.ndarray]:
        """Get signals for all stages"""
        return dict(self._stage_signals)

    def get_stage_names(self) -> list[str]:
        """Get all available stage names"""
        return list(self._stage_signals.keys())

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def fs(self) -> float:
        """Get sampling rate"""
        return self._fs

    @fs.setter
    def fs(self, value: float) -> None:
        """Set sampling rate"""
        self._fs = float(value)

    @property
    def fc(self) -> float:
        """Get center frequency"""
        return self._fc

    @fc.setter
    def fc(self, value: float) -> None:
        """Set center frequency"""
        self._fc = float(value)

    @property
    def seed(self) -> int:
        """Get random seed"""
        return self._seed

    @seed.setter
    def seed(self, value: int) -> None:
        """Set random seed"""
        self._seed = int(value)

    @property
    def n_samples(self) -> int | None:
        """Get number of samples"""
        if self._iq_data is None:
            return None
        return len(self._iq_data)

    @property
    def duration(self) -> float | None:
        """Get signal duration in seconds"""
        if self._iq_data is None or self._fs <= 0:
            return None
        return len(self._iq_data) / self._fs

    @property
    def power(self) -> float | None:
        """Get signal power in dBFS"""
        if self._iq_data is None:
            return None
        power = float(np.mean(np.abs(self._iq_data) ** 2))
        if power <= 0:
            return None
        return 10 * np.log10(power)
