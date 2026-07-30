"""
Preset Template Manager
~~~~~~~~~~~~~~~~~~~~~~~

Manages preset configuration templates, supports loading, saving, and deleting presets.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class PresetsManager:
    """Preset template manager"""

    PRESETS_DIR = Path(__file__).parent.parent / "core" / "presets"

    def __init__(self):
        self._presets: Dict[str, Dict[str, Any]] = {}
        # name -> source yaml path. Ground truth for delete_preset: names read
        # from disk are NOT validated (a legacy or hand-edited yaml may carry a
        # name the strict _safe_stem rejects), so the recorded path — which came
        # from PRESETS_DIR.glob and is therefore always inside the presets dir —
        # is what keeps such presets deletable through this API.
        self._preset_files: Dict[str, Path] = {}
        self._load_presets()

    def _load_presets(self) -> None:
        """Load preset templates"""
        self._presets.clear()
        self._preset_files.clear()

        # Ensure directory exists
        self.PRESETS_DIR.mkdir(parents=True, exist_ok=True)

        # Load all YAML files
        for yaml_file in self.PRESETS_DIR.glob("*.yaml"):
            try:
                with open(yaml_file, "r", encoding="utf-8") as f:
                    preset = yaml.safe_load(f)

                if preset and isinstance(preset, dict):
                    name = preset.get("name", yaml_file.stem)
                    self._presets[name] = preset
                    self._preset_files[name] = yaml_file
            except Exception as e:
                print(f"[WARNING] Failed to load preset {yaml_file}: {e}")

    def get_presets(self) -> List[str]:
        """Get all preset names

        Returns:
            List of preset names
        """
        return list(self._presets.keys())

    def get_preset(self, name: str) -> Optional[Dict[str, Any]]:
        """Get specified preset

        Args:
            name: Preset name

        Returns:
            Preset configuration dict, or None
        """
        preset = self._presets.get(name)
        if preset:
            return copy.deepcopy(preset)
        return None

    # Windows reserves these device basenames regardless of extension —
    # open("CON.yaml", "w") talks to the console device instead of creating a
    # file, so the preset would silently never reach disk.
    _RESERVED_STEMS = frozenset(
        {"CON", "PRN", "AUX", "NUL"}
        | {f"COM{i}" for i in range(1, 10)}
        | {f"LPT{i}" for i in range(1, 10)}
    )

    @classmethod
    def _safe_stem(cls, name: str) -> str:
        """Validate a GUI-supplied preset name as a safe bare filename stem.

        The name doubles as the yaml filename, so rather than rewriting it —
        which silently maps distinct names onto one file, overwriting or
        deleting an unrelated preset — reject anything that is not already a
        plain single-component name:

        - '/', '\\' and ':' cover POSIX/UNC traversal, Windows drive-relative
          escapes ("C:evil" resets the pathlib join anchor and lands outside
          PRESETS_DIR) and NTFS alternate data streams;
        - the remaining Windows-invalid filename characters would otherwise
          only fail later inside open() with an opaque OSError;
        - leading/trailing spaces or dots collide after Windows' implicit
          name trimming ("config." and "config" map to the same file);
        - reserved device basenames (CON, NUL, ...) silently write to the
          device instead of a file.

        Plain names (e.g. "UMa-AV (3GPP)") pass through unchanged, so existing
        preset files still round-trip.
        """
        if not isinstance(name, str) or not name:
            raise ValueError(f"Preset name must be a non-empty string: {name!r}")
        if set(name) & set('/\\:<>"|?*') or any(ord(c) < 32 for c in name):
            raise ValueError(
                f"Preset name contains characters not allowed in filenames: {name!r}"
            )
        if name != name.strip(" ."):
            raise ValueError(
                f"Preset name may not start or end with spaces or dots: {name!r}"
            )
        if name.split(".")[0].upper() in cls._RESERVED_STEMS:
            raise ValueError(f"Preset name is a reserved Windows device name: {name!r}")
        return name

    def add_preset(
        self,
        name: str,
        config: Dict[str, Any],
        description: str = "",
        category: str = "custom",
    ) -> None:
        """Add or update preset

        Args:
            name: Preset name
            config: Preset configuration
            description: Description
            category: Category (satellite / uav / cellular / radar / custom)
        """
        # Validate the name and write the file BEFORE touching the in-memory
        # table: if the name is rejected (or the write fails), self._presets
        # must not be left holding a phantom entry that has no backing file
        # and would silently vanish on the next reload().
        file_path = self.PRESETS_DIR / f"{self._safe_stem(name)}.yaml"

        preset = {
            "name": name,
            "description": description,
            "category": category,
            **copy.deepcopy(config),
        }

        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(preset, f, default_flow_style=False, allow_unicode=True)

        self._presets[name] = preset
        self._preset_files[name] = file_path

    def delete_preset(self, name: str) -> bool:
        """Delete preset

        Args:
            name: Preset name

        Returns:
            Whether deletion was successful
        """
        if name not in self._presets:
            return False

        # Resolve the path before touching in-memory state, so a failure here
        # cannot half-delete the preset. Prefer the recorded source file (see
        # __init__: names loaded from disk are unvalidated, and must remain
        # deletable); fall back to the validated name->path mapping — the same
        # validation as add_preset, so it resolves to the file actually written.
        file_path = self._preset_files.get(name)
        if file_path is None:
            file_path = self.PRESETS_DIR / f"{self._safe_stem(name)}.yaml"
        if file_path.exists():
            file_path.unlink()

        del self._presets[name]
        self._preset_files.pop(name, None)
        return True

    def get_presets_by_category(self, category: str) -> List[str]:
        """Get presets by category

        Args:
            category: Category name

        Returns:
            List of preset names in this category
        """
        return [
            name
            for name, preset in self._presets.items()
            if preset.get("category") == category
        ]

    def reload(self) -> None:
        """Reload presets"""
        self._load_presets()


# Create default presets
def create_default_presets() -> None:
    """Create default preset templates"""
    presets_dir = Path(__file__).parent.parent / "core" / "presets"
    presets_dir.mkdir(parents=True, exist_ok=True)

    # 3GPP TR 38.901 UMa-AV (Urban Macro-Aerial) - for NeurIPS UAV paper
    uma_av_config = {
        "name": "UMa-AV (3GPP)",
        "description": "Urban Macro-Aerial: 3GPP TR 38.901 UAV channel. BS h=25m, UAV h=22.5-300m. "
                       "High LoS probability (>100m). TDL-D/E for LoS (K=15-25dB), TDL-A/B for NLoS. "
                       "CFO: ±5-10 ppm (29-58 kHz at 5.8GHz). IQ imbalance: ΔA≈0.5dB, Δφ≈2-5°.",
        "category": "3gpp_uav",
        "fs": 100_000_000,
        "fc": 5.8e9,
        "seed": 42,
        "preprocess": {
            "dc_removal": {"enabled": True},
            "digital_filter": {"enabled": False},
            "resampling": {"enabled": False},
            "gain_control": {"enabled": False},
        },
        "stage_configs": {
            "fading": {
                "enabled": True,
                "method": "rician",
                "params": {"k_factor": 20.0, "block_size": 512},
            },
            "multipath": {
                "enabled": True,
                "method": "tdl",
                "params": {
                    "max_delay_samp": 30,
                    "n_paths": 5,
                    "exp_decay": 0.25,
                },
            },
            "cfo": {
                "enabled": True,
                "method": "constant",
                "params": {"cfo_hz": 43500.0},
            },
            "phase_noise": {
                "enabled": True,
                "method": "wiener",
                "params": {"phase_std_per_sample": 0.0015},
            },
            "iq_imbalance": {
                "enabled": True,
                "method": "widely_linear",
                "params": {"gain_imbalance_db": 0.5, "phase_imbalance_deg": 3.0},
            },
            "doppler": {
                "enabled": True,
                "method": "jakes",
                "params": {
                    "doppler_shift_hz": 2900.0,
                    "doppler_spread_hz": 200.0,
                },
            },
            "awgn": {
                "enabled": True,
                "method": "gaussian",
                "params": {"snr_db": 15.0},
            },
        },
        "3gpp_metadata": {
            "standard": "3GPP TR 38.901 / TR 36.777",
            "scenario": "UMa-AV",
            "bs_height_m": 25.0,
            "ut_height_m": 150.0,
            "los_probability": "High (>100m altitude)",
            "shadow_fading_db": {"LoS": 4.0, "NLoS": 6.0},
            "rms_delay_spread_ns": 100.0,
            "tdl_model_los": "TDL-D / TDL-E",
            "tdl_model_nlos": "TDL-A / TDL-B",
            "ricean_k_factor_db": 20.0,
            "cfo_ppm": 7.5,
            "iq_gain_mismatch_db": 0.5,
            "iq_phase_mismatch_deg": 3.0,
        },
    }

    # 3GPP TR 38.901 UMi-AV (Urban Micro-Aerial) - for NeurIPS UAV paper
    umi_av_config = {
        "name": "UMi-AV (3GPP)",
        "description": "Urban Micro-Aerial: 3GPP TR 38.901 UAV channel. BS h=10m, UAV h=22.5-300m. "
                       "Lower LoS probability than UMa due to building density. TDL-D/E for LoS (K=15-25dB), "
                       "TDL-A/B for NLoS (RMS DS=100-300ns). CFO: ±5-10 ppm. IQ imbalance: ΔA≈0.5dB, Δφ≈2-5°.",
        "category": "3gpp_uav",
        "fs": 100_000_000,
        "fc": 5.8e9,
        "seed": 123,
        "preprocess": {
            "dc_removal": {"enabled": True},
            "digital_filter": {"enabled": False},
            "resampling": {"enabled": False},
            "gain_control": {"enabled": False},
        },
        "stage_configs": {
            "fading": {
                "enabled": True,
                "method": "rician",
                "params": {"k_factor": 15.0, "block_size": 512},
            },
            "multipath": {
                "enabled": True,
                "method": "tdl",
                "params": {
                    "max_delay_samp": 60,
                    "n_paths": 7,
                    "exp_decay": 0.4,
                },
            },
            "cfo": {
                "enabled": True,
                "method": "constant",
                "params": {"cfo_hz": 58000.0},
            },
            "phase_noise": {
                "enabled": True,
                "method": "wiener",
                "params": {"phase_std_per_sample": 0.002},
            },
            "iq_imbalance": {
                "enabled": True,
                "method": "widely_linear",
                "params": {"gain_imbalance_db": 0.5, "phase_imbalance_deg": 4.0},
            },
            "doppler": {
                "enabled": True,
                "method": "jakes",
                "params": {
                    "doppler_shift_hz": 3867.0,
                    "doppler_spread_hz": 300.0,
                },
            },
            "awgn": {
                "enabled": True,
                "method": "gaussian",
                "params": {"snr_db": 12.0},
            },
        },
        "3gpp_metadata": {
            "standard": "3GPP TR 38.901 / TR 36.777",
            "scenario": "UMi-AV",
            "bs_height_m": 10.0,
            "ut_height_m": 150.0,
            "los_probability": "Moderate (building density)",
            "shadow_fading_db": {"LoS": 4.0, "NLoS": 7.82},
            "rms_delay_spread_ns": 200.0,
            "tdl_model_los": "TDL-D / TDL-E",
            "tdl_model_nlos": "TDL-A / TDL-B",
            "ricean_k_factor_db": 15.0,
            "cfo_ppm": 10.0,
            "iq_gain_mismatch_db": 0.5,
            "iq_phase_mismatch_deg": 4.0,
        },
    }

    # Satellite downlink (LEO)
    satellite_config = {
        "name": "Satellite LEO",
        "description": "LEO satellite communication channel with high Doppler + Rayleigh fading + multipath",
        "category": "satellite",
        "fs": 100_000_000,
        "fc": 0.0,
        "seed": 42,
        "preprocess": {
            "dc_removal": {"enabled": True},
            "digital_filter": {"enabled": False},
            "resampling": {"enabled": False},
            "gain_control": {"enabled": False},
        },
        "stage_configs": {
            "fading": {
                "enabled": True,
                "method": "rayleigh",
                "params": {"k_factor": 1.0, "block_size": 256},
            },
            "multipath": {
                "enabled": True,
                "method": "tdl",
                "params": {
                    "max_delay_samp": 50,
                    "n_paths": 6,
                    "exp_decay": 0.5,
                },
            },
            "doppler": {
                "enabled": True,
                "method": "jakes",
                "params": {
                    "doppler_shift_hz": 5000.0,
                    "doppler_spread_hz": 200.0,
                },
            },
            "awgn": {
                "enabled": True,
                "method": "gaussian",
                "params": {"snr_db": 10.0},
            },
        },
    }

    # UAV swarm
    uav_config = {
        "name": "UAV Swarm",
        "description": "UAV swarm network with Rayleigh fading + moderate Doppler + Phase Noise",
        "category": "uav",
        "fs": 100_000_000,
        "fc": 0.0,
        "seed": 123,
        "preprocess": {
            "dc_removal": {"enabled": True},
            "digital_filter": {"enabled": False},
            "resampling": {"enabled": False},
            "gain_control": {"enabled": False},
        },
        "stage_configs": {
            "fading": {
                "enabled": True,
                "method": "rayleigh",
                "params": {"k_factor": 2.0, "block_size": 128},
            },
            "doppler": {
                "enabled": True,
                "method": "jakes",
                "params": {
                    "doppler_shift_hz": 1000.0,
                    "doppler_spread_hz": 50.0,
                },
            },
            "phase_noise": {
                "enabled": True,
                "method": "wiener",
                "params": {"phase_std_per_sample": 0.005},
            },
            "awgn": {
                "enabled": True,
                "method": "gaussian",
                "params": {"snr_db": 20.0},
            },
        },
    }

    # Ground cellular
    cellular_config = {
        "name": "Cellular 4G-5G",
        "description": "4G/5G terrestrial channel with Rician fading + light multipath + AWGN",
        "category": "cellular",
        "fs": 100_000_000,
        "fc": 0.0,
        "seed": 456,
        "preprocess": {
            "dc_removal": {"enabled": True},
            "digital_filter": {"enabled": False},
            "resampling": {"enabled": False},
            "gain_control": {"enabled": False},
        },
        "stage_configs": {
            "fading": {
                "enabled": True,
                "method": "rician",
                "params": {"k_factor": 10.0, "block_size": 256},
            },
            "multipath": {
                "enabled": True,
                "method": "tdl",
                "params": {
                    "max_delay_samp": 10,
                    "n_paths": 3,
                    "exp_decay": 0.3,
                },
            },
            "awgn": {
                "enabled": True,
                "method": "gaussian",
                "params": {"snr_db": 25.0},
            },
        },
    }

    # Radar echo
    radar_config = {
        "name": "Radar Echo",
        "description": "Radar system simulation with multipath + additive noise + target Doppler",
        "category": "radar",
        "fs": 100_000_000,
        "fc": 0.0,
        "seed": 789,
        "preprocess": {
            "dc_removal": {"enabled": True},
            "digital_filter": {"enabled": False},
            "resampling": {"enabled": False},
            "gain_control": {"enabled": False},
        },
        "stage_configs": {
            "multipath": {
                "enabled": True,
                "method": "tdl",
                "params": {
                    "max_delay_samp": 100,
                    "n_paths": 8,
                    "exp_decay": 0.4,
                },
            },
            "doppler": {
                "enabled": True,
                "method": "jakes",
                "params": {
                    "doppler_shift_hz": 10000.0,
                    "doppler_spread_hz": 500.0,
                },
            },
            "awgn": {
                "enabled": True,
                "method": "gaussian",
                "params": {"snr_db": 15.0},
            },
        },
    }

    # Clean reference
    clean_config = {
        "name": "Clean Reference",
        "description": "All channels disabled, for baseline comparison",
        "category": "reference",
        "fs": 100_000_000,
        "fc": 0.0,
        "seed": 0,
        "preprocess": {
            "dc_removal": {"enabled": True},
            "digital_filter": {"enabled": False},
            "resampling": {"enabled": False},
            "gain_control": {"enabled": False},
        },
        "stage_configs": {
            "fading": {"enabled": False},
            "multipath": {"enabled": False},
            "doppler": {"enabled": False},
            "awgn": {"enabled": False},
            "cfo": {"enabled": False},
            "phase_noise": {"enabled": False},
            "iq_imbalance": {"enabled": False},
            "crop_time_shift": {"enabled": False},
        },
    }

    # Save presets
    for config in [
        uma_av_config,
        umi_av_config,
        satellite_config,
        uav_config,
        cellular_config,
        radar_config,
        clean_config,
    ]:
        name = config["name"]
        file_path = presets_dir / f"{name}.yaml"
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


# Singleton
_presets_manager: Optional[PresetsManager] = None


def get_presets_manager() -> PresetsManager:
    """Get presets manager singleton"""
    global _presets_manager
    if _presets_manager is None:
        _presets_manager = PresetsManager()
    return _presets_manager
