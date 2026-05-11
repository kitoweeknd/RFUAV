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
        self._load_presets()

    def _load_presets(self) -> None:
        """Load preset templates"""
        self._presets.clear()

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
        preset = {
            "name": name,
            "description": description,
            "category": category,
            **copy.deepcopy(config),
        }

        self._presets[name] = preset

        # Save to file
        file_path = self.PRESETS_DIR / f"{name}.yaml"
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(preset, f, default_flow_style=False, allow_unicode=True)

    def delete_preset(self, name: str) -> bool:
        """Delete preset

        Args:
            name: Preset name

        Returns:
            Whether deletion was successful
        """
        if name not in self._presets:
            return False

        del self._presets[name]

        # Delete file
        file_path = self.PRESETS_DIR / f"{name}.yaml"
        if file_path.exists():
            file_path.unlink()

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
