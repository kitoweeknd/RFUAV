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

    # Satellite downlink (LEO)
    satellite_config = {
        "name": "卫星下行 (LEO)",
        "description": "LEO 卫星通信信道：高多普勒 + Rayleigh 衰落 + 多径",
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
        "name": "无人机集群",
        "description": "UAV 蜂群网络：Rayleigh 衰落 + 中等多普勒 + Phase Noise",
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
        "name": "地面蜂窝",
        "description": "4G/5G 地面信道：Rician 衰落 + 轻度多径 + AWGN",
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
        "name": "雷达回波",
        "description": "雷达系统仿真：多径 + 加性噪声 + 目标 Doppler",
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
        "name": "纯净参考",
        "description": "所有信道关闭，用于基线对比",
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
