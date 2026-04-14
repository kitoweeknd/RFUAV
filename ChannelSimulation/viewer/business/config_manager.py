"""
Configuration Manager
~~~~~~~~~~~~~~~~~~~~~~

Save, load, validate YAML configuration files, and compare configurations.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class ConfigManager:
    """Configuration Manager

    Handles configuration save, load, validation, and comparison.
    """

    CONFIG_VERSION = "1.0"

    def __init__(self):
        self._current_config: Dict[str, Any] = {}
        self._config_history: List[Dict[str, Any]] = []
        self._history_index: int = -1
        self._max_history: int = 50

    def load_from_file(self, file_path: str | Path) -> Dict[str, Any]:
        """Load configuration from file

        Args:
            file_path: Path to YAML configuration file

        Returns:
            Parsed configuration dictionary

        Raises:
            FileNotFoundError: File does not exist
            ValueError: Invalid configuration format
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if config is None:
            config = {}

        if not isinstance(config, dict):
            raise ValueError("Invalid configuration format, root must be a dictionary")

        # Validate and fill default values
        config = self._normalize_config(config)

        self._current_config = copy.deepcopy(config)
        self._add_to_history(self._current_config)

        return config

    def save_to_file(
        self,
        config: Dict[str, Any],
        file_path: str | Path,
    ) -> None:
        """Save configuration to file

        Args:
            config: Configuration dictionary
            file_path: Target file path
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Add version information
        output_config = {
            "version": self.CONFIG_VERSION,
            **copy.deepcopy(config),
        }

        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(
                output_config,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )

    def _normalize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize configuration (fill default values, validate fields)

        Args:
            config: Raw configuration dictionary

        Returns:
            Normalized configuration dictionary
        """
        result = copy.deepcopy(config)

        # Ensure basic fields exist
        if "version" not in result:
            result["version"] = self.CONFIG_VERSION

        if "fs" not in result:
            result["fs"] = 100_000_000

        if "fc" not in result:
            result["fc"] = 0.0

        if "seed" not in result:
            result["seed"] = 42

        # Normalize stage_configs
        if "stage_configs" not in result:
            result["stage_configs"] = {}

        for stage_name, stage_cfg in result.get("stage_configs", {}).items():
            if not isinstance(stage_cfg, dict):
                result["stage_configs"][stage_name] = {"enabled": True, "params": {}}
                continue

            # Ensure enabled field exists (backward compatibility)
            if "enabled" not in stage_cfg:
                stage_cfg["enabled"] = True

            # Ensure params field exists
            if "params" not in stage_cfg:
                stage_cfg["params"] = {}

        # Normalize preprocess
        if "preprocess" not in result:
            result["preprocess"] = {
                "dc_removal": {"enabled": True},
                "digital_filter": {"enabled": False},
                "resampling": {"enabled": False},
                "gain_control": {"enabled": False},
            }

        return result

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate configuration validity

        Args:
            config: Configuration dictionary

        Returns:
            List of error messages (empty list means valid)
        """
        errors = []

        # Check fs
        fs = config.get("fs")
        if fs is None or not isinstance(fs, (int, float)) or fs <= 0:
            errors.append("Sampling rate (fs) must be a positive number")

        # Check seed
        seed = config.get("seed")
        if seed is not None and not isinstance(seed, int):
            errors.append("Random seed must be an integer")

        # Check stage_configs
        stage_configs = config.get("stage_configs", {})
        if not isinstance(stage_configs, dict):
            errors.append("stage_configs must be a dictionary")
        else:
            for stage_name, stage_cfg in stage_configs.items():
                if not isinstance(stage_cfg, dict):
                    errors.append(f"stage_configs.{stage_name} must be a dictionary")
                    continue

                # Check parameter ranges
                params = stage_cfg.get("params", {})
                if not isinstance(params, dict):
                    errors.append(f"stage_configs.{stage_name}.params must be a dictionary")
                    continue

                # Parameter range validation
                if stage_name == "awgn":
                    snr = params.get("snr_db")
                    if snr is not None and (snr < -50 or snr > 50):
                        errors.append("AWGN SNR must be in range -50~50 dB")

                elif stage_name == "cfo":
                    cfo = params.get("cfo_hz")
                    if cfo is not None and abs(cfo) > fs / 4 if fs else False:
                        errors.append("CFO frequency offset cannot exceed fs/4")

        return errors

    def compare_configs(
        self,
        config1: Dict[str, Any],
        config2: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Compare differences between two configurations

        Args:
            config1: Configuration 1
            config2: Configuration 2

        Returns:
            List of differences, each containing path, value1, value2
        """
        differences = []

        def compare_dicts(d1: Dict, d2: Dict, path: str = "") -> None:
            """Recursively compare dictionaries"""
            all_keys = set(d1.keys()) | set(d2.keys())

            for key in all_keys:
                current_path = f"{path}.{key}" if path else key
                v1 = d1.get(key)
                v2 = d2.get(key)

                if isinstance(v1, dict) and isinstance(v2, dict):
                    compare_dicts(v1, v2, current_path)
                elif v1 != v2:
                    differences.append({
                        "path": current_path,
                        "value1": v1,
                        "value2": v2,
                    })

        compare_dicts(config1, config2)
        return differences

    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration

        Returns:
            Default configuration dictionary
        """
        return {
            "version": self.CONFIG_VERSION,
            "fs": 100_000_000,
            "fc": 0.0,
            "seed": 42,
            "preprocess": {
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
            },
            "stage_configs": {
                "awgn": {
                    "enabled": False,
                    "method": "gaussian",
                    "params": {"snr_db": 15.0},
                },
                "cfo": {
                    "enabled": False,
                    "method": "constant",
                    "params": {"cfo_hz": 15000.0},
                },
                "fading": {
                    "enabled": False,
                    "method": "rayleigh",
                    "params": {"k_factor": 5.0, "block_size": 256},
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
                "phase_noise": {
                    "enabled": False,
                    "method": "wiener",
                    "params": {"phase_std_per_sample": 0.002},
                },
                "iq_imbalance": {
                    "enabled": False,
                    "method": "widely_linear",
                    "params": {
                        "gain_imbalance_db": 1.0,
                        "phase_imbalance_deg": 3.0,
                    },
                },
                "crop_time_shift": {
                    "enabled": False,
                    "method": "basic",
                    "params": {"min_keep_ratio": 0.7, "max_shift": 300},
                },
                "doppler": {
                    "enabled": False,
                    "method": "jakes",
                    "params": {
                        "doppler_shift_hz": 0.0,
                        "doppler_spread_hz": 100.0,
                    },
                },
            },
        }

    # =========================================================================
    # Undo/Redo Support
    # =========================================================================

    def _add_to_history(self, config: Dict[str, Any]) -> None:
        """Add configuration to history"""
        # Clear history after current position
        if self._history_index < len(self._config_history) - 1:
            self._config_history = self._config_history[: self._history_index + 1]

        # Add new configuration
        self._config_history.append(copy.deepcopy(config))

        # Limit history length
        if len(self._config_history) > self._max_history:
            self._config_history.pop(0)
        else:
            self._history_index += 1

    def undo(self) -> Optional[Dict[str, Any]]:
        """Undo last configuration change

        Returns:
            Configuration after undo, or None
        """
        if self._history_index > 0:
            self._history_index -= 1
            return copy.deepcopy(self._config_history[self._history_index])
        return None

    def redo(self) -> Optional[Dict[str, Any]]:
        """Redo last undo

        Returns:
            Configuration after redo, or None
        """
        if self._history_index < len(self._config_history) - 1:
            self._history_index += 1
            return copy.deepcopy(self._config_history[self._history_index])
        return None

    def can_undo(self) -> bool:
        """Check if undo is available"""
        return self._history_index > 0

    def can_redo(self) -> bool:
        """Check if redo is available"""
        return self._history_index < len(self._config_history) - 1

    def clear_history(self) -> None:
        """Clear history"""
        self._config_history.clear()
        self._history_index = -1
        if self._current_config:
            self._add_to_history(self._current_config)


# Global singleton
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get configuration manager singleton"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager
