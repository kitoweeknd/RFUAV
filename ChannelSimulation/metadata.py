from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def build_run_metadata(
    *,
    input_file: str | Path,
    output_iq: str | Path,
    image_dir: str | Path,
    fs: float,
    fading: str,
    k_factor: float,
    cfo_hz: float,
    phase_std: float,
    iq_gain_db: float,
    iq_phase_deg: float,
    snr_db: float,
    seed: int,
    fc: float = 0.0,
) -> dict:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": str(Path(input_file)),
        "output_iq": str(Path(output_iq)),
        "image_dir": str(Path(image_dir)),
        "pipeline": [
            "original",
            "fading",
            "multipath",
            "cfo",
            "phase_noise",
            "iq_imbalance",
            "awgn",
            "crop_time_shift",
        ],
        "params": {
            "fs": fs,
            "fc": fc,
            "fading": fading,
            "k_factor": k_factor,
            "cfo_hz": cfo_hz,
            "phase_std_per_sample": phase_std,
            "iq_gain_imbalance_db": iq_gain_db,
            "iq_phase_imbalance_deg": iq_phase_deg,
            "snr_db": snr_db,
            "seed": seed,
        },
    }


def write_json(file_path: str | Path, data: dict) -> None:
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
