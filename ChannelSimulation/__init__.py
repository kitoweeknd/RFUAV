"""
RFUAV ChannelSimulation: Wireless channel impairment pipeline.

Sub-packages:
- channels  : Channel effect implementations (AWGN, CFO, fading, multipath, etc.)
- core     : Common utilities (power normalization, IQ I/O)
- viz      : Time-frequency visualization

Top-level exports:
- process_one_file, run_channel_pipeline, run_channel_pipeline_streaming
- build_run_metadata, write_json
"""
from ChannelSimulation.channels import (
    apply_awgn,
    apply_cfo,
    apply_doppler,
    apply_iq_imbalance,
    apply_multipath,
    apply_phase_noise,
    apply_random_crop_and_time_shift,
    apply_rayleigh_or_rician_fading,
)
from ChannelSimulation.core import (
    EPS,
    normalize_power,
    read_interleaved_iq,
    save_interleaved_iq,
    append_interleaved_iq,
    count_complex_samples,
    iter_iq_chunks,
)
from ChannelSimulation.channel_pipeline import (
    process_one_file,
    run_channel_pipeline,
    run_channel_pipeline_streaming,
    STAGE_ORDER,
    STAGE_REGISTRY,
    default_stage_configs,
    load_yaml_config,
)
from ChannelSimulation.metadata import build_run_metadata, write_json
from ChannelSimulation.viz import plot_time_frequency

__all__ = [
    # channels
    "apply_awgn",
    "apply_cfo",
    "apply_doppler",
    "apply_iq_imbalance",
    "apply_multipath",
    "apply_phase_noise",
    "apply_random_crop_and_time_shift",
    "apply_rayleigh_or_rician_fading",
    # core
    "EPS",
    "normalize_power",
    "read_interleaved_iq",
    "save_interleaved_iq",
    "append_interleaved_iq",
    "count_complex_samples",
    "iter_iq_chunks",
    # pipeline (now at root)
    "process_one_file",
    "run_channel_pipeline",
    "run_channel_pipeline_streaming",
    "build_run_metadata",
    "write_json",
    # viz
    "plot_time_frequency",
]
