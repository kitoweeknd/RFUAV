from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np

try:
    from ChannelSimulation.channels.awgn import apply_awgn
    from ChannelSimulation.channels.cfo import apply_cfo
    from ChannelSimulation.channels.crop_shift import apply_random_crop_and_time_shift
    from ChannelSimulation.channels.doppler import apply_doppler
    from ChannelSimulation.channels.fading import apply_rayleigh_or_rician_fading
    from ChannelSimulation.channels.iq_imbalance import apply_iq_imbalance
    from ChannelSimulation.channels.multipath import apply_multipath
    from ChannelSimulation.channels.phase_noise import apply_phase_noise
    from ChannelSimulation.core.common import normalize_power
    from ChannelSimulation.core.io_utils import (
        append_interleaved_iq,
        count_complex_samples,
        iter_iq_chunks,
        read_interleaved_iq,
        save_interleaved_iq,
    )
    from ChannelSimulation.metadata import build_run_metadata, write_json
    from ChannelSimulation.viz.visualization import plot_time_frequency
except ImportError:  # Compatible with direct script execution
    from ChannelSimulation.channels.awgn import apply_awgn
    from ChannelSimulation.channels.cfo import apply_cfo
    from ChannelSimulation.channels.crop_shift import apply_random_crop_and_time_shift
    from ChannelSimulation.channels.doppler import apply_doppler
    from ChannelSimulation.channels.fading import apply_rayleigh_or_rician_fading
    from ChannelSimulation.channels.iq_imbalance import apply_iq_imbalance
    from ChannelSimulation.channels.multipath import apply_multipath
    from ChannelSimulation.channels.phase_noise import apply_phase_noise
    from ChannelSimulation.core.common import normalize_power
    from ChannelSimulation.core.io_utils import (
        append_interleaved_iq,
        count_complex_samples,
        iter_iq_chunks,
        read_interleaved_iq,
        save_interleaved_iq,
    )
    from ChannelSimulation.metadata import build_run_metadata, write_json
    from ChannelSimulation.viz.visualization import plot_time_frequency


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

# Chunks that require the full signal (cannot be processed per-chunk independently).
# Doppler builds a continuous phase (arange-based rotation + cumulative phase noise),
# so processing it per-chunk would reset the phase and create boundary discontinuities.
_STATEFUL_STAGES = frozenset({"crop_time_shift", "doppler"})


def parse_enabled_stages(stage_spec: str | list[str]) -> list[str]:
    """Parse stage selection parameter. Supports all, comma-separated string or list."""
    if isinstance(stage_spec, list):
        picked = [str(x).strip().lower() for x in stage_spec if str(x).strip()]
        invalid = [x for x in picked if x not in STAGE_ORDER]
        if invalid:
            raise ValueError(f"Invalid stages: {invalid}, valid options: {STAGE_ORDER} or all")
        return [s for s in STAGE_ORDER if s in set(picked)]

    spec = str(stage_spec).strip().lower()
    if spec == "all":
        return STAGE_ORDER.copy()

    picked = [x.strip().lower() for x in spec.split(",") if x.strip()]
    invalid = [x for x in picked if x not in STAGE_ORDER]
    if invalid:
        raise ValueError(f"Invalid stages: {invalid}, valid options: {STAGE_ORDER} or all")

    return [s for s in STAGE_ORDER if s in set(picked)]


def _print_stage_progress(enabled_stages: list[str], stage: str, prefix: str = "") -> None:
    idx = enabled_stages.index(stage) + 1
    total = len(enabled_stages)
    lead = f"[{prefix}] " if prefix else ""
    print(f"{lead}Progress {idx}/{total}: {stage}")


# ----------------------------------------------------------------------
# Stage wrappers
# ----------------------------------------------------------------------
def _stage_fading_rayleigh(x: np.ndarray, fs: float, rng: np.random.Generator, **params) -> np.ndarray:
    return apply_rayleigh_or_rician_fading(
        x,
        fading_type="rayleigh",
        k_factor=float(params.get("k_factor", 4.0)),
        block_size=int(params.get("block_size", 256)),
        rng=rng,
    )


def _stage_fading_rician(x: np.ndarray, fs: float, rng: np.random.Generator, **params) -> np.ndarray:
    return apply_rayleigh_or_rician_fading(
        x,
        fading_type="rician",
        k_factor=float(params.get("k_factor", 5.0)),
        block_size=int(params.get("block_size", 256)),
        rng=rng,
    )


def _stage_fading_rice_mild(x: np.ndarray, fs: float, rng: np.random.Generator, **params) -> np.ndarray:
    return apply_rayleigh_or_rician_fading(
        x,
        fading_type="rice_mild",
        k_factor=float(params.get("k_factor", 20.0)),
        block_size=int(params.get("block_size", 1024)),
        rng=rng,
    )


def _stage_doppler_jakes(x: np.ndarray, fs: float, rng: np.random.Generator, **params) -> np.ndarray:
    return apply_doppler(
        x,
        fs=fs,
        doppler_shift_hz=float(params.get("doppler_shift_hz", 0.0)),
        doppler_spread_hz=float(params.get("doppler_spread_hz", 100.0)),
        rng=rng,
    )


def _stage_multipath_tdl(x: np.ndarray, fs: float, rng: np.random.Generator, **params) -> np.ndarray:
    return apply_multipath(
        x,
        max_delay_samp=int(params.get("max_delay_samp", 20)),
        n_paths=int(params.get("n_paths", 4)),
        exp_decay=float(params.get("exp_decay", 0.35)),
        rng=rng,
    )


def _stage_cfo_constant(x: np.ndarray, fs: float, rng: np.random.Generator, **params) -> np.ndarray:
    return apply_cfo(x, fs=fs, cfo_hz=float(params.get("cfo_hz", 0.0)))


def _stage_phase_noise_wiener(x: np.ndarray, fs: float, rng: np.random.Generator, **params) -> np.ndarray:
    return apply_phase_noise(x, phase_std_per_sample=float(params.get("phase_std_per_sample", 2e-3)), rng=rng)


def _stage_iq_imbalance_widely_linear(x: np.ndarray, fs: float, rng: np.random.Generator, **params) -> np.ndarray:
    return apply_iq_imbalance(
        x,
        gain_imbalance_db=float(params.get("gain_imbalance_db", 1.0)),
        phase_imbalance_deg=float(params.get("phase_imbalance_deg", 3.0)),
    )


def _stage_awgn_gaussian(x: np.ndarray, fs: float, rng: np.random.Generator, **params) -> np.ndarray:
    return apply_awgn(x, snr_db=float(params.get("snr_db", 15.0)), rng=rng,
                      ref_power=params.get("ref_power"))


def _stage_crop_shift_basic(x: np.ndarray, fs: float, rng: np.random.Generator, **params) -> np.ndarray:
    return apply_random_crop_and_time_shift(
        x,
        min_keep_ratio=float(params.get("min_keep_ratio", 0.7)),
        max_shift=int(params.get("max_shift", 300)),
        rng=rng,
    )


STAGE_REGISTRY: dict[str, dict[str, Callable[..., np.ndarray]]] = {
    "fading": {
        "rayleigh": _stage_fading_rayleigh,
        "rician": _stage_fading_rician,
        "rice_mild": _stage_fading_rice_mild,
    },
    "doppler": {
        "jakes": _stage_doppler_jakes,
    },
    "multipath": {
        "tdl": _stage_multipath_tdl,
    },
    "cfo": {
        "constant": _stage_cfo_constant,
    },
    "phase_noise": {
        "wiener": _stage_phase_noise_wiener,
    },
    "iq_imbalance": {
        "widely_linear": _stage_iq_imbalance_widely_linear,
    },
    "awgn": {
        "gaussian": _stage_awgn_gaussian,
    },
    "crop_time_shift": {
        "basic": _stage_crop_shift_basic,
    },
}


def default_stage_configs() -> dict[str, dict[str, object]]:
    return {
        "fading": {"method": "rayleigh", "params": {"k_factor": 5.0, "block_size": 256}},
        "doppler": {"method": "jakes", "params": {"doppler_shift_hz": 0.0, "doppler_spread_hz": 100.0}},
        "multipath": {"method": "tdl", "params": {"max_delay_samp": 20, "n_paths": 4, "exp_decay": 0.35}},
        "cfo": {"method": "constant", "params": {"cfo_hz": 0.0}},
        "phase_noise": {"method": "wiener", "params": {"phase_std_per_sample": 2e-3}},
        "iq_imbalance": {"method": "widely_linear", "params": {"gain_imbalance_db": 1.0, "phase_imbalance_deg": 3.0}},
        "awgn": {"method": "gaussian", "params": {"snr_db": 15.0}},
        "crop_time_shift": {"method": "basic", "params": {"min_keep_ratio": 0.7, "max_shift": 300}},
    }


def _merge_dict(base: dict, updates: dict) -> dict:
    out = dict(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_dict(out[k], v)
        else:
            out[k] = v
    return out


def _normalize_stage_configs(stage_configs: dict[str, object]) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for stage, cfg in stage_configs.items():
        if stage not in STAGE_REGISTRY:
            continue
        if not isinstance(cfg, dict):
            raise ValueError(f"stage_configs.{stage} must be a dictionary")
        method = str(cfg.get("method", "")).strip().lower()
        params = cfg.get("params", {})
        if not isinstance(params, dict):
            raise ValueError(f"stage_configs.{stage}.params must be a dictionary")
        if method not in STAGE_REGISTRY[stage]:
            allowed = list(STAGE_REGISTRY[stage].keys())
            raise ValueError(f"stage {stage} method={method} not supported, valid options: {allowed}")
        out[stage] = {
            "method": method,
            "params": params,
            "save": bool(cfg.get("save", False)),
        }
    return out


def _build_stage_configs(
    cfg: dict[str, object],
    fading: str,
    k_factor: float,
    cfo_hz: float,
    phase_std: float,
    iq_gain_db: float,
    iq_phase_deg: float,
    snr_db: float,
) -> dict[str, dict[str, object]]:
    result = default_stage_configs()

    result["fading"]["method"] = "rician" if str(fading).lower() == "rician" else "rayleigh"
    result["fading"]["params"]["k_factor"] = float(k_factor)
    result["cfo"]["params"]["cfo_hz"] = float(cfo_hz)
    result["phase_noise"]["params"]["phase_std_per_sample"] = float(phase_std)
    result["iq_imbalance"]["params"]["gain_imbalance_db"] = float(iq_gain_db)
    result["iq_imbalance"]["params"]["phase_imbalance_deg"] = float(iq_phase_deg)
    result["awgn"]["params"]["snr_db"] = float(snr_db)

    cfg_stage_configs = cfg.get("stage_configs", {})
    if cfg_stage_configs:
        if not isinstance(cfg_stage_configs, dict):
            raise ValueError("stage_configs must be a dictionary")
        result = _merge_dict(result, cfg_stage_configs)

    return _normalize_stage_configs(result)


# ----------------------------------------------------------------------
# Pipeline: original (full-signal, for small files)
# ----------------------------------------------------------------------
def run_channel_pipeline(
    iq: np.ndarray,
    fs: float,
    out_dir: str | Path,
    stage_configs: dict[str, dict[str, object]],
    seed: int = 42,
    enabled_stages: list[str] | None = None,
    show_progress: bool = False,
    progress_prefix: str = "",
    fc: float = 0.0,
) -> np.ndarray:
    """Apply all channel stages to the full IQ signal in memory.

    Each stage whose ``save`` flag is True will write its intermediate IQ to
    ``out_dir / <stage_name>.iq`` immediately after the stage completes.
    Suitable for files that fit comfortably in RAM.  For large files use
    `run_channel_pipeline_streaming` instead.
    """
    if enabled_stages is None:
        enabled_stages = STAGE_ORDER.copy()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    x = normalize_power(iq)

    if "original" in enabled_stages:
        if show_progress:
            _print_stage_progress(enabled_stages, "original", prefix=progress_prefix)
        plot_time_frequency(x, fs, "stage0_original", out_dir, fc=fc)
        if _stage_save_flag(enabled_stages, stage_configs, "original"):
            save_interleaved_iq(out_dir / "original.iq", x)

    # Reference power for all AWGN stages: use the normalized signal power
    # so that each AWGN stage's SNR target is relative to the same reference.
    ref_power = float(np.mean(np.abs(x) ** 2))

    for stage in STAGE_ORDER:
        if stage == "original" or stage not in enabled_stages:
            continue
        if show_progress:
            _print_stage_progress(enabled_stages, stage, prefix=progress_prefix)

        stage_cfg = stage_configs.get(stage, {})
        method = str(stage_cfg.get("method", "")).lower()
        params = stage_cfg.get("params", {})
        if not isinstance(params, dict):
            raise ValueError(f"stage_configs.{stage}.params must be a dictionary")

        # Pass ref_power to AWGN so it computes noise relative to the
        # normalized input signal rather than the accumulated pipeline signal.
        if stage == "awgn":
            params = {**params, "ref_power": ref_power}

        fn = STAGE_REGISTRY[stage].get(method)
        if fn is None:
            allowed = list(STAGE_REGISTRY[stage].keys())
            raise ValueError(f"stage {stage} method={method} not supported, valid options: {allowed}")

        x = fn(x, fs, rng, **params)
        plot_time_frequency(x, fs, f"stage{STAGE_ORDER.index(stage)}_{stage}", out_dir, fc=fc)

        if stage_cfg.get("save", False):
            save_interleaved_iq(out_dir / f"{stage}.iq", x)

    return x


def _stage_save_flag(
    enabled_stages: list[str],
    stage_configs: dict[str, dict[str, object]],
    stage: str,
) -> bool:
    """Return True if a stage has save=True and is enabled."""
    return stage in enabled_stages and bool(stage_configs.get(stage, {}).get("save", False))


# ----------------------------------------------------------------------
# Pipeline: streaming (chunk-by-chunk, bounded memory)
# ----------------------------------------------------------------------
def _run_stages_on_chunk(
    chunk: np.ndarray,
    fs: float,
    rng: np.random.Generator,
    enabled_stages: list[str],
    stage_configs: dict[str, dict[str, object]],
) -> np.ndarray:
    """Apply all enabled stages (except stateful ones) to a single chunk."""
    x = normalize_power(chunk)
    # Reference power for AWGN stages: use the normalized chunk power.
    ref_power = float(np.mean(np.abs(x) ** 2))
    for stage in enabled_stages:
        if stage == "original" or stage in _STATEFUL_STAGES:
            continue
        stage_cfg = stage_configs.get(stage, {})
        method = str(stage_cfg.get("method", "")).lower()
        params = stage_cfg.get("params", {})
        fn = STAGE_REGISTRY[stage].get(method)
        if fn is None:
            continue
        if stage == "awgn":
            params = {**params, "ref_power": ref_power}
        x = fn(x, fs, rng, **params)
    return x


def run_channel_pipeline_streaming(
    input_path: str | Path,
    output_path: str | Path,
    fs: float,
    stage_configs: dict[str, dict[str, object]],
    enabled_stages: list[str],
    seed: int = 42,
    chunk_size: int = 10_000_000,
    show_progress: bool = False,
    progress_prefix: str = "",
    out_dir: str | Path | None = None,
    fc: float = 0.0,
) -> None:
    """Stream-process a large IQ file in bounded memory.

    Each chunk is read, passed through all (non-stateful) stages, and written
    to `output_path`.  Stateful stages (crop_time_shift) are deferred to a
    single full-signal pass after all chunks are accumulated.

    Stages with ``save=True`` in their config will additionally write their
    intermediate IQ output to ``<out_dir> / <stage_name>.iq`` after processing
    the full signal.  In streaming mode this requires accumulating the full
    signal for those stages, so enable only when needed.

    Parameters:
    - input_path:  interleaved IQ file path.
    - output_path:  output interleaved IQ file path.
    - fs:           sampling rate in Hz.
    - stage_configs: per-stage configuration dict (includes ``save`` flag).
    - enabled_stages: which stages to apply.
    - seed:         RNG seed (each chunk advances the RNG independently).
    - chunk_size:   max complex samples per chunk (default 10 M ≈ 160 MB for complex128).
    - show_progress: print progress messages.
    - progress_prefix: prefix for progress messages.
    """
    total_samples = count_complex_samples(input_path)
    n_chunks = (total_samples + chunk_size - 1) // chunk_size

    # Stages that require full signal (cannot stream per chunk independently)
    accumulate = any(s in _STATEFUL_STAGES for s in enabled_stages)

    # Stages with save=True also need accumulation (full signal written to disk)
    stages_to_save = [
        stage for stage in enabled_stages
        if _stage_save_flag(enabled_stages, stage_configs, stage)
    ]
    needs_accumulation = accumulate or bool(stages_to_save)

    out_path = Path(output_path)
    out_dir = Path(out_dir) if out_dir is not None else out_path.parent
    if not needs_accumulation:
        if out_path.exists():
            out_path.unlink()
        out_dir.mkdir(parents=True, exist_ok=True)

    buf: list[np.ndarray] = []
    # Track per-stage accumulated buffers for intermediate saves
    stage_bufs: dict[str, list[np.ndarray]] = {s: [] for s in stages_to_save}
    samples_written = 0

    for i, chunk in enumerate(iter_iq_chunks(input_path, chunk_size)):
        if show_progress:
            print(f"{progress_prefix}  chunk {i + 1}/{n_chunks}  "
                  f"({len(chunk):,} samples) ...")

        # Deterministic per-chunk RNG derived from (seed, chunk index). This keeps
        # the main output independent of which stages are saved, and lets each
        # save-pass re-derive the identical draws so the saved intermediate is an
        # exact prefix of the main pipeline output for that chunk.
        processed = _run_stages_on_chunk(
            chunk, fs, np.random.default_rng([seed, i]), enabled_stages, stage_configs,
        )

        if needs_accumulation:
            buf.append(processed)
            for stage in stages_to_save:
                stage_out = _run_stages_on_chunk(
                    chunk, fs, np.random.default_rng([seed, i]),
                    [s for s in enabled_stages
                     if enabled_stages.index(s) <= enabled_stages.index(stage)
                     and s not in _STATEFUL_STAGES],
                    stage_configs,
                )
                stage_bufs[stage].append(stage_out)
        else:
            append_interleaved_iq(output_path, processed)
            samples_written += len(processed)

    if needs_accumulation:
        full_signal = normalize_power(np.concatenate(buf))
        del buf

        # Stateful stages run once over the full accumulated signal. Use a
        # dedicated deterministic stream so their draws do not depend on the
        # number of chunks processed above.
        rng_acc = np.random.default_rng([seed, n_chunks])

        # Write intermediate IQ for stages with save=True
        for stage in stages_to_save:
            if show_progress:
                print(f"{progress_prefix}  saving intermediate: {stage}.iq ...")
            stage_sig = normalize_power(np.concatenate(stage_bufs[stage]))
            save_interleaved_iq(out_dir / f"{stage}.iq", stage_sig)

        del stage_bufs

        # Apply remaining stateful stages
        if show_progress and accumulate:
            print(f"{progress_prefix}  applying stateful stages ...")
        for stage in enabled_stages:
            if stage not in _STATEFUL_STAGES:
                continue
            stage_cfg = stage_configs.get(stage, {})
            method = str(stage_cfg.get("method", "")).lower()
            params = stage_cfg.get("params", {})
            fn = STAGE_REGISTRY[stage].get(method)
            if fn is None:
                continue
            full_signal = fn(full_signal, fs, rng_acc, **params)
            if stage_cfg.get("save", False):
                save_interleaved_iq(out_dir / f"{stage}.iq", full_signal)

        if out_path.exists():
            out_path.unlink()
        save_interleaved_iq(output_path, full_signal)
        samples_written = len(full_signal)

    print(f"{progress_prefix}  done.  {samples_written:,} samples written → {output_path}")


# ----------------------------------------------------------------------
# Per-file processing
# ----------------------------------------------------------------------
def process_one_file(
    input_path: str | Path,
    fs: float,
    out_dir: str | Path,
    output_iq: str | Path,
    seed: int,
    metadata_path: str | Path,
    enabled_stages: list[str],
    show_progress: bool,
    stage_configs: dict[str, dict[str, object]],
    progress_prefix: str = "",
    chunk_size: int | None = None,
    chunk_limit: int | None = None,
    fc: float = 0.0,
) -> None:
    """Process one IQ file: decide streaming vs. full-signal automatically.

    Streaming is triggered when:
    - chunk_size > 0 is explicitly set (forced streaming), or
    - chunk_limit > 0 and the file exceeds chunk_limit samples (auto-switch).
    """
    n_total = count_complex_samples(input_path)
    auto_limit = chunk_limit if chunk_limit is not None else _auto_chunk_limit()
    forced_chunk = chunk_size if chunk_size and chunk_size > 0 else None
    use_streaming = forced_chunk is not None or (auto_limit > 0 and n_total > auto_limit)

    if show_progress:
        mode = "streaming" if use_streaming else "full-signal"
        print(f"{progress_prefix}  mode: {mode}  ({n_total:,} complex samples)")

    if use_streaming:
        # Streaming: visualize first chunk only (full-signal visualization is too expensive)
        first_chunk = next(iter_iq_chunks(input_path, auto_limit))
        rng_preview = np.random.default_rng(seed)
        x_preview = normalize_power(first_chunk)
        if "original" in enabled_stages:
            plot_time_frequency(x_preview, fs, "stage0_original", out_dir, fc=fc)
        del first_chunk, x_preview

        run_channel_pipeline_streaming(
            input_path=input_path,
            output_path=output_iq,
            fs=fs,
            stage_configs=stage_configs,
            enabled_stages=enabled_stages,
            seed=seed,
            chunk_size=forced_chunk if forced_chunk is not None else auto_limit,
            show_progress=show_progress,
            progress_prefix=progress_prefix,
            out_dir=out_dir,
            fc=fc,
        )
    else:
        iq = read_interleaved_iq(input_path)
        y = run_channel_pipeline(
            iq,
            fs=fs,
            out_dir=out_dir,
            stage_configs=stage_configs,
            seed=seed,
            enabled_stages=enabled_stages,
            show_progress=show_progress,
            progress_prefix=progress_prefix,
            fc=fc,
        )
        save_interleaved_iq(output_iq, y)

    metadata = build_run_metadata(
        input_file=input_path,
        output_iq=output_iq,
        image_dir=out_dir,
        fs=fs,
        fading=str(stage_configs["fading"]["method"]),
        k_factor=float(stage_configs["fading"].get("params", {}).get("k_factor", 0.0)),
        cfo_hz=float(stage_configs["cfo"].get("params", {}).get("cfo_hz", 0.0)),
        phase_std=float(stage_configs["phase_noise"].get("params", {}).get("phase_std_per_sample", 0.0)),
        iq_gain_db=float(stage_configs["iq_imbalance"].get("params", {}).get("gain_imbalance_db", 0.0)),
        iq_phase_deg=float(stage_configs["iq_imbalance"].get("params", {}).get("phase_imbalance_deg", 0.0)),
        snr_db=float(stage_configs["awgn"].get("params", {}).get("snr_db", 0.0)),
        seed=seed,
        fc=fc,
    )
    metadata["enabled_stages"] = enabled_stages
    metadata["stage_configs"] = stage_configs
    write_json(metadata_path, metadata)

    print(f"{progress_prefix}  processing done → {output_iq}")


def _auto_chunk_limit() -> int:
    """Default chunk size based on available RAM (in complex samples).

    Returns 0 to disable auto-streaming (always load full file).
    Currently returns a conservative 10 M samples (~160 MB for complex128)
    as a sensible default; override with --chunk-size 0 to disable or a
    custom value to tune.
    """
    return 0  # disabled by default; use --chunk-size to enable streaming


# ----------------------------------------------------------------------
# Config loading
# ----------------------------------------------------------------------
def load_yaml_config(config_path: str | Path | None) -> dict[str, object]:
    if config_path is None:
        return {}

    try:
        import yaml
    except ImportError as e:
        raise ImportError("PyYAML not installed, please install pyyaml>=6.0.2") from e

    p = Path(config_path)
    if not p.exists() or not p.is_file():
        raise ValueError(f"Config file does not exist or is not a file: {p}")

    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        return {}
    if not isinstance(cfg, dict):
        raise ValueError("YAML top-level must be a dictionary")
    return cfg


def _pick(cli_value, cfg: dict[str, object], key: str, default):
    return cli_value if cli_value is not None else cfg.get(key, default)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="ChannelSimulation: Complete channel degradation pipeline + "
                    "time-frequency plots for each stage",
    )

    parser.add_argument("--input", nargs="?", type=str, default="", help="Input IQ file (float32 interleaved IQ)")
    parser.add_argument("--config", type=str, default=None, help="YAML config file path")

    parser.add_argument("--input-dir", type=str, default=None,
                        help="Batch processing input directory (mutually exclusive with input)")
    parser.add_argument("--input-glob", type=str, default=None,
                        help="Batch processing file matching pattern, default *.iq")

    parser.add_argument("--fs", type=float, default=None, help="Sampling rate (Hz)")
    parser.add_argument("--fc", type=float, default=None,
                        help="Center frequency in Hz (for frequency-axis labeling in plots). "
                             "Example: --fc 5.8e9  [default: 0 Hz / baseband]")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Root directory for images and output")
    parser.add_argument("--output-iq", type=str, default=None,
                        help="Final degraded IQ output in single-file mode")
    parser.add_argument("--meta-json", type=str, default=None,
                        help="Parameters record JSON output in single-file mode")

    # Backward compatibility with old parameters
    parser.add_argument("--fading", type=str, choices=["rayleigh", "rician"], default=None)
    parser.add_argument("--k-factor", type=float, default=None, help="Rician K factor")
    parser.add_argument("--cfo-hz", type=float, default=None)
    parser.add_argument("--phase-std", type=float, default=None,
                        help="Standard deviation of phase increment per sample")
    parser.add_argument("--iq-gain-db", type=float, default=None)
    parser.add_argument("--iq-phase-deg", type=float, default=None)
    parser.add_argument("--snr-db", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument(
        "--stages", type=str, default=None,
        help=(
            "Execution stages. all or comma-separated list: "
            "original,fading,multipath,cfo,phase_noise,iq_imbalance,awgn,crop_time_shift"
        ),
    )
    parser.add_argument("--show-progress", action="store_true", default=None,
                        help="Show progress for each stage and batch processing")

    # Memory optimisation
    parser.add_argument(
        "--chunk-size", type=int, default=None,
        help=(
            "Max complex samples held in RAM at once (streaming mode). "
            "Set > 0 to enable streaming for all files. "
            "0 disables streaming.  Default: 0 (full-signal, for small files). "
            "Example: --chunk-size 10000000 ≈ 160 MB RAM per chunk."
        ),
    )
    parser.add_argument(
        "--chunk-limit", type=int, default=None,
        help=(
            "Auto-switch to streaming when file > chunk-limit complex samples. "
            "Default: 20_000_000 (≈ 320 MB for complex128).  Set to 0 to disable auto-switch."
        ),
    )

    args = parser.parse_args()
    cfg = load_yaml_config(args.config)

    input_path = _pick(args.input, cfg, "input", None)
    input_dir = _pick(args.input_dir, cfg, "input_dir", None)
    input_glob = _pick(args.input_glob, cfg, "input_glob", "*.iq")

    fs = _pick(args.fs, cfg, "fs", None)
    fc = _pick(args.fc, cfg, "fc", 0.0)
    out_dir = _pick(args.out_dir, cfg, "out_dir", "./results")
    output_iq = _pick(args.output_iq, cfg, "output_iq", "./results/processed.iq")
    meta_json = _pick(args.meta_json, cfg, "meta_json", "./results/run_meta.json")

    fading = _pick(args.fading, cfg, "fading", "rayleigh")
    k_factor = _pick(args.k_factor, cfg, "k_factor", 5.0)
    cfo_hz = _pick(args.cfo_hz, cfg, "cfo_hz", 0.0)
    phase_std = _pick(args.phase_std, cfg, "phase_std", 2e-3)
    iq_gain_db = _pick(args.iq_gain_db, cfg, "iq_gain_db", 1.0)
    iq_phase_deg = _pick(args.iq_phase_deg, cfg, "iq_phase_deg", 3.0)
    snr_db = _pick(args.snr_db, cfg, "snr_db", 15.0)
    seed = _pick(args.seed, cfg, "seed", 42)

    stages_spec = _pick(args.stages, cfg, "stages", "all")
    show_progress = bool(_pick(args.show_progress, cfg, "show_progress", False))

    chunk_size_cli = args.chunk_size
    chunk_limit_cli = args.chunk_limit

    if fs is None:
        raise ValueError("Missing fs, please provide via --fs or YAML config")

    enabled_stages = parse_enabled_stages(stages_spec)
    stage_configs = _build_stage_configs(
        cfg,
        fading=str(fading),
        k_factor=float(k_factor),
        cfo_hz=float(cfo_hz),
        phase_std=float(phase_std),
        iq_gain_db=float(iq_gain_db),
        iq_phase_deg=float(iq_phase_deg),
        snr_db=float(snr_db),
    )

    if (input_path is None and input_dir is None) or (input_path is not None and input_dir is not None):
        raise ValueError("Please specify only one input type: single file input or batch input_dir")

    if input_path is not None:
        process_one_file(
            input_path=input_path,
            fs=float(fs),
            out_dir=out_dir,
            output_iq=output_iq,
            seed=int(seed),
            metadata_path=meta_json,
            enabled_stages=enabled_stages,
            show_progress=show_progress,
            stage_configs=stage_configs,
            progress_prefix=Path(str(input_path)).name,
            chunk_size=chunk_size_cli,
            chunk_limit=chunk_limit_cli,
            fc=float(fc),
        )
        return

    input_dir_path = Path(str(input_dir))
    if not input_dir_path.exists() or not input_dir_path.is_dir():
        raise ValueError(f"Input directory does not exist or is not a directory: {input_dir_path}")

    files = sorted(input_dir_path.glob(str(input_glob)))
    if not files:
        raise ValueError(f"No matching files found in directory: {input_dir_path} / {input_glob}")

    out_root = Path(str(out_dir))
    out_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "input_dir": str(input_dir_path),
        "input_glob": str(input_glob),
        "total_files": len(files),
        "enabled_stages": enabled_stages,
        "stage_configs": stage_configs,
        "runs": [],
    }

    for idx, file_path in enumerate(files):
        if show_progress:
            print(f"File {idx + 1}/{len(files)}: {file_path.name}")

        sample_name = file_path.stem
        sample_out = out_root / sample_name
        sample_img_dir = sample_out / "images"
        sample_iq = sample_out / "processed.iq"
        sample_meta = sample_out / "run_meta.json"

        process_one_file(
            input_path=file_path,
            fs=float(fs),
            out_dir=sample_img_dir,
            output_iq=sample_iq,
            seed=int(seed) + idx,
            metadata_path=sample_meta,
            enabled_stages=enabled_stages,
            show_progress=show_progress,
            stage_configs=stage_configs,
            progress_prefix=file_path.name,
            chunk_size=chunk_size_cli,
            chunk_limit=chunk_limit_cli,
            fc=float(fc),
        )

        summary["runs"].append(
            {
                "input_file": str(file_path),
                "output_iq": str(sample_iq),
                "image_dir": str(sample_img_dir),
                "meta_json": str(sample_meta),
            }
        )

    summary_path = out_root / "batch_summary.json"
    write_json(summary_path, summary)
    print(f"Batch done: {len(files)} files → {summary_path}")


if __name__ == "__main__":
    main()
