from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from .awgn import apply_awgn
    from .cfo import apply_cfo
    from .common import normalize_power
    from .crop_shift import apply_random_crop_and_time_shift
    from .fading import apply_rayleigh_or_rician_fading
    from .io_utils import read_interleaved_iq, save_interleaved_iq
    from .iq_imbalance import apply_iq_imbalance
    from .metadata import build_run_metadata, write_json
    from .multipath import apply_multipath
    from .phase_noise import apply_phase_noise
    from .visualization import plot_time_frequency
except ImportError:  # 兼容直接脚本方式
    from awgn import apply_awgn
    from cfo import apply_cfo
    from common import normalize_power
    from crop_shift import apply_random_crop_and_time_shift
    from fading import apply_rayleigh_or_rician_fading
    from io_utils import read_interleaved_iq, save_interleaved_iq
    from iq_imbalance import apply_iq_imbalance
    from metadata import build_run_metadata, write_json
    from multipath import apply_multipath
    from phase_noise import apply_phase_noise
    from visualization import plot_time_frequency


STAGE_ORDER = [
    "original",
    "fading",
    "multipath",
    "cfo",
    "phase_noise",
    "iq_imbalance",
    "awgn",
    "crop_time_shift",
]


def parse_enabled_stages(stage_spec: str) -> list[str]:
    """解析命令行阶段选择参数。支持 all 或逗号分隔阶段名。"""
    spec = stage_spec.strip().lower()
    if spec == "all":
        return STAGE_ORDER.copy()

    picked = [x.strip().lower() for x in spec.split(",") if x.strip()]
    invalid = [x for x in picked if x not in STAGE_ORDER]
    if invalid:
        raise ValueError(f"非法 stages: {invalid}，可选: {STAGE_ORDER} 或 all")

    return [s for s in STAGE_ORDER if s in set(picked)]


def _print_stage_progress(enabled_stages: list[str], stage: str, prefix: str = "") -> None:
    idx = enabled_stages.index(stage) + 1
    total = len(enabled_stages)
    lead = f"[{prefix}] " if prefix else ""
    print(f"{lead}进度 {idx}/{total}: {stage}")


def run_channel_pipeline(
    iq: np.ndarray,
    fs: float,
    out_dir: str | Path,
    fading_type: str = "rayleigh",
    k_factor: float = 5.0,
    cfo_hz: float = 15e3,
    phase_std_per_sample: float = 2e-3,
    gain_imbalance_db: float = 1.0,
    phase_imbalance_deg: float = 3.0,
    snr_db: float = 15.0,
    seed: int = 42,
    enabled_stages: list[str] | None = None,
    show_progress: bool = False,
    progress_prefix: str = "",
) -> np.ndarray:
    if enabled_stages is None:
        enabled_stages = STAGE_ORDER.copy()

    rng = np.random.default_rng(seed)
    x = normalize_power(iq)

    if "original" in enabled_stages:
        if show_progress:
            _print_stage_progress(enabled_stages, "original", prefix=progress_prefix)
        plot_time_frequency(x, fs, "stage0_original", out_dir)

    if "fading" in enabled_stages:
        if show_progress:
            _print_stage_progress(enabled_stages, "fading", prefix=progress_prefix)
        x = apply_rayleigh_or_rician_fading(x, fading_type=fading_type, k_factor=k_factor, rng=rng)
        plot_time_frequency(x, fs, "stage1_fading", out_dir)

    if "multipath" in enabled_stages:
        if show_progress:
            _print_stage_progress(enabled_stages, "multipath", prefix=progress_prefix)
        x = apply_multipath(x, max_delay_samp=20, n_paths=4, exp_decay=0.35, rng=rng)
        plot_time_frequency(x, fs, "stage2_multipath", out_dir)

    if "cfo" in enabled_stages:
        if show_progress:
            _print_stage_progress(enabled_stages, "cfo", prefix=progress_prefix)
        x = apply_cfo(x, fs=fs, cfo_hz=cfo_hz)
        plot_time_frequency(x, fs, "stage3_cfo", out_dir)

    if "phase_noise" in enabled_stages:
        if show_progress:
            _print_stage_progress(enabled_stages, "phase_noise", prefix=progress_prefix)
        x = apply_phase_noise(x, phase_std_per_sample=phase_std_per_sample, rng=rng)
        plot_time_frequency(x, fs, "stage4_phase_noise", out_dir)

    if "iq_imbalance" in enabled_stages:
        if show_progress:
            _print_stage_progress(enabled_stages, "iq_imbalance", prefix=progress_prefix)
        x = apply_iq_imbalance(x, gain_imbalance_db=gain_imbalance_db, phase_imbalance_deg=phase_imbalance_deg)
        plot_time_frequency(x, fs, "stage5_iq_imbalance", out_dir)

    if "awgn" in enabled_stages:
        if show_progress:
            _print_stage_progress(enabled_stages, "awgn", prefix=progress_prefix)
        x = apply_awgn(x, snr_db=snr_db, rng=rng)
        plot_time_frequency(x, fs, "stage6_awgn", out_dir)

    if "crop_time_shift" in enabled_stages:
        if show_progress:
            _print_stage_progress(enabled_stages, "crop_time_shift", prefix=progress_prefix)
        x = apply_random_crop_and_time_shift(x, min_keep_ratio=0.7, max_shift=300, rng=rng)
        plot_time_frequency(x, fs, "stage7_crop_time_shift", out_dir)

    return x


def process_one_file(
    input_path: str | Path,
    fs: float,
    out_dir: str | Path,
    output_iq: str | Path,
    fading: str,
    k_factor: float,
    cfo_hz: float,
    phase_std: float,
    iq_gain_db: float,
    iq_phase_deg: float,
    snr_db: float,
    seed: int,
    metadata_path: str | Path,
    enabled_stages: list[str],
    show_progress: bool,
    progress_prefix: str = "",
) -> None:
    iq = read_interleaved_iq(input_path)
    y = run_channel_pipeline(
        iq,
        fs=fs,
        out_dir=out_dir,
        fading_type=fading,
        k_factor=k_factor,
        cfo_hz=cfo_hz,
        phase_std_per_sample=phase_std,
        gain_imbalance_db=iq_gain_db,
        phase_imbalance_deg=iq_phase_deg,
        snr_db=snr_db,
        seed=seed,
        enabled_stages=enabled_stages,
        show_progress=show_progress,
        progress_prefix=progress_prefix,
    )

    save_interleaved_iq(output_iq, y)

    metadata = build_run_metadata(
        input_file=input_path,
        output_iq=output_iq,
        image_dir=out_dir,
        fs=fs,
        fading=fading,
        k_factor=k_factor,
        cfo_hz=cfo_hz,
        phase_std=phase_std,
        iq_gain_db=iq_gain_db,
        iq_phase_deg=iq_phase_deg,
        snr_db=snr_db,
        seed=seed,
    )
    metadata["enabled_stages"] = enabled_stages
    write_json(metadata_path, metadata)

    print(f"处理完成: {input_path}")
    print(f"  图像目录: {out_dir}")
    print(f"  输出IQ: {output_iq}")
    print(f"  参数记录: {metadata_path}")


def load_yaml_config(config_path: str | Path | None) -> dict[str, object]:
    if config_path is None:
        return {}

    try:
        import yaml
    except ImportError as e:
        raise ImportError("未安装 PyYAML，请先安装 pyyaml>=6.0.2") from e

    p = Path(config_path)
    if not p.exists() or not p.is_file():
        raise ValueError(f"配置文件不存在或不是文件: {p}")

    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        return {}
    if not isinstance(cfg, dict):
        raise ValueError("YAML 顶层必须是字典")
    return cfg


def _pick(cli_value, cfg: dict[str, object], key: str, default):
    return cli_value if cli_value is not None else cfg.get(key, default)


def main() -> None:
    parser = argparse.ArgumentParser(description="ChannelSimulation: 完整信道退化 pipeline + 每步时频图")

    parser.add_argument("input", nargs="?", type=str, default=None, help="输入 IQ 文件（float32 交织IQ）")
    parser.add_argument("--config", type=str, default=None, help="YAML 配置文件路径")

    parser.add_argument("--input-dir", type=str, default=None, help="批处理输入目录（与 input 二选一）")
    parser.add_argument("--input-glob", type=str, default=None, help="批处理文件匹配模式，默认 *.iq")

    parser.add_argument("--fs", type=float, default=None, help="采样率（Hz）")
    parser.add_argument("--out-dir", type=str, default=None, help="图像与输出保存根目录")
    parser.add_argument("--output-iq", type=str, default=None, help="单文件模式下最终退化后 IQ 输出")
    parser.add_argument("--meta-json", type=str, default=None, help="单文件模式下参数记录 JSON 输出")

    parser.add_argument("--fading", type=str, choices=["rayleigh", "rician"], default=None)
    parser.add_argument("--k-factor", type=float, default=None, help="Rician K 因子")
    parser.add_argument("--cfo-hz", type=float, default=None)
    parser.add_argument("--phase-std", type=float, default=None, help="每采样相位增量标准差")
    parser.add_argument("--iq-gain-db", type=float, default=None)
    parser.add_argument("--iq-phase-deg", type=float, default=None)
    parser.add_argument("--snr-db", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument(
        "--stages",
        type=str,
        default=None,
        help=(
            "选择执行阶段。all 或逗号分隔列表："
            "original,fading,multipath,cfo,phase_noise,iq_imbalance,awgn,crop_time_shift"
        ),
    )
    parser.add_argument("--show-progress", action="store_true", default=None, help="显示每个阶段与批处理进度")

    args = parser.parse_args()
    cfg = load_yaml_config(args.config)

    input_path = _pick(args.input, cfg, "input", None)
    input_dir = _pick(args.input_dir, cfg, "input_dir", None)
    input_glob = _pick(args.input_glob, cfg, "input_glob", "*.iq")

    fs = _pick(args.fs, cfg, "fs", None)
    out_dir = _pick(args.out_dir, cfg, "out_dir", "ChannelSimulation/results")
    output_iq = _pick(args.output_iq, cfg, "output_iq", "ChannelSimulation/results/processed.iq")
    meta_json = _pick(args.meta_json, cfg, "meta_json", "ChannelSimulation/results/run_meta.json")

    fading = _pick(args.fading, cfg, "fading", "rayleigh")
    k_factor = _pick(args.k_factor, cfg, "k_factor", 5.0)
    cfo_hz = _pick(args.cfo_hz, cfg, "cfo_hz", 15e3)
    phase_std = _pick(args.phase_std, cfg, "phase_std", 2e-3)
    iq_gain_db = _pick(args.iq_gain_db, cfg, "iq_gain_db", 1.0)
    iq_phase_deg = _pick(args.iq_phase_deg, cfg, "iq_phase_deg", 3.0)
    snr_db = _pick(args.snr_db, cfg, "snr_db", 15.0)
    seed = _pick(args.seed, cfg, "seed", 42)

    stages_spec = _pick(args.stages, cfg, "stages", "all")
    show_progress = bool(_pick(args.show_progress, cfg, "show_progress", False))

    if fs is None:
        raise ValueError("缺少 fs，请通过 --fs 或 YAML 配置提供")

    enabled_stages = parse_enabled_stages(str(stages_spec))

    if (input_path is None and input_dir is None) or (input_path is not None and input_dir is not None):
        raise ValueError("请仅指定一种输入：单文件 input 或批处理 input_dir")

    if input_path is not None:
        process_one_file(
            input_path=input_path,
            fs=float(fs),
            out_dir=out_dir,
            output_iq=output_iq,
            fading=str(fading),
            k_factor=float(k_factor),
            cfo_hz=float(cfo_hz),
            phase_std=float(phase_std),
            iq_gain_db=float(iq_gain_db),
            iq_phase_deg=float(iq_phase_deg),
            snr_db=float(snr_db),
            seed=int(seed),
            metadata_path=meta_json,
            enabled_stages=enabled_stages,
            show_progress=show_progress,
            progress_prefix=Path(str(input_path)).name,
        )
        return

    input_dir_path = Path(str(input_dir))
    if not input_dir_path.exists() or not input_dir_path.is_dir():
        raise ValueError(f"输入目录不存在或不是目录: {input_dir_path}")

    files = sorted(input_dir_path.glob(str(input_glob)))
    if not files:
        raise ValueError(f"目录中未找到匹配文件: {input_dir_path} / {input_glob}")

    out_root = Path(str(out_dir))
    out_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "input_dir": str(input_dir_path),
        "input_glob": str(input_glob),
        "total_files": len(files),
        "enabled_stages": enabled_stages,
        "runs": [],
    }

    for idx, file_path in enumerate(files):
        if show_progress:
            print(f"文件进度 {idx + 1}/{len(files)}: {file_path.name}")

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
            fading=str(fading),
            k_factor=float(k_factor),
            cfo_hz=float(cfo_hz),
            phase_std=float(phase_std),
            iq_gain_db=float(iq_gain_db),
            iq_phase_deg=float(iq_phase_deg),
            snr_db=float(snr_db),
            seed=int(seed) + idx,
            metadata_path=sample_meta,
            enabled_stages=enabled_stages,
            show_progress=show_progress,
            progress_prefix=file_path.name,
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
    print(f"批处理完成，共 {len(files)} 个文件。汇总: {summary_path}")


if __name__ == "__main__":
    main()
