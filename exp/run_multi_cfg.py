"""
exp/run_multi_cfg.py
===================
对同一个输入信号，按 exp/cfg/channel_pipeline.*.yaml 逐一运行 pipeline，
输出结果分散到 results/<cfg_name>/ 目录下。

用法
----
    python exp/run_multi_cfg.py
    python exp/run_multi_cfg.py --input data/input.iq --fs 1e8
    python exp/run_multi_cfg.py --dry-run          # 只打印命令，不执行
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# ── 默认值（与 channel_pipeline.sample.yaml 保持一致）─────────────────────────
DEFAULT_INPUT   = Path("data/input.iq")
DEFAULT_FS      = 1e8
DEFAULT_FC      = 5.8e9
DEFAULT_CFGS_DIR = Path(__file__).parent / "cfg"
DEFAULT_RESULTS  = Path(__file__).parent.parent / "results"


def build_cmd(
    cfg_path: Path,
    input_file: Path,
    fs: float,
    fc: float,
    results_dir: Path,
) -> list[str]:
    """根据单个 YAML 配置生成完整的 channel_pipeline 命令行参数列表。"""
    name = cfg_path.stem.removeprefix("channel_pipeline.")   # e.g. "cfo"
    out_dir   = results_dir / name / "images"
    output_iq = results_dir / name / "processed.iq"
    meta_json = results_dir / name / "run_meta.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    return [
        sys.executable,
        "-m", "ChannelSimulation.channel_pipeline",
        "--config",    str(cfg_path),
        "--input",     str(input_file),
        "--fs",        str(fs),
        "--fc",        str(fc),
        "--out-dir",   str(out_dir),
        "--output-iq", str(output_iq),
        "--meta-json", str(meta_json),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="批量运行 channel_pipeline：同一个信号 × 多个 YAML 配置",
    )
    parser.add_argument(
        "--input",  type=Path, default=None,
        help=f"输入 IQ 文件 [默认: {DEFAULT_INPUT}]",
    )
    parser.add_argument(
        "--fs", type=float, default=None,
        help=f"采样率 (Hz) [默认: {DEFAULT_FS}]",
    )
    parser.add_argument(
        "--fc", type=float, default=None,
        help=f"中心频率 (Hz)，用于绘图标注 [默认: {DEFAULT_FC}]",
    )
    parser.add_argument(
        "--cfgs-dir", type=Path, default=None,
        help=f"配置文件目录 [默认: {DEFAULT_CFGS_DIR}]",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=None,
        help=f"输出根目录 [默认: {DEFAULT_RESULTS}]",
    )
    parser.add_argument(
        "--glob", type=str, default="channel_pipeline.*.yaml",
        help="配置文件通配符 [默认: channel_pipeline.*.yaml]",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只打印将要执行的命令，不实际运行",
    )
    args = parser.parse_args()

    input_file   = args.input   or DEFAULT_INPUT
    fs           = args.fs      or DEFAULT_FS
    fc           = args.fc      or DEFAULT_FC
    cfgs_dir     = args.cfgs_dir or DEFAULT_CFGS_DIR
    results_dir  = args.results_dir or DEFAULT_RESULTS
    glob_pat     = args.glob

    if not input_file.exists():
        sys.exit(f"[错误] 输入文件不存在: {input_file}")
    if not cfgs_dir.is_dir():
        sys.exit(f"[错误] 配置目录不存在: {cfgs_dir}")

    configs = sorted(cfgs_dir.glob(glob_pat))
    if not configs:
        sys.exit(f"[错误] 在 {cfgs_dir} 中没有匹配 '{glob_pat}' 的配置文件")
    # 排除 sample.yaml
    configs = [c for c in configs if c.name != "channel_pipeline.sample.yaml"]

    print(f"=== 找到 {len(configs)} 个配置 ===")
    for cfg in configs:
        print(f"  - {cfg.name}")
    print()

    for idx, cfg_path in enumerate(configs, 1):
        name = cfg_path.stem.removeprefix("channel_pipeline.")
        cmd  = build_cmd(cfg_path, input_file, fs, fc, results_dir)
        cmd_str = " ".join(cmd)

        print(f"[{idx}/{len(configs)}] ▶ {name}")
        if args.dry_run:
            print(f"    {cmd_str}")
            continue

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"\n[错误] {name} 失败，退出码: {e.returncode}")
            print("继续处理下一个配置（Ctrl+C 可中断）...")
        except KeyboardInterrupt:
            print("\n[中断] 用户主动终止。")
            break

    print("\n=== 全部完成 ===")
    print(f"结果目录: {results_dir}")
    for cfg in configs:
        name = cfg.stem.removeprefix("channel_pipeline.")
        out  = results_dir / name / "processed.iq"
        status = "✓" if out.exists() else "✗"
        print(f"  {status} {name}/processed.iq")


if __name__ == "__main__":
    main()
