"""CLI entry-point smoke tests (subprocess).

`--help` on every documented entry point proves argparse wiring + imports
resolve; a few self-contained commands (channel pipeline, SNR estimation,
spectrogram batch) run for real against synthetic IQ.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _env():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["MPLBACKEND"] = "Agg"
    env["QT_QPA_PLATFORM"] = "offscreen"
    return env


def _run(args, timeout=300):
    return subprocess.run(
        [sys.executable, *args],
        cwd=str(REPO_ROOT), env=_env(),
        capture_output=True, text=True, timeout=timeout,
    )


# ---------------------------------------------------------------------------
# --help wiring for every entry point
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "args",
    [
        ["train.py", "--help"],
        ["inference.py", "--help"],
        ["benchmark.py", "--help"],
        ["-m", "graphic.RawDataProcessor", "--help"],
        ["-m", "ChannelSimulation.channel_pipeline", "--help"],
        ["-m", "SNREstimation.SNREstimation", "--help"],
        ["tools/DroneMixing.py", "--help"],
        ["tools/LabelGenerator.py", "--help"],
        ["tools/RandomSeg.py", "--help"],
        ["exp/run_multi_cfg.py", "--help"],
    ],
)
def test_cli_help(args):
    r = _run(args, timeout=120)
    assert r.returncode == 0, f"{args} exited {r.returncode}\nSTDERR:\n{r.stderr}"
    assert "usage" in (r.stdout + r.stderr).lower()


# ---------------------------------------------------------------------------
# real runs against synthetic IQ
# ---------------------------------------------------------------------------
def test_cli_channel_pipeline(iq_file, tmp_path):
    out = tmp_path / "out"
    r = _run([
        "-m", "ChannelSimulation.channel_pipeline",
        "--input", str(iq_file),
        "--fs", "1e6",
        "--stages", "awgn",
        "--out-dir", str(out),
        "--output-iq", str(tmp_path / "processed.iq"),
        "--meta-json", str(tmp_path / "meta.json"),
    ])
    assert r.returncode == 0, f"STDERR:\n{r.stderr}"
    assert (tmp_path / "processed.iq").exists()


def test_cli_snr_estimation(iq_file):
    r = _run(["-m", "SNREstimation.SNREstimation", str(iq_file), "--fs", "100e6"])
    assert r.returncode == 0, f"STDERR:\n{r.stderr}"
    assert "SNR" in r.stdout or "snr" in r.stdout.lower()


def test_cli_spectrogram_batch(iq_file, tmp_path):
    # draw_and_save expects <data_dir>/<label>/<pack.iq>
    indir = tmp_path / "raw" / "DroneA"
    indir.mkdir(parents=True)
    (indir / "synth.iq").write_bytes(Path(iq_file).read_bytes())
    outdir = tmp_path / "spec_out"
    r = _run([
        "-m", "graphic.RawDataProcessor", "spectrogram",
        "-i", str(tmp_path / "raw"),
        "-o", str(outdir),
        "--fs", "1e6", "--duration", "0.02", "--stft-point", "256",
    ])
    assert r.returncode == 0, f"STDERR:\n{r.stderr}"
    assert list(outdir.rglob("*.jpg")), "no spectrogram images written"
