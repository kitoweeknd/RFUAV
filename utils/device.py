"""
Compute-device selection for training and inference.

Centralizes PyTorch backend selection:
- CUDA (NVIDIA GPU)
- MPS  (Apple Metal Performance Shaders — GPU/NPU on Apple Silicon Macs)
- CPU  (fallback when the requested backend is unavailable)

Used by `utils.build.check_cfg()` and `utils.trainer.Basetrainer`.
The `device` value from a YAML config passes through `resolve_device()`
before a `torch.device` is created, so configs may specify `cpu`, `cuda`,
`mps`, or `cuda:0`.
"""

import logging

import torch


def mps_is_available() -> bool:
    """
    Check whether Apple MPS is available in the current PyTorch install.

    Returns:
        True if `torch.backends.mps` exists and MPS is available.
    """
    return bool(
        getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
    )


def resolve_device(device: str = "cuda") -> torch.device:
    """
    Convert a string device identifier into a `torch.device`.

    Resolution order:
    1. CUDA — if `cuda`/`gpu`/`cuda:N` is requested and a GPU is available.
    2. MPS  — if `mps` is requested and the Apple Silicon backend is available.
    3. CPU  — in all other cases (including an unavailable backend).

    Args:
        device: Value from a config or CLI argument. Supported values:
            `cpu`, `cuda`, `cuda:0`, `gpu`, `mps`.

    Returns:
        A `torch.device` ready to pass to `model.to(device)`.
    """
    requested = str(device).strip().lower()

    # NVIDIA GPU: prefer when CUDA is explicitly requested and available.
    if requested in ("cuda", "gpu") or requested.startswith("cuda:"):
        if torch.cuda.is_available():
            return torch.device(requested if requested.startswith("cuda:") else "cuda")
        logging.warning("CUDA requested but not available, falling back")

    # Apple Silicon: Metal Performance Shaders (M4 Pro and similar).
    if requested == "mps":
        if mps_is_available():
            return torch.device("mps")
        logging.warning("MPS requested but not available, falling back to CPU")

    if requested not in ("cpu", "cuda", "mps", "gpu") and not requested.startswith("cuda:"):
        logging.warning("Unknown device '%s', falling back to CPU", device)

    return torch.device("cpu")
