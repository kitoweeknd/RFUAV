# RFUAV GPU test/runtime image.
#
# Base: reuse the CUDA 12.9.1 devel image already present on the box (zero pull).
# The actual compute stack is provided by the pip torch **cu130** wheels installed
# by `uv sync` below — those wheels bundle the CUDA 13 runtime and native sm_89
# (RTX 40-series) kernels, so the system CUDA label (12.9) is irrelevant to torch.
FROM nvcr.io/nvidia/cuda:12.9.1-devel-ubuntu22.04

# --- build-time proxy (host Clash on Docker Desktop) -------------------------
# github/pytorch/apt all route through the host proxy during build. Override at
# build time with `--build-arg HTTP_PROXY=...` if your proxy differs.
ARG HTTP_PROXY=http://host.docker.internal:7897
ARG HTTPS_PROXY=http://host.docker.internal:7897
ARG NO_PROXY=localhost,127.0.0.1
ENV HTTP_PROXY=${HTTP_PROXY} HTTPS_PROXY=${HTTPS_PROXY} NO_PROXY=${NO_PROXY} \
    http_proxy=${HTTP_PROXY} https_proxy=${HTTPS_PROXY} no_proxy=${NO_PROXY}

ENV DEBIAN_FRONTEND=noninteractive

# System libraries:
#   ffmpeg                         -> video subprocess in RawDataProcessor
#   libgl1 libglib2.0-0 libegl1    -> OpenCV / Qt runtime
#   libxkbcommon0 libdbus-1-3 fontconfig libxcb-* -> PySide6 offscreen platform
#   git curl ca-certificates       -> uv, wheel fetches
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 libglib2.0-0 libegl1 libxkbcommon0 libdbus-1-3 fontconfig \
        libxcb1 libxcb-cursor0 libxcb-render0 libxcb-shape0 libxcb-xfixes0 \
        git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# --- uv ----------------------------------------------------------------------
# Install uv via the official installer, through the build-time proxy set above.
# Deliberately NOT `COPY --from=ghcr.io/astral-sh/uv:latest` — pulling a registry
# image at build time makes BuildKit resolve registry metadata, which invokes
# Docker's credential helper. On Docker Desktop for Windows driven over a
# headless SSH session that helper fails hard ("error getting credentials ... A
# specified logon session does not exist"), and a client-side DOCKER_CONFIG
# override does not reach BuildKit's resolver. The install script instead pulls
# the uv binary over HTTPS (via the proxy) from GitHub releases — no registry
# auth, so the build works both over SSH and on a normal workstation.
RUN curl -LsSf https://astral.sh/uv/install.sh \
      | env UV_INSTALL_DIR=/usr/local/bin INSTALLER_NO_MODIFY_PATH=1 sh \
    && uv --version

# Environment lives OUTSIDE the mounted repo so the volume mount at
# /workspace/RFUAV cannot shadow it. copy link-mode avoids hardlink-across-fs
# warnings when the cache and target are on different layers.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    PATH=/opt/venv/bin:$PATH

WORKDIR /workspace/RFUAV

# Resolve + install dependencies only (project is package=false, so nothing to
# build). Copying just the manifest + lock keeps this layer cached across code
# changes. If the ~2.5 GB torch cu130 wheel stalls through the proxy, prefetch it
# resumably and install from the local file, e.g.:
#   curl -x $HTTP_PROXY -C - --retry 15 -L \
#     -o /tmp/torch.whl https://download.pytorch.org/whl/cu130/torch-<ver>-cp312-...whl
#   uv pip install --python /opt/venv /tmp/torch.whl
COPY pyproject.toml uv.lock* ./
RUN uv python install 3.12
RUN uv sync --all-extras --group dev --no-install-project \
    $( [ -f uv.lock ] && echo --frozen )

# --- runtime defaults --------------------------------------------------------
# Headless Qt + unbuffered stdout for live test logs. The repo itself is bind
# mounted at runtime (see docker/docker-compose.yml) rather than COPYed, so tests run
# against the live checkout.
ENV QT_QPA_PLATFORM=offscreen \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace/RFUAV

# Default: run the full functional suite. The `uv sync --frozen` first
# reconciles the baked /opt/venv against uv.lock — a fast no-op on a freshly
# built image, and a self-heal if the image ever lags the lock (e.g. a newly
# added pure-Python dep) — then `--no-sync` runs pytest against that env.
# docker/docker-compose.yml sets the same command; overriding it on the CLI (e.g.
# `uv run --no-sync python train.py ...`) skips the suite as expected.
CMD ["bash", "-lc", "uv sync --frozen --all-extras --group dev --no-install-project && uv run --no-sync pytest -v"]
