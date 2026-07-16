# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.3.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ── System packages + Python 3.11 (deadsnakes) ──────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common gpg-agent wget curl git && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

# Common packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 vim sqlite3 && \
    rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.10.0 /uv /uvx /bin/

WORKDIR /workspace

# Enable bytecode compilation and link mode for uv.
# UV_ENV_FILE: every `uv run` auto-loads /workspace/.env (absolute -> cwd-independent).
# Only the path is baked; the file is read at run time, so live edits take effect.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON=python3.11 \
    UV_ENV_FILE=/workspace/.env

# System cuDNN 9 for CUDA 12 (JAX cuda plugin expects it)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libcudnn9-cuda-12 libcudnn9-dev-cuda-12 && \
    rm -rf /var/lib/apt/lists/*

# Install 'just'
RUN curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to /usr/local/bin

# Remove forward-compat libcuda that can conflict with an older host driver.
# Harmless on newer drivers; keep for reproducibility on the original host.
RUN rm -rf /usr/local/cuda/compat && \
    rm -f /etc/ld.so.conf.d/*compat* || true && \
    ldconfig

# Land interactive shells (plain `docker exec`, VS Code attach, etc.) in /workspace.
# Editor-agnostic: only jumps when the shell starts at $HOME, so manual `cd` and
# subshells are left alone.
RUN printf '\n[ "$PWD" = "$HOME" ] && cd /workspace\n' >> /root/.bashrc

CMD ["/bin/bash"]
