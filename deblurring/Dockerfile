FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive

# Common packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    vim \
    wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.10.0 /uv /uvx /bin/

# Install 'just'
RUN curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to /usr/local/bin

WORKDIR /workspace

# uv settings.
# UV_ENV_FILE: every `uv run` auto-loads /workspace/.env (absolute -> cwd-independent).
#   Only the path is baked; the file is read at run time, so live edits take effect.
# UV_LINK_MODE=copy: .venv lives on a named volume, a different fs than uv's cache,
#   so hardlinking would fail; copy instead.
ENV UV_ENV_FILE=/workspace/.env \
    UV_LINK_MODE=copy

# Land interactive shells (plain `docker exec`, VS Code attach, etc.) in /workspace.
# Editor-agnostic: only jumps when the shell starts at $HOME, so manual `cd` and
# subshells are left alone.
RUN printf '\n[ "$PWD" = "$HOME" ] && cd /workspace\n' >> /root/.bashrc

CMD ["/bin/bash"]
