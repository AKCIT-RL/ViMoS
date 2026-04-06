# ─────────────────────────────────────────────────────────────────────────────
# GMR — General Motion Retargeting for Humanoid Robots
# Headless: no GLFW viewer, no display. Offscreen rendering via EGL (GPU).
# MUJOCO_GL=egl is set in docker-compose.yml.
# ─────────────────────────────────────────────────────────────────────────────
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    # Python 3.10
    python3.10 python3.10-dev python3.10-distutils python3-pip \
    # MuJoCo EGL (offscreen rendering with GPU)
    libegl1-mesa libegl1 \
    libgles2-mesa \
    libglvnd-dev libglvnd0 \
    # Video
    ffmpeg \
    # Build tools (qpsolvers[proxqp] requires cmake + C++)
    build-essential cmake \
    # Utilities
    libgomp1 libstdc++6 \
    git wget curl \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default
RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.10 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
 && update-alternatives --install /usr/bin/pip     pip     /usr/bin/pip3       1

WORKDIR /app

# ── Install GMR and dependencies ──────────────────────────────────────────────
COPY setup.py README.md ./
COPY general_motion_retargeting/ ./general_motion_retargeting/

RUN pip install --upgrade pip setuptools wheel \
 && pip install -e .

# ── Copy scripts and third_party ─────────────────────────────────────────────
COPY scripts/ ./scripts/
COPY third_party/ ./third_party/

# ── Runtime directories (populated via bind mounts in compose) ────────────────
RUN mkdir -p output/pkl output/csv videos assets

# ── PYTHONPATH: include third_party/poselib (no own setup.py) ────────────────
ENV PYTHONPATH="/app:/app/third_party"

WORKDIR /app
