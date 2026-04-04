# Do you still need Mocap? Comparative Study between Inertial Suit Driven Motion and Video Extraction Motion for Humanoid Control

**DYSNM** is an end-to-end pipeline for human motion imitation, retargeting and policy training for humanoids robots, leveraging the GENMO, GMR and Whole Body Tracking frameworks. It enables the conversion of video input into robot motion, combining state-of-the-art motion generation (GENMO), retargeting (GMR) and policy training for Booster T1 and Unitree G1 (Whole Body Tracking).

---

## Acknowledgements

DYSNM is built on top of:
- **GENMO** (Human motion generation)
- **GMR** (General Motion Retargeting)
- **whole_body_tracking** (Policy training)

Special thanks to the authors and maintainers of both projects. Please refer to their respective repositories for credits, documentation, and support.

---

## Setup

**Important:**
- You must follow the setup instructions in the README of each submodule (GENMO and GMR) to ensure all dependencies and environments are correctly installed.
- GENMO and GMR are included as git submodules. After cloning DYSNM, initialize and update them:

```bash
git submodule update --init --recursive
```

- Each submodule has its own Python environment and requirements. Refer to their README files for installation steps.

---

## Usage

All commands below should be run from the `DYSNM/` directory.

### Minimal Example — Single Video

```bash
python scripts/run_pipeline.py \
    --video /path/to/video.mp4 \
    --robot booster_t1
```

### Minimal Example — Folder of Videos

```bash
python scripts/run_pipeline.py \
    --videos_path /path/to/folder/ \
    --robot unitree_g1
```

### Full Example

```bash
python scripts/run_pipeline.py \
    --video /path/to/video.mp4 \
    --video_name my_dance \
    --robot booster_t1 \
    --ckpt_path GENMO/inputs/checkpoints/s050000.ckpt \
    --orig_fps 30 \
    --save_path outputs/my_dance/robot_motion.pkl \
    --record_video \
    --rate_limit
```

---

## Flags

### Input (mutually exclusive — exactly one required)

| Flag | Description |
|---|---|
| `--video` / `-v` | Path to a **single** `.mp4` file |
| `--videos_path` | Path to a **folder** of `.mp4` files (searched recursively) |

### GENMO — Video to SMPL

| Flag | Default | Description |
|---|---|---|
| `--video_name` | file stem | Name used for the output folder (ignored with `--videos_path`) |
| `--output_dir` | `GENMO/outputs/demo/<video_name>` | GENMO output directory |
| `--ckpt_path` | `GENMO/inputs/checkpoints/s050000.ckpt` | Model checkpoint |
| `--exp` | `genmo_lg` | Experiment configuration |
| `--orig_fps` | `30` | Original FPS of input video |
| `--static_cam` | enabled by default | Assumes static camera (no SLAM) |
| `--no_static_cam` | — | Enables SLAM for moving camera |
| `--pose` | disabled by default | Generates debug PNGs for YOLO bounding box and pose skeleton |

### GMR — SMPL to Robot

| Flag | Default | Description |
|---|---|---|
| `--robot` | `unitree_g1` | Target robot for retargeting (see list below) |
| `--save_path` | `<output_dir>/robot_motion_<robot>.pkl` | Where to save robot motion |
| `--record_video` | disabled | Save a visualization video |
| `--rate_limit` | disabled | Limit playback rate to human FPS |
| `--loop` | disabled | Repeat motion indefinitely in viewer |
| `--headless` | disabled | Run GMR/MuJoCo in headless mode (no window, for SSH/servers) |

### Supported Robots

```
unitree_g1        unitree_g1_with_hands    unitree_h1       unitree_h1_2
booster_t1        booster_t1_29dof         stanford_toddy   fourier_n1
engineai_pm01     kuavo_s45                hightorque_hi    galaxea_r1pro
berkeley_humanoid_lite   booster_k1        pnd_adam_lite    openloong
tienkung
```

---

## Outputs

The pipeline runs three stages — **GENMO → GMR → PKL-to-CSV** — and produces the following files:

| File | Description |
|---|---|
| `GENMO/outputs/demo/<video_name>/hmr4d_results.pt` | GENMO SMPL pose estimation |
| `GMR/output/pkl/<robot>_<video_name>.pkl` | Retargeted robot motion (PKL) |
| `GMR/output/csv/<robot>_<video_name>.csv` | Retargeted robot motion (CSV, joints over time) |
| `GMR/videos/<robot>_<video_name>.mp4` | Visualization video (only if `--record_video` is used) |
| `GENMO/outputs/demo/<video_name>/pose_debug_vid1_f0000.png` | Debug PNG — pose skeleton (only if `--pose` is used) |
| `GENMO/outputs/demo/<video_name>/yolo_debug_vid1_f0000.png` | Debug PNG — YOLO bounding box (only if `--pose` is used) |

---

## Structure

```
DYSNM/
├── scripts/
│   ├── run_pipeline.py          ← main pipeline (venv)
│   ├── run_pipeline_refpose.py  ← pipeline with sandwich mode (venv)
│   └── run_pipeline_docker.py   ← pipeline via Docker (see section below)
├── docker-compose.yml           ← Docker services definition
├── GENMO/                       ← GENMO repository (submodule)
│   ├── Dockerfile
│   └── scripts/sandwich_runner.py
└── GMR/                         ← GMR repository (submodule)
    └── Dockerfile
```

---

## Troubleshooting

- If you encounter errors, check the README and issues for GENMO and GMR first.
- Make sure all environments are activated and dependencies installed as described in each submodule.
- For submodule updates:
  ```bash
  git submodule update --remote --checkout
  ```

---

## Docker

GENMO and GMR can be run in separate containers communicating via Docker Compose, replicating the behavior of `run_pipeline_refpose.py`.

### Prerequisites

- Docker Engine 20.10+
- [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed and configured
- Data in place before building:
  - `GENMO/inputs/checkpoints/s050000.ckpt`
  - `GMR/assets/` (robot models and SMPL-X body models)

### Build images

```bash
# Build both images (only needed once, or after code changes)
docker compose build

# Build individually
docker compose build genmo
docker compose build gmr
```

### Usage — identical to run_pipeline_refpose.py

```bash
# Single video
python scripts/run_pipeline_docker.py --video /path/to/video.mp4 --robot booster_t1

# Folder of videos
python scripts/run_pipeline_docker.py --videos_path fut_do_t1/ --robot unitree_g1

# Without sandwich mode
python scripts/run_pipeline_docker.py --video video.mp4 --robot booster_t1 --no-sandwich

# With pose debug
python scripts/run_pipeline_docker.py --video video.mp4 --robot booster_t1 --pose

# Record GMR visualization video
python scripts/run_pipeline_docker.py --video video.mp4 --robot booster_t1 --record_video
```

> **Note:** `--headless` is ignored in Docker mode — the GMR container always uses `MUJOCO_GL=egl` (headless EGL rendering).

### Docker Architecture

| Service | Image | Role |
|---|---|---|
| `genmo` | `dysnm-genmo:latest` | SMPL-X inference from video (requires GPU/CUDA 12.1) |
| `gmr` | `dysnm-gmr:latest` | SMPL-X → robot motion retargeting (headless via EGL) |

**Shared volumes** (bind mounts relative to `DYSNM/`):

| Host | genmo container | gmr container | Purpose |
|---|---|---|---|
| `GENMO/inputs/` | `/app/inputs` (ro) | — | Checkpoints |
| `GENMO/outputs/` | `/app/outputs` (rw) | `/genmo_outputs` (ro) | GENMO outputs → GMR inputs |
| `GMR/assets/` | — | `/app/assets` (ro) | Robot models and SMPL-X body models |
| `GMR/output/` | — | `/app/output` (rw) | PKL and CSV outputs |
| `GMR/videos/` | — | `/app/videos` (rw) | Visualization videos |

Sandwich mode (`--sandwich`, enabled by default) runs `sandwich_runner.py` **inside the GENMO container** (which has torch available), with no torch requirement on the host.

### Docker Limitations

- **Dynamic camera (SLAM):** `--no-static-cam` requires DPVO compiled inside the image, which is not included by default. Use the default `--static-cam` mode.
- **Custom output paths:** `--output_dir` and `--save_path` outside the default directories are supported via automatic extra volumes, but staying with default paths is recommended.

---

## License

See the LICENSE files in each submodule for licensing information.

---

## Contact

For questions or contributions, please refer to the GENMO and GMR repositories, or open an issue in DYSNM.
