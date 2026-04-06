# Do you still need Mocap? Comparative Study between Inertial Suit Driven Motion and Video Extraction Motion for Humanoid Control

**DYSNM** is an end-to-end pipeline for human motion imitation, retargeting and policy training for humanoids robots, leveraging the GENMO, GMR and Whole Body Tracking frameworks. It enables the conversion of video input into robot motion, combining state-of-the-art motion generation (GENMO), retargeting (GMR) and policy training for Booster T1 and Unitree G1 (Whole Body Tracking).

---

## Acknowledgements

DYSNM is built on top of:
- **GENMO** (Human motion generation)
- **GMR** (General Motion Retargeting)
- **whole_body_tracking** (BeyondMimic — policy training in simulation)
- **motion_tracking_controller** (ROS 2 deployment — sim and real robot inference)

Special thanks to the authors and maintainers of these projects. Please refer to their respective repositories for credits, documentation, and support.

---

## Setup

**Important:**
- You must follow the setup instructions in the README of **each** submodule you use (GENMO, GMR, `whole_body_tracking`, `motion_tracking_controller`) so dependencies and environments match upstream.
- These repositories are included as **git submodules**. After cloning DYSNM, initialize and update all of them:

```bash
git submodule update --init --recursive
```

- Each submodule has its own environment (Python, ROS 2, Isaac Lab, etc.). For Docker-based BeyondMimic training, see [whole_body_tracking/README_DOCKER.md](whole_body_tracking/README_DOCKER.md) (NGC login, GPU, and W&B).

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

## BeyondMimic: after retargeting 

After **GENMO → GMR**, CSV motion files live under `GMR/output/csv/` (see **Outputs** above). The **whole_body_tracking** submodule converts retargeted CSV into training assets, trains policies in Isaac Lab, and exports ONNX; **motion_tracking_controller** runs that policy in MuJoCo or on hardware (ROS 2).

**Flow:** GMR CSV → `whole_body_tracking` (`csv_to_npz.py`, training, export) → ONNX → `motion_tracking_controller` (`mujoco.launch.py` / `real.launch.py`).

### whole_body_tracking (training)

- Requires Isaac Lab 2.1 / Isaac Sim 4.5 class stack, a capable NVIDIA GPU, and (for Docker) `docker login nvcr.io` with an NGC API key before the first image build.
- From the **DYSNM repo root**, using the root [docker-compose.yml](docker-compose.yml) and profile `wbt`:

```bash
docker compose --profile wbt build
docker compose --profile wbt run --rm whole-body-tracking
```

- The WBT image uses `ENTRYPOINT ["/bin/bash"]`. For a one-off command, pass **`-c '...'`** to that shell (do **not** prefix with `bash -c`, or you get `cannot execute binary file`). Example: `docker compose --profile wbt run -T --rm whole-body-tracking -c "echo ok"`. See [whole_body_tracking/README_DOCKER.md](whole_body_tracking/README_DOCKER.md).
- Inside the container, typical commands include `python scripts/csv_to_npz.py ...`, `python scripts/rsl_rl/train.py ...`, and `play.py` — see [whole_body_tracking/README.md](whole_body_tracking/README.md) and [whole_body_tracking/README_DOCKER.md](whole_body_tracking/README_DOCKER.md) for W&B registry, batch CSV training, and troubleshooting.
- To use CSV files from the host, bind-mount a directory when running, for example:

```bash
docker compose --profile wbt run --rm \
  -v /path/on/host/motions:/workspace/motions:ro \
  whole-body-tracking
```

- The submodule’s own [whole_body_tracking/docker-compose.yaml](whole_body_tracking/docker-compose.yaml) is kept for standalone clones of that repository; in DYSNM the **canonical** Compose file is at the repo root.

### motion_tracking_controller (deployment)

- Native install follows [motion_tracking_controller/README.md](motion_tracking_controller/README.md) (ROS 2 Jazzy, `legged_control2`, Unitree packages).
- **Docker (from DYSNM root, profile `mtc`):** build the image, allow local Docker clients to use your X11 display, then run interactively:

```bash
xhost +local:docker   # revoke later with: xhost -local:docker
docker compose --profile mtc build
docker compose --profile mtc run --rm motion-tracking-controller
```

- Place a policy at `motion_tracking_controller/data/policy.onnx` (host path; mounted read-write at `/workspace/data` in the container), or pass extra `-v` mounts and launch arguments as in the submodule README, for example:

```bash
docker compose --profile mtc run --rm motion-tracking-controller \
  bash -c "ros2 launch motion_tracking_controller mujoco.launch.py policy_path:=/workspace/data/policy.onnx"
```

> **Safety (real robot):** Running policies on a physical robot is **dangerous** and for **research only**. See the disclaimer in the motion_tracking_controller README. Use `real.launch.py` only with proper safeguards.

---

## Structure

```
DYSNM/
├── scripts/
│   ├── run_pipeline.py          ← main pipeline (venv)
│   ├── run_pipeline_refpose.py  ← pipeline with sandwich mode (venv)
│   └── run_pipeline_docker.py   ← pipeline via Docker (see section below)
├── docker-compose.yml           ← all services (GENMO, GMR, WBT, MTC — see Docker)
├── GENMO/                       ← GENMO repository (submodule)
│   ├── Dockerfile
│   └── scripts/sandwich_runner.py
├── GMR/                         ← GMR repository (submodule)
│   └── Dockerfile
├── whole_body_tracking/                   ← BeyondMimic / Isaac Lab (submodule)
└── motion_tracking_controller/         ← ROS 2 inference (submodule); profile mtc
```

---

## Troubleshooting

- If you encounter errors, check the README and issues for the relevant submodule (GENMO, GMR, whole_body_tracking, motion_tracking_controller) first.
- Make sure all environments are activated and dependencies installed as described in each submodule.
- If new submodules are missing after clone: `git submodule update --init --recursive`.
- For submodule updates:
  ```bash
  git submodule update --remote --checkout
  ```

---

## Docker

GENMO and GMR can be run in separate containers communicating via Docker Compose, replicating the behavior of `run_pipeline_refpose.py`.

### Docker: single compose file at repo root

All services are defined in [docker-compose.yml](docker-compose.yml) at the **DYSNM** root. You do **not** need to `cd` into each submodule to use Compose.

| Service | Profile | When it is built |
| --- | --- | --- |
| `genmo`, `gmr` | _(none)_ | `docker compose build` (default) |
| `whole-body-tracking` | `wbt` | `docker compose --profile wbt build` |
| `motion-tracking-controller` | `mtc` | `docker compose --profile mtc build` |

- **Do not** rely on `docker compose up` to “start the whole project”: GENMO+GMR are driven by `docker compose run` via `scripts/run_pipeline_docker.py`; Isaac Lab and ROS 2 are long-running or interactive stacks. Use **`compose run`** (or the Python script) per stage.
- **whole_body_tracking:** before the first WBT build, run `docker login nvcr.io` (NGC API key; see [whole_body_tracking/README_DOCKER.md](whole_body_tracking/README_DOCKER.md)).
- **motion_tracking_controller:** use `xhost +local:docker` and a valid `DISPLAY` when you need MuJoCo or GUI (see [BeyondMimic: after retargeting](#beyondmimic-after-retargeting-optional)).

### Prerequisites

- Docker Engine 20.10+
- [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed and configured
- Data in place before building:
  - `GENMO/inputs/checkpoints/s050000.ckpt`
  - `GMR/assets/` (robot models and SMPL-X body models)

### Build images

```bash
# Default: GENMO + GMR only (no Isaac / ROS extra images)
docker compose build

# Optional stacks (see table above)
docker compose --profile wbt build
docker compose --profile mtc build

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
| `genmo` | `copycat-genmo:latest` | SMPL-X inference from video (requires GPU/CUDA 12.1) |
| `gmr` | `copycat-gmr:latest` | SMPL-X → robot motion retargeting (headless via EGL) |
| `whole-body-tracking` | `whole-body-tracking:latest` | BeyondMimic / Isaac Lab (profile `wbt`; NGC base image) |
| `motion-tracking-controller` | `motion-tracking-controller:latest` | ROS 2 Jazzy controller (profile `mtc`; X11 for simulation) |

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

For questions or contributions, please refer to the GENMO, GMR, whole_body_tracking, and motion_tracking_controller repositories as appropriate, or open an issue in DYSNM.
