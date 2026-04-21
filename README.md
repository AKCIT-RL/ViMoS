# ViMoS — Do You Still Need Mocap?

**[[Project Website]](https://akcit-rl.github.io/ViMoS/)** | **[[Paper]](https://arxiv.org/abs/)** <!-- TODO: update arXiv link -->

**ViMoS** is an end-to-end pipeline for humanoid robot motion imitation — from raw video to deployed policy on a real robot. It chains four stages: **video → SMPL pose (GENMO) → robot motion (GMR) → policy training (BeyondMimic) → deploy**.

Supports **Booster T1** and **Unitree G1** (and 15+ other humanoids via GMR).

---

## Pipeline Overview

```
Video (.mp4)
    │
    ▼
[Stage 1] GENMO — Video → SMPL-X pose estimation
    │  retarget/GENMO/
    │
    ▼
[Stage 2] GMR — SMPL-X → Robot joint motion (CSV/PKL)
    │  retarget/GMR/
    │
    ▼
[Stage 3] whole_body_tracking — CSV → NPZ → Policy training (Isaac Lab / PPO)
    │  train/whole_body_tracking/
    │  exports: ONNX (G1) or TorchScript JIT (T1)
    │
    ▼
[Stage 4] Deploy — Policy inference on MuJoCo sim or real robot (ROS 2)
           deploy/motion_tracking_controller/   ← G1 (ONNX)
           deploy/booster_deploy/               ← T1 (JIT)
```

---

## Repository Structure

```
ViMoS/
├── scripts/
│   ├── video_to_robot.py   ← main pipeline (venv)
│   └── video_to_robot_docker.py   ← pipeline via Docker
├── retarget/
│   ├── GENMO/                   ← video → SMPL-X (submodule)
│   └── GMR/                     ← SMPL-X → robot motion (submodule)
├── train/
│   └── whole_body_tracking/     ← BeyondMimic / Isaac Lab (submodule)
├── deploy/
│   ├── motion_tracking_controller/  ← ROS 2 inference — G1 (submodule)
│   └── booster_deploy/              ← T1 deploy (submodule)
└── docker-compose.yml
```

---

## Setup

### Clone with submodules

```bash
git clone --recurse-submodules https://github.com/AKCIT-RL/ViMoS.git
cd ViMoS
```

Or after cloning:
```bash
git submodule update --init --recursive
```

### Environment setup

Each stage has its own environment. You have two options:

**Option A — Manual setup (per submodule README):**

| Stage | Submodule | Setup guide |
|---|---|---|
| Retarget (GENMO) | `retarget/GENMO/` | [GENMO README](retarget/GENMO/README.md) — uv venv (`uv venv --python 3.10`) |
| Retarget (GMR) | `retarget/GMR/` | [GMR README](retarget/GMR/README.md) — uv venv (`uv venv --python 3.10`) |
| Training | `train/whole_body_tracking/` | [WBT README](train/whole_body_tracking/README.md) — conda Isaac Lab 2.1 / Isaac Sim 4.5 |
| Deploy G1 | `deploy/motion_tracking_controller/` | [MTC README](deploy/motion_tracking_controller/README.md) — ROS 2 Jazzy |
| Deploy T1 | `deploy/booster_deploy/` | [Booster Deploy README](deploy/booster_deploy/README.md) |

**Option B — Docker (recommended for Stages 1–3):**

```bash
docker compose build              # GENMO + GMR
docker compose --profile wbt build   # + whole_body_tracking (Isaac Lab)
docker compose --profile mtc build  # + motion_tracking_controller (ROS 2)
```

> Requires Docker Engine 20.10+, [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html), and `GENMO/inputs/checkpoints/s050000.ckpt`.

---

## Stage 1 + 2 — Video → Robot Motion

### With Docker (recommended)

```bash
# Single video
python scripts/video_to_robot_docker.py --video /path/to/video.mp4 --robot booster_t1

# Folder of videos
python scripts/video_to_robot_docker.py --videos_path /path/to/folder/ --robot unitree_g1

# With visualization video and headless mode (no display, for SSH)
python scripts/video_to_robot_docker.py --video video.mp4 --robot booster_t1 --record_video --headless
```

### With venv (manual)

```bash
python scripts/video_to_robot.py --video /path/to/video.mp4 --robot booster_t1
```

Activate the GENMO venv (`retarget/GENMO/.venv`) before running. The script manages both internally via `GENMO_PYTHON` and `GMR_PYTHON`.

### Sandwich mode (default: disabled)

Some videos start or end abruptly — mid-motion, mid-kick, or in an unstable pose. When the robot trains on this reference as-is, it may learn to not maintain balance in the final pose, since the reference itself never shows a stable recovery.

Sandwich mode wraps the motion with "stand still" anchor frames at both ends:

```
[30 frames stand still] + [motion from video] + [60 frames stand still]
```

This gives the policy a clear signal for what a safe start and end look like.

**Use it when:** the video starts/ends in a pose the robot cannot safely hold (e.g. mid-kick, leaning, one foot up).  
**Skip it when:** the video already starts and ends with the robot standing still.

To enable:
```bash
python scripts/video_to_robot.py --video video.mp4 --robot booster_t1 --sandwich
```

### Outputs

| File | Description |
|---|---|
| `retarget/GENMO/outputs/demo/<name>/hmr4d_results.pt` | SMPL-X pose estimation |
| `retarget/GMR/output/pkl/<robot>_<name>.pkl` | Retargeted robot motion |
| `retarget/GMR/output/csv/<robot>_<name>.csv` | Robot motion as CSV (joints × frames) |
| `retarget/GMR/videos/<robot>_<name>.mp4` | Visualization (with `--record_video`) |

### Supported robots

```
booster_t1   booster_t1_29dof   booster_k1
unitree_g1   unitree_g1_with_hands   unitree_h1   unitree_h1_2
fourier_n1   engineai_pm01   kuavo_s45   hightorque_hi
galaxea_r1pro   stanford_toddy   berkeley_humanoid_lite
pnd_adam_lite   openloong   tienkung
```

---

## Stage 3 — Policy Training

Uses **BeyondMimic** ([paper](https://arxiv.org/abs/2508.08241)) — PPO in Isaac Lab with adaptive sampling and anchor-frame motion tracking. The same reward formulation works for any motion without parameter tuning.

### Prerequisites

- NVIDIA GPU (RTX 3090+ recommended)
- Isaac Lab 2.1 / Isaac Sim 4.5 (or Docker with NGC login)
- W&B account for artifact registry

### Step 1 — Convert CSV → NPZ and upload to W&B

```bash
cd train/whole_body_tracking

# Booster T1
python scripts/csv_to_npz.py \
  --input_file /path/to/motion.csv \
  --input_fps 30 \
  --output_name my_motion \
  --robot booster_t1 \
  --wandb_project Booster_t1 \
  --headless

# Unitree G1
python scripts/csv_to_npz.py \
  --input_file /path/to/motion.csv \
  --input_fps 30 \
  --output_name my_motion \
  --wandb_project G1_motions \
  --headless
```

### Step 2 — Train

**Booster T1:**
```bash
python scripts/rsl_rl/train.py \
  --task=Tracking-Flat-T1-Wo-State-Estimation-v0 \
  --num_envs 4096 \
  --headless --video \
  --logger wandb \
  --log_project_name <project_name> \
  --registry_name <entity>/<project_name>/<motion_name>:latest \
  --experiment_name <motion_name> \
  --run_name <motion_name>
```

**Unitree G1:**
```bash
python scripts/rsl_rl/train.py \
  --task=Tracking-Flat-G1-v0 \
  --num_envs 4096 \
  --headless --video \
  --logger wandb \
  --log_project_name <project_name> \
  --registry_name <entity>/<project_name>/<motion_name>:latest \
  --experiment_name <motion_name> \
  --run_name <motion_name>
```

### Resume training from checkpoint

```bash
python scripts/rsl_rl/train.py \
  --task=Tracking-Flat-T1-Wo-State-Estimation-v0 \
  --num_envs 4096 \
  --headless --video \
  --logger wandb \
  --log_project_name <project_name> \
  --registry_name <entity>/<project_name>/<motion_name>:latest \
  --experiment_name <motion_name> \
  --run_name <motion_name> \
  --resume 1 \
  --load_run <run_folder_name> \
  --checkpoint model_XXXXX.pt
```

### Available tasks

| Robot | Task | Notes |
|---|---|---|
| Booster T1 | `Tracking-Flat-T1-Wo-State-Estimation-v0` | Use this — hardware has no state estimator |
| Unitree G1 | `Tracking-Flat-G1-v0` | Full observations |
| Unitree G1 | `Tracking-Flat-G1-Wo-State-Estimation-v0` | Without state estimation |
| Unitree G1 | `Tracking-Flat-G1-Low-Freq-v0` | Low-frequency control |

### Step 3 — Export policy

```bash
# Booster T1 — exports TorchScript JIT (.pt)
python scripts/rsl_rl/play.py \
  --task=Tracking-Flat-T1-Wo-State-Estimation-v0 \
  --num_envs 1 \
  --headless \
  --wandb_path <entity>/<project_name>/<run_id>

# Unitree G1 — exports ONNX (.onnx)
python scripts/rsl_rl/play.py \
  --task=Tracking-Flat-G1-v0 \
  --num_envs 1 \
  --headless \
  --wandb_path <entity>/<project_name>/<run_id>
```

Exported files go to `logs/rsl_rl/temp/exported/`.

### Batch training (T1)

```bash
export WANDB_ENTITY=<your-entity>
bash scripts/batch_csv_train.sh
```

Edit the motion list inside `scripts/batch_csv_train.sh` before running.

---

## Stage 4 — Deploy

### Booster T1 (TorchScript JIT)

See [deploy/booster_deploy/README.md](deploy/booster_deploy/README.md).

Place the exported `.pt` file as required by the deploy package and follow the startup instructions.

### Unitree G1 (ONNX — ROS 2)

```bash
# MuJoCo simulation
ros2 launch motion_tracking_controller mujoco.launch.py \
  policy_path:=/path/to/policy.onnx

# Real robot (use with extreme care)
ros2 launch motion_tracking_controller real.launch.py \
  policy_path:=/path/to/policy.onnx
```

Via Docker:
```bash
xhost +local:docker
docker compose --profile mtc run --rm motion-tracking-controller \
  bash -c "ros2 launch motion_tracking_controller mujoco.launch.py policy_path:=/workspace/data/policy.onnx"
```

> **Safety:** Running policies on a real robot is dangerous and for research only. See the disclaimer in [deploy/motion_tracking_controller/README.md](deploy/motion_tracking_controller/README.md).

---

## T1 vs G1 — Quick Reference

| | Booster T1 | Unitree G1 |
|---|---|---|
| Retarget flag | `--robot booster_t1` | `--robot unitree_g1` |
| csv_to_npz flag | `--robot booster_t1` | _(default)_ |
| Training task | `Tracking-Flat-T1-Wo-State-Estimation-v0` | `Tracking-Flat-G1-v0` |
| Export format | TorchScript JIT (`.pt`) | ONNX (`.onnx`) |
| Deploy | `deploy/booster_deploy/` | `deploy/motion_tracking_controller/` |
| Max iterations | 15.000 | 30.000 |

---

## Docker — All Services

| Service | Profile | Description |
|---|---|---|
| `genmo` | _(default)_ | Video → SMPL-X |
| `gmr` | _(default)_ | SMPL-X → robot motion |
| `whole-body-tracking` | `wbt` | Isaac Lab training |
| `motion-tracking-controller` | `mtc` | ROS 2 G1 deploy |

> For `wbt`: run `docker login nvcr.io` before the first build (NGC API key required).

---

## Troubleshooting

- **Empty submodules after clone:** `git submodule update --init --recursive`
- **Update submodules:** `git submodule update --remote --checkout`
- **Errors in each stage:** check the corresponding submodule README
- **GENMO can't find checkpoint:** place it at `retarget/GENMO/inputs/checkpoints/s050000.ckpt`
- **GMR no display (SSH):** use `--headless` or Docker (MUJOCO_GL=egl set automatically)
- **W&B artifact not found:** verify `WANDB_ENTITY` and the exact artifact name with `wandb artifact ls <entity>/<project>`

---

## Acknowledgements

ViMoS integrates:
- **[GENMO](retarget/GENMO/)** — Human motion generation from video
- **[GMR](retarget/GMR/)** — General Motion Retargeting (Yanjie Ze et al.)
- **[BeyondMimic / whole_body_tracking](train/whole_body_tracking/)** — Motion tracking policy training (HybridRobotics)
- **[motion_tracking_controller](deploy/motion_tracking_controller/)** — ROS 2 deployment
- **[booster_deploy](deploy/booster_deploy/)** — Booster T1 deployment

Please refer to each submodule for credits, licenses, and upstream documentation.

---

## License

See the LICENSE files in each submodule.
