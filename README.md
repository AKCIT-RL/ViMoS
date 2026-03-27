# CopyCat 

**CopyCat** is an end-to-end pipeline for human motion imitation and retargeting to robots, leveraging the GENMO and GMR frameworks. It enables the conversion of video input into robot motion, combining state-of-the-art motion generation (GENMO) and retargeting (GMR).

---

## Acknowledgements

CopyCat is built on top of:
- **GENMO** (Human motion generation)
- **GMR** (General Motion Retargeting)

Special thanks to the authors and maintainers of both projects. Please refer to their respective repositories for credits, documentation, and support.

---

## Setup

**Important:**
- You must follow the setup instructions in the README of each submodule (GENMO and GMR) to ensure all dependencies and environments are correctly installed.
- GENMO and GMR are included as git submodules. After cloning CopyCat, initialize and update them:

```bash
git submodule update --init --recursive
```

- Each submodule has its own Python environment and requirements. Refer to their README files for installation steps.

---

## Usage

All commands below should be run from the `CopyCat/` directory.

### Minimal Example — Single Video

```bash
python run_pipeline.py \
    --video /path/to/video.mp4 \
    --robot booster_t1
```

### Minimal Example — Folder of Videos

```bash
python run_pipeline.py \
    --videos_path /path/to/folder/ \
    --robot unitree_g1
```

### Full Example

```bash
python run_pipeline.py \
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
unitree_g1        unitree_g1_with_hands    unitree_h1       unitree_h1_2
booster_t1        booster_t1_29dof         stanford_toddy   fourier_n1
engineai_pm01     kuavo_s45                hightorque_hi    galaxea_r1pro
berkeley_humanoid_lite   booster_k1        pnd_adam_lite    openloong
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
CopyCat/
├── run_pipeline.py   ← main script
├── GENMO/            ← GENMO repository (submodule)
└── GMR/              ← GMR repository (submodule)
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

## License

See the LICENSE files in each submodule for licensing information.

---

## Contact

For questions or contributions, please refer to the GENMO and GMR repositories, or open an issue in CopyCat.
