# CopyCat 🐱

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

### Minimal Example

```bash
python run_pipeline.py \
    --video /path/to/video.mp4 \
    --robot booster_t1
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

### GENMO — Video to SMPL

| Flag | Default | Description |
|---|---|---|
| `--video` | *required* | Path to input video **or** folder containing `.mp4` files |
| `--video_name` | file stem | Name used for output folder (ignored if `--video` is a folder) |
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

| File | Description |
|---|---|
| `<output_dir>/hmr4d_results.pt` | GENMO SMPL estimation |
| `<output_dir>/robot_motion_<robot>.pkl` | Retargeted robot motion |
| `GMR/videos/<robot>_hmr4d_results.mp4` | Visualization video (if `--record_video` is used) |
| `<output_dir>/pose_debug_vid1_f0000.png` | Debug PNG of pose skeleton (if `--pose` is used) |
| `<output_dir>/yolo_debug_vid1_f0000.png` | Debug PNG of YOLO bounding box (if `--pose` is used) |

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
