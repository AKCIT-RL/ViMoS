# lowpass_mocap

Auto-tuned **low-pass filter** for the G1 mocap motions, calibrated so their
**mean jerk** matches the GenMo (video) source — plus a Table-VII-style metric
comparison **mocap vs mocap low-pass vs GenMo**.

Running `run_all.sh` regenerates everything under `output/` (git-ignored):
filtered CSVs/pkls, comparison videos, `report.md` (auto-tuning log) and
`metrics_table.md` (jerk, JL Viol, MPJPE, MAE for the three sources).

## The three sources (same 21 motions, Unitree G1)

| name | folder | fps | notes |
|---|---|--:|---|
| **mocap** | `../csv_g1_mocap/` = `../BVH_pkl/` | 59 / 239 | Xsens suit. The CSVs are byte-for-byte the dof/root data of the BVH pkls. |
| **mocap low-pass** | `output/csv/` , `output/pkl/` | 30 | this repo — see below |
| **GenMo** | `CSV_Genmo/` = `../retarget/GMR/output/{csv,pkl}/Paper_CoRL/` | 30 | video → GenMo → GMR |

## How the filter works (`lowpass_filter.py`)

Per motion:

1. read the mocap CSV at its **native fps** (59 Hz, or 239 Hz for `aceno`, `boxe`,
   `macarena`, `what_is_love` — from the BVH pkl metadata);
2. **zero-phase Butterworth low-pass** (order 4, `scipy.filtfilt`, no lag) on root
   translation (3), root quaternion (4, filtered component-wise + renormalized),
   and the 29 joint angles;
3. resample to **30 Hz** (linear for pos/joints, SLERP for the quaternion);
4. **binary-search the cutoff `fc`** so `jerk(filtered) ≈ jerk(GenMo)` for that
   motion (tol 2 %). Tuned cutoffs: **3.0–11 Hz** (`output/report.md`).

Outputs `output/csv/<name>.csv` (30 Hz, `[root_pos(3)|quat_xyzw(4)|dof(29)]`, no
header) and `output/pkl/unitree_g1_<stem>.pkl` (for the eval pipeline).

## How the table is built (`build_metrics_table.py`)

Runs the **unmodified** GMR `eval_retarget_metrics.py` (same script + settings that
produced `retarget/GMR/reports/BVH_vs_CoRL_normalized/`) — `xyzw/xyzw`,
heading-normalized, DTW aligned, `target_fps = min = 30`. Three pairs per motion:

| pair | mocap arg | genmo arg | gives |
|---|---|---|---|
| A | BVH (mocap) | Paper_CoRL | jerk & JLV of mocap + GenMo, MPJPE/MAE GenMo–mocap |
| B | low-pass | Paper_CoRL | jerk & JLV of low-pass, MPJPE/MAE GenMo–low-pass |
| C | BVH (mocap) | low-pass | MPJPE/MAE low-pass–mocap |

Raw reports in `output/eval/{A,B,C}_<name>.json`.

## Layout

```
lowpass_mocap/
├── lowpass_filter.py       # 1. filter + auto-tune           -> output/csv, output/pkl
├── build_metrics_table.py  # 3. GMR eval pipeline, 3 pairs    -> output/metrics_table.*
├── render_compare.py       # 4. 3-panel videos               -> output/videos
├── run_all.sh              # all steps (2 = refresh CSV_Genmo)
├── CSV_Genmo/              # copy of the GenMo CSVs, bare motion names
└── output/
    ├── csv/                # 21 low-pass CSVs @30 Hz
    ├── pkl/                # 21 low-pass pkls @30 Hz
    ├── eval/              # raw per-pair JSON from eval_retarget_metrics.py
    ├── videos/             # 21 three-panel MP4s (1920x480, 30fps), robot facing camera:
    │                       #   mocap | mocap low-pass | GenMo   (yaw-normalized;
    │                       #   GenMo is a different take, panels not time-synced)
    ├── report.md / .csv       # jerk auto-tuning log (cutoffs, before/after)
    └── metrics_table.md / .csv # THE table (jerk, JL Viol, MPJPE, MAE)
```

## Run

```bash
bash run_all.sh                       # all 21 motions, all steps
bash run_all.sh --only run boxe        # a subset
python build_metrics_table.py --force  # recompute metrics only
```

Uses the GMR venv (`../retarget/GMR/.venv/bin/python`); rendering needs
`MUJOCO_GL=egl` (set by the scripts). Filtered CSVs are 30 Hz — feed them to
`batch_csv_to_npz_mocap.sh` with `--input_fps 30`.
