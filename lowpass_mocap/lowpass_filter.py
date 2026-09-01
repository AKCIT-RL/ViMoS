#!/usr/bin/env python3
"""
Low-pass filter for G1 mocap CSVs, auto-tuned so the filtered mean-jerk
approaches the GenMo (video) reference from GMR/output/csv/Paper_CoRL.

Pipeline per motion:
  1. read mocap CSV at its native fps (60 Hz, or 240 Hz for aceno/boxe/macarena/what_is_love)
  2. zero-phase Butterworth low-pass (scipy.filtfilt, order 4) on:
        - root translation (3)
        - root orientation (quaternion xyzw, filtered component-wise + renormalized)
        - 29 joint angles
  3. resample to OUTPUT_FPS (30 Hz, same as the GenMo CSVs) -- linear for
     pos/joints, SLERP for the root quaternion
  4. binary-search the cutoff frequency fc so that
        mean_jerk(filtered joints @ 30 Hz)  ~=  mean_jerk(GenMo joints @ 30 Hz)

Jerk is the exact metric used by GMR (general_motion_retargeting/evaluation/joint_metrics.py):
third time-derivative of the joint angles via np.gradient(edge_order=2),
L2 norm across joints, averaged over frames. Units: rad/s^3.

CSV layout (no header, 36 columns): [root_pos(3) | root_quat_xyzw(4) | dof(29)]
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation, Slerp

# --------------------------------------------------------------------------- #
# paths / constants
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
COPYCAT = os.path.abspath(os.path.join(HERE, ".."))

MOCAP_DIR = os.path.join(COPYCAT, "csv_g1_mocap")
GENMO_DIR = os.path.join(COPYCAT, "retarget", "GMR", "output", "csv", "Paper_CoRL")

# the mocap CSVs are byte-for-byte the dof/root data of these pkls (verified),
# which is what the paper's Table VII "Suit" column was computed from.
BVH_PKL_DIR = os.path.join(COPYCAT, "BVH_pkl")
GENMO_PKL_DIR = os.path.join(COPYCAT, "retarget", "GMR", "output", "pkl", "Paper_CoRL")

OUT_DIR = os.path.join(HERE, "output")
OUT_CSV_DIR = os.path.join(OUT_DIR, "csv")
OUT_PKL_DIR = os.path.join(OUT_DIR, "pkl")

OUTPUT_FPS = 30.0          # match the GenMo / beyondmimic CSVs
GENMO_FPS = 30.0           # GMR CSVs are downsampled to 30 Hz by batch_gmr_pkl_to_csv.py
# native fps taken from the BVH pkl metadata (Xsens 60/240 Hz recorded as 59/239)
DEFAULT_INPUT_FPS = 59.0
INPUT_FPS_OVERRIDE = {"aceno": 239.0, "boxe": 239.0, "macarena": 239.0, "what_is_love": 239.0}

FILTER_ORDER = 4
FC_LO, FC_HI = 0.5, 14.5   # Hz  (Nyquist at 30 Hz output is 15 Hz)
TUNE_TOL = 0.02            # stop when |jerk - target| / target < 2 %
TUNE_ITERS = 40

# mocap motion name  ->  GenMo file stem (retarget/.../Paper_CoRL/unitree_g1_<stem>.csv)
NAME_MAP = {
    "aceno": "aceno", "boxe": "boxe", "bunny_hop": "bunny_hop", "cr7": "cr7",
    "drunk": "drunk", "get_Down_get_Up": "get_Down_get_Up", "jump": "jump",
    "jumping_puddle": "jumping_puddle", "like_jennie": "like_jennie",
    "macarena": "macarena", "one_foot_balance": "one_foot_balance",
    "paradinha": "paradinha", "pedalada": "pedalada", "polichinelo": "polichinelo",
    "run": "run", "side_walking": "side_walking", "square": "square",
    "stealth": "stealth", "trivela": "trivela_side", "trote": "trot",
    "what_is_love": "what_is_love",
}

ROOT_POS = slice(0, 3)
ROOT_QUAT = slice(3, 7)     # xyzw
DOF = slice(7, 36)


# --------------------------------------------------------------------------- #
# core helpers
# --------------------------------------------------------------------------- #
def load_csv(path: str) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", dtype=np.float64)


def jerk_mean(q: np.ndarray, fps: float) -> float:
    """Exact GMR jerk_l2_mean (joint_metrics.jerk_metrics)."""
    dt = 1.0 / fps
    if q.shape[0] < 4:
        return 0.0
    d1 = np.gradient(q, dt, axis=0, edge_order=2)
    d2 = np.gradient(d1, dt, axis=0, edge_order=2)
    d3 = np.gradient(d2, dt, axis=0, edge_order=2)
    return float(np.mean(np.linalg.norm(d3, axis=1)))


def _quat_hemisphere(quat: np.ndarray) -> np.ndarray:
    """Flip signs so consecutive quaternions stay on the same hemisphere."""
    q = quat.copy()
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0.0:
            q[i] = -q[i]
    return q


def _butter_lp(x: np.ndarray, fc: float, fs: float, order: int = FILTER_ORDER) -> np.ndarray:
    """Zero-phase Butterworth low-pass along axis 0 (columns filtered independently)."""
    wn = min(fc / (0.5 * fs), 0.999)
    b, a = butter(order, wn, btype="low")
    padlen = 3 * max(len(a), len(b))
    if x.shape[0] <= padlen:
        padlen = x.shape[0] - 1
    return filtfilt(b, a, x, axis=0, padlen=padlen)


def _resample_linear(x: np.ndarray, t_in: np.ndarray, t_out: np.ndarray) -> np.ndarray:
    return np.stack([np.interp(t_out, t_in, x[:, i]) for i in range(x.shape[1])], axis=1)


def filter_and_resample(traj: np.ndarray, fs_in: float, fc: float,
                        fs_out: float = OUTPUT_FPS) -> np.ndarray:
    """Filter at native fps, then resample to fs_out. Returns (M, 36)."""
    pos = traj[:, ROOT_POS]
    quat = _quat_hemisphere(traj[:, ROOT_QUAT])
    dof = traj[:, DOF]

    pos_f = _butter_lp(pos, fc, fs_in)
    dof_f = _butter_lp(dof, fc, fs_in)
    quat_f = _butter_lp(quat, fc, fs_in)
    quat_f /= np.linalg.norm(quat_f, axis=1, keepdims=True)

    n = traj.shape[0]
    dur = (n - 1) / fs_in
    t_in = np.arange(n) / fs_in
    m = int(round(dur * fs_out)) + 1
    t_out = np.arange(m) / fs_out
    t_out[-1] = min(t_out[-1], t_in[-1])

    pos_r = _resample_linear(pos_f, t_in, t_out)
    dof_r = _resample_linear(dof_f, t_in, t_out)
    quat_r = Slerp(t_in, Rotation.from_quat(quat_f))(t_out).as_quat()  # xyzw

    return np.concatenate([pos_r, quat_r, dof_r], axis=1)


def autotune(traj: np.ndarray, fs_in: float, target_jerk: float):
    """Binary-search fc so jerk(filtered dof @ OUTPUT_FPS) ~= target_jerk.

    jerk is monotonically increasing in fc, so a plain bisection works.
    Returns (fc, achieved_jerk, note).
    """
    def jerk_at(fc: float) -> float:
        out = filter_and_resample(traj, fs_in, fc)
        return jerk_mean(out[:, DOF], OUTPUT_FPS)

    j_hi = jerk_at(FC_HI)
    if j_hi <= target_jerk:
        return FC_HI, j_hi, "target already met with minimal filtering (fc=fc_max)"

    j_lo = jerk_at(FC_LO)
    if j_lo >= target_jerk:
        return FC_LO, j_lo, "target not reachable without over-smoothing (fc=fc_min)"

    lo, hi = FC_LO, FC_HI
    fc = 0.5 * (lo + hi)
    j = jerk_at(fc)
    for _ in range(TUNE_ITERS):
        if abs(j - target_jerk) / target_jerk < TUNE_TOL:
            break
        if j < target_jerk:
            lo = fc
        else:
            hi = fc
        fc = 0.5 * (lo + hi)
        j = jerk_at(fc)
    return fc, j, "tuned"


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to these motion names")
    ap.add_argument("--fc", type=float, default=None,
                    help="use a FIXED cutoff (Hz) for every motion instead of auto-tuning")
    args = ap.parse_args()

    os.makedirs(OUT_CSV_DIR, exist_ok=True)
    os.makedirs(OUT_PKL_DIR, exist_ok=True)
    motions = args.only or sorted(NAME_MAP)

    rows = []
    for name in motions:
        mocap_path = os.path.join(MOCAP_DIR, f"{name}.csv")
        genmo_path = os.path.join(GENMO_DIR, f"unitree_g1_{NAME_MAP[name]}.csv")
        if not (os.path.exists(mocap_path) and os.path.exists(genmo_path)):
            print(f"[skip] {name}: missing input")
            continue

        fs_in = INPUT_FPS_OVERRIDE.get(name, DEFAULT_INPUT_FPS)
        traj = load_csv(mocap_path)
        genmo = load_csv(genmo_path)

        target = jerk_mean(genmo[:, DOF], GENMO_FPS)
        j_native = jerk_mean(traj[:, DOF], fs_in)
        j_raw30 = jerk_mean(filter_and_resample(traj, fs_in, FC_HI)[:, DOF], OUTPUT_FPS)

        if args.fc is not None:
            fc = args.fc
            out = filter_and_resample(traj, fs_in, fc)
            j_filt = jerk_mean(out[:, DOF], OUTPUT_FPS)
            note = f"fixed fc={fc:g}"
        else:
            fc, j_filt, note = autotune(traj, fs_in, target)
            out = filter_and_resample(traj, fs_in, fc)

        out_path = os.path.join(OUT_CSV_DIR, f"{name}.csv")
        np.savetxt(out_path, out, delimiter=",", fmt="%.12g")

        # pkl for the GMR evaluation pipeline (root_rot stays xyzw, fps = OUTPUT_FPS)
        with open(os.path.join(OUT_PKL_DIR, f"unitree_g1_{NAME_MAP[name]}.pkl"), "wb") as f:
            pickle.dump({"root_pos": out[:, ROOT_POS], "root_rot": out[:, ROOT_QUAT],
                         "dof_pos": out[:, DOF], "fps": OUTPUT_FPS}, f)

        err = 100.0 * (j_filt - target) / target
        rows.append(dict(motion=name, fs_in=fs_in, n_in=traj.shape[0], n_out=out.shape[0],
                         fc_hz=fc, jerk_mocap_native=j_native, jerk_mocap_30=j_raw30,
                         jerk_genmo_30=target, jerk_filtered_30=j_filt, err_pct=err, note=note))
        print(f"{name:18s} fc={fc:5.2f}Hz  mocap@30={j_raw30:7.1f}  "
              f"genmo={target:7.1f}  filtered={j_filt:7.1f}  ({err:+.1f}%)  {note}")

    _write_reports(rows)


def _write_reports(rows: list[dict]) -> None:
    if not rows:
        return
    csv_path = os.path.join(OUT_DIR, "report.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    m_native = np.mean([r["jerk_mocap_native"] for r in rows])
    m_raw30 = np.mean([r["jerk_mocap_30"] for r in rows])
    m_genmo = np.mean([r["jerk_genmo_30"] for r in rows])
    m_filt = np.mean([r["jerk_filtered_30"] for r in rows])

    md = os.path.join(OUT_DIR, "report.md")
    with open(md, "w") as f:
        f.write("# Low-pass auto-tuning log\n\n")
        f.write("Mean jerk (rad/s^3), joints only. `mocap @30` / `filtered @30` here use this\n"
                "script's own SLERP resample and are ~0.5% off the official pipeline numbers in\n"
                "`metrics_table.md` (linspace resample) — that file is the authoritative table.\n\n")
        f.write("| Motion | fs_in | fc (Hz) | mocap native | mocap @30 | GenMo @30 | filtered @30 | err |\n")
        f.write("|---|--:|--:|--:|--:|--:|--:|--:|\n")
        for r in rows:
            f.write(f"| {r['motion']} | {r['fs_in']:.0f} | {r['fc_hz']:.2f} | "
                    f"{r['jerk_mocap_native']:.0f} | {r['jerk_mocap_30']:.0f} | "
                    f"{r['jerk_genmo_30']:.0f} | {r['jerk_filtered_30']:.0f} | "
                    f"{r['err_pct']:+.1f}% |\n")
        f.write(f"| **MEAN** | | | **{m_native:.0f}** | **{m_raw30:.0f}** | "
                f"**{m_genmo:.0f}** | **{m_filt:.0f}** | "
                f"**{100*(m_filt-m_genmo)/m_genmo:+.1f}%** |\n\n")
        f.write(f"- mocap (raw, resampled to 30 Hz): **{m_raw30:.0f}**\n")
        f.write(f"- GenMo reference (30 Hz): **{m_genmo:.0f}**\n")
        f.write(f"- mocap after low-pass (30 Hz): **{m_filt:.0f}**\n")
    print(f"\nreport -> {md}\n        -> {csv_path}")
    print(f"MEAN  mocap@30={m_raw30:.0f}  genmo={m_genmo:.0f}  filtered={m_filt:.0f}")


if __name__ == "__main__":
    main()
