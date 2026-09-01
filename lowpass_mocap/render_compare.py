#!/usr/bin/env python3
"""
3-panel MP4 at 30 fps, robot facing the camera:
    raw mocap  |  low-pass filtered mocap  |  GenMo (video)

The GenMo clip is a different recording, so panels are NOT time-synced;
each stream plays its own frames and holds on its last frame once finished.

Headless MuJoCo (EGL) offscreen rendering -- no display needed.
Run lowpass_filter.py first (needs output/csv/<motion>.csv).

    MUJOCO_GL=egl python render_compare.py [--only run boxe ...]
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation, Slerp

from general_motion_retargeting import ROBOT_XML_DICT, ROBOT_BASE_DICT, VIEWER_CAM_DISTANCE_DICT
from lowpass_filter import (
    DOF, GENMO_FPS, INPUT_FPS_OVERRIDE, DEFAULT_INPUT_FPS, MOCAP_DIR, GENMO_DIR,
    NAME_MAP, OUT_CSV_DIR, OUT_DIR, OUTPUT_FPS, ROOT_POS, ROOT_QUAT,
    _quat_hemisphere, jerk_mean, load_csv,
)

ROBOT = "unitree_g1"
W, H = 640, 480
VIDEO_DIR = os.path.join(OUT_DIR, "videos")


def yaw_normalize(traj: np.ndarray) -> np.ndarray:
    """Rotate the whole clip about world-Z so frame 0 faces +x (toward the camera)."""
    q = Rotation.from_quat(traj[:, ROOT_QUAT])          # xyzw
    yaw0 = q[0].as_euler("zyx")[0]
    corr = Rotation.from_euler("z", -yaw0)
    out = traj.copy()
    out[:, ROOT_QUAT] = (corr * q).as_quat()
    out[:, ROOT_POS] = corr.apply(traj[:, ROOT_POS])
    return out


def resample_raw(traj: np.ndarray, fs_in: float, fs_out: float = OUTPUT_FPS) -> np.ndarray:
    """Resample (no filtering) to fs_out. Linear for pos/dof, SLERP for quat."""
    n = traj.shape[0]
    t_in = np.arange(n) / fs_in
    m = int(round((n - 1) / fs_in * fs_out)) + 1
    t_out = np.arange(m) / fs_out
    t_out[-1] = min(t_out[-1], t_in[-1])
    pos = np.stack([np.interp(t_out, t_in, traj[:, i]) for i in range(0, 3)], axis=1)
    dof = np.stack([np.interp(t_out, t_in, traj[:, i]) for i in range(7, 36)], axis=1)
    quat = _quat_hemisphere(traj[:, ROOT_QUAT])
    quat_r = Slerp(t_in, Rotation.from_quat(quat))(t_out).as_quat()
    return np.concatenate([pos, quat_r, dof], axis=1)


class Renderer:
    def __init__(self) -> None:
        self.model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[ROBOT]))
        self.data = mj.MjData(self.model)
        self.renderer = mj.Renderer(self.model, H, W)
        self.base_id = self.model.body(ROBOT_BASE_DICT[ROBOT]).id
        self.cam = mj.MjvCamera()
        self.cam.distance = VIEWER_CAM_DISTANCE_DICT[ROBOT]
        self.cam.elevation = -10
        self.cam.azimuth = 180  # robot faces the camera

    def frame(self, root_pos, quat_xyzw, dof) -> np.ndarray:
        self.data.qpos[:3] = root_pos
        self.data.qpos[3:7] = quat_xyzw[[3, 0, 1, 2]]  # xyzw -> wxyz
        self.data.qpos[7:] = dof
        mj.mj_forward(self.model, self.data)
        self.cam.lookat = self.data.xpos[self.base_id]
        self.renderer.update_scene(self.data, camera=self.cam)
        return self.renderer.render()


def _label(img: np.ndarray, text: str) -> np.ndarray:
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 22], fill=(0, 0, 0))
    d.text((8, 5), text, fill=(255, 255, 255))
    return np.asarray(im)


def render_motion(name: str, rnd: Renderer) -> None:
    fs_in = INPUT_FPS_OVERRIDE.get(name, DEFAULT_INPUT_FPS)
    raw = resample_raw(load_csv(os.path.join(MOCAP_DIR, f"{name}.csv")), fs_in)
    filt = load_csv(os.path.join(OUT_CSV_DIR, f"{name}.csv"))
    genmo = load_csv(os.path.join(GENMO_DIR, f"unitree_g1_{NAME_MAP[name]}.csv"))

    j_raw = jerk_mean(raw[:, DOF], OUTPUT_FPS)
    j_filt = jerk_mean(filt[:, DOF], OUTPUT_FPS)
    j_gen = jerk_mean(genmo[:, DOF], GENMO_FPS)

    panels = [
        (yaw_normalize(raw), f"mocap @30  jerk={j_raw:.0f}"),
        (yaw_normalize(filt), f"mocap low-pass @30  jerk={j_filt:.0f}"),
        (yaw_normalize(genmo), f"GenMo @30  jerk={j_gen:.0f}"),
    ]
    n = max(len(p) for p, _ in panels)

    os.makedirs(VIDEO_DIR, exist_ok=True)
    path = os.path.join(VIDEO_DIR, f"{name}.mp4")
    with imageio.get_writer(path, fps=OUTPUT_FPS, macro_block_size=1) as w:
        for i in range(n):
            cols = []
            for traj, txt in panels:
                k = min(i, len(traj) - 1)
                cols.append(_label(
                    rnd.frame(traj[k, ROOT_POS], traj[k, ROOT_QUAT], traj[k, DOF]), txt))
            w.append_data(np.concatenate(cols, axis=1))
    print(f"{name:18s} -> {path}   ({n} frames)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    rnd = Renderer()
    for name in (args.only or sorted(NAME_MAP)):
        if not os.path.exists(os.path.join(OUT_CSV_DIR, f"{name}.csv")):
            print(f"[skip] {name}: run lowpass_filter.py first")
            continue
        render_motion(name, rnd)


if __name__ == "__main__":
    main()
