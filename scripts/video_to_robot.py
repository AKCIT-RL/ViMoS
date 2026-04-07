#!/usr/bin/env python3
"""
CopyCat — Full pipeline: Video → SMPL (GENMO) → Robot Motion (GMR)

Minimal usage:
    python run_pipeline.py --video /path/to/video.mp4 --robot booster_t1

Sandwich mode (default): generates a "stand still" prefix + core (video) + "stand still" suffix
before passing to GMR, ensuring a stable transition for the T1.

Process an entire folder of videos:
    python run_pipeline.py --video /path/to/folder/ --robot unitree_g1
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths relative to CopyCat root
# ---------------------------------------------------------------------------
COPYCAT_DIR = Path(__file__).resolve().parent.parent
GENMO_DIR   = COPYCAT_DIR / "retarget" / "GENMO"
GMR_DIR     = COPYCAT_DIR / "retarget" / "GMR"

GENMO_SCRIPT      = GENMO_DIR / "scripts" / "demo" / "demo_text.py"
GMR_SCRIPT        = GMR_DIR / "scripts" / "gvhmr_to_robot.py"
GMR_PKL_TO_CSV    = GMR_DIR / "scripts" / "batch_gmr_pkl_to_csv.py"

# GMR output folders
GMR_OUTPUT_PKL = GMR_DIR / "output" / "pkl"
GMR_OUTPUT_CSV = GMR_DIR / "output" / "csv"
GMR_VIDEOS_DIR = GMR_DIR / "videos"

# Default GENMO checkpoint
DEFAULT_CKPT = GENMO_DIR / "inputs" / "checkpoints" / "s050000.ckpt"

# Anchor (sandwich) parameters
ANCHOR_FPS    = 30          # fixed FPS — IsaacLab / T1 controller compatibility
ANCHOR_FRAMES = 30          # prefix: 1 s × 30 fps
SUFFIX_FRAMES = 60          # suffix:  2 s × 30 fps

# Neutral body pose used in stand still anchors.
# SMPL-X body_pose: 21 joints × 3 (axis-angle) = 63 values.
# Joint map (body_pose, excluding root/pelvis):
#   0  left_hip      [0:3]    |  11 neck          [33:36]
#   1  right_hip     [3:6]    |  12 left_collar   [36:39]
#   2  spine1        [6:9]    |  13 right_collar  [39:42]
#   3  left_knee     [9:12]   |  14 head          [42:45]
#   4  right_knee    [12:15]  |  15 left_shoulder [45:48]
#   5  spine2        [15:18]  |  16 right_shoulder[48:51]
#   6  left_ankle    [18:21]  |  17 left_elbow    [51:54]
#   7  right_ankle   [21:24]  |  18 right_elbow   [54:57]
#   8  spine3        [24:27]  |  19 left_wrist    [57:60]
#   9  left_foot     [27:30]  |  20 right_wrist   [60:63]
#  10  right_foot    [30:33]
NEUTRAL_BODY_POSE_DIM = 63


def _extract_yaw_orient(global_orient_aa):
    """
    Takes a global_orient axis-angle (3,) and returns another axis-angle
    with only the yaw component (rotation around Y axis) preserved.
    Pitch and roll are zeroed → robot upright facing the same direction as in the video.
    """
    import torch
    aa = global_orient_aa.float()
    angle = aa.norm()
    if angle < 1e-6:
        return aa.clone()

    # Axis-angle → rotation matrix (Rodrigues)
    axis = aa / angle
    K = torch.zeros(3, 3)
    K[0, 1], K[0, 2] = -axis[2],  axis[1]
    K[1, 0], K[1, 2] =  axis[2], -axis[0]
    K[2, 0], K[2, 1] = -axis[1],  axis[0]
    R = torch.eye(3) + angle.sin() * K + (1 - angle.cos()) * (K @ K)

    # Extract yaw: project the "forward" vector (column Z of R) onto the XZ plane
    fwd = R[:, 2]
    yaw = torch.atan2(fwd[0], fwd[2])

    # Build pure Y rotation matrix (yaw only)
    c, s = yaw.cos(), yaw.sin()
    Ry = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    # Rotation matrix → axis-angle
    theta = ((Ry.trace() - 1) / 2).clamp(-1, 1).acos()
    if theta.abs() < 1e-6:
        return torch.zeros(3)
    ax = torch.stack([Ry[2,1]-Ry[1,2], Ry[0,2]-Ry[2,0], Ry[1,0]-Ry[0,1]]) / (2 * theta.sin())
    return ax * theta


def _make_standing_body_pose():
    """
    Creates a (63,) tensor with an upright pose and arms along the body.
    Adjust the values below with --neutral_pose_path if needed.
    """
    import torch as _t
    bp = _t.zeros(NEUTRAL_BODY_POSE_DIM)
    # Collars: slight inward rotation to position shoulders naturally
    bp[36:39] = _t.tensor([0.0,  0.0, -0.4])   # left_collar
    bp[39:42] = _t.tensor([0.0,  0.0,  0.4])   # right_collar
    # Shoulders: ~46° to bring arms from horizontal to alongside the body
    bp[45:48] = _t.tensor([0.0,  0.0, -0.8])   # left_shoulder
    bp[48:51] = _t.tensor([0.0,  0.0,  0.8])   # right_shoulder
    return bp


# ---------------------------------------------------------------------------
# Python executable for each tool
# ---------------------------------------------------------------------------
def _find_python(venv_dirs: list[Path]) -> str:
    """Returns the first python found among the candidate venvs."""
    for venv in venv_dirs:
        candidate = venv / "bin" / "python"
        if candidate.exists():
            return str(candidate)
    return sys.executable

GENMO_PYTHON = _find_python([
    GENMO_DIR / ".venv",
    COPYCAT_DIR.parent / "GENMO" / ".venv",
])
GMR_PYTHON = _find_python([
    GMR_DIR / ".venv",
    COPYCAT_DIR / ".venv",
    COPYCAT_DIR.parent / "GMR" / ".venv",
])

SUPPORTED_ROBOTS = [
    "unitree_g1", "unitree_g1_with_hands", "unitree_h1", "unitree_h1_2",
    "booster_t1", "booster_t1_29dof", "stanford_toddy", "fourier_n1",
    "engineai_pm01", "kuavo_s45", "hightorque_hi", "galaxea_r1pro",
    "berkeley_humanoid_lite", "booster_k1", "pnd_adam_lite", "openloong",
    "tienkung",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_videos_in_folder(folder: Path) -> list[Path]:
    videos = sorted(folder.rglob("*.mp4"))
    if not videos:
        print(f"[WARN] No .mp4 files found in: {folder}")
    return videos


def _run_genmo_cmd(cmd: list, label: str, output_pt: Path):
    """Runs a GENMO command and validates that hmr4d_results.pt was generated."""
    print("\n" + "=" * 60)
    print(f"[GENMO] {label}")
    print("=" * 60)
    print("Command:", " ".join(cmd))

    result = subprocess.run(cmd, cwd=str(GENMO_DIR))
    if result.returncode != 0:
        print(f"[ERROR] GENMO failed: {label}")
        sys.exit(result.returncode)

    if not output_pt.exists():
        print(f"[ERROR] hmr4d_results.pt not found at: {output_pt}")
        sys.exit(1)

    return output_pt


def run_genmo(video_path: Path, video_name: str, output_dir: Path, args) -> Path:
    """
    Runs GENMO in video mode and returns the path to hmr4d_results.pt.
    """
    hmr4d_results = output_dir / "hmr4d_results.pt"

    if hmr4d_results.exists() and not args.force:
        print(f"[GENMO] Skipping video — result already exists: {hmr4d_results}")
        return hmr4d_results

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        GENMO_PYTHON,
        str(GENMO_SCRIPT),
        f"video1_path={video_path}",
        f"video1_name={video_name}",
        f"output_dir={output_dir}",
        f"exp={args.exp}",
        f"ckpt_path={args.ckpt_path}",
        "rsync_ckpt=false",
        f"static_cam1={'true' if args.static_cam else 'false'}",
        f"orig_fps1={args.orig_fps}",
    ]
    if args.pose:
        cmd.append("--pose")

    return _run_genmo_cmd(cmd, f"Processing video: {video_path.name}", hmr4d_results)


def smooth_transitions(
    combined_pt: Path,
    t1: int,
    t2: int,
    n_frames: int,
    neutral_body_pose,  # torch.Tensor (63,) — zeros or custom pose
) -> None:
    """
    Builds stand still anchors before and after the motion core.

    PREFIX [0 : t1]
      · transl        = core[0].transl        — robot at initial position
      · global_orient = core[0].global_orient — same orientation as start
      · body_pose     = neutral (zeros/custom), lerp → core[0] over last n_frames
      · betas         = core[0].betas

    SUFFIX [t2 : end]
      · transl        = core[-1].transl        — no teleport
      · global_orient = core[-1].global_orient — same orientation as end
      · body_pose     = lerp core[-1] → neutral over first n_frames, then neutral
      · betas         = core[-1].betas
    """
    import torch

    pred = torch.load(combined_pt, map_location="cpu")
    nbp  = neutral_body_pose  # short alias

    for section in ("smpl_params_global", "smpl_params_incam"):
        params = pred[section]

        for key in params:
            x        = params[key].float()
            T        = x.shape[0]
            n_suffix = T - t2
            c0       = x[t1].clone()       # first frame of core
            cN       = x[t2 - 1].clone()   # last  frame of core

            # Neutral value for this field
            if key == "body_pose":
                neutral_pre = nbp.to(x.dtype)
                neutral_suf = nbp.to(x.dtype)
            elif key == "global_orient":
                # Core yaw preserved; pitch/roll zeroed → upright facing correct direction
                neutral_pre = _extract_yaw_orient(c0).to(x.dtype)
                neutral_suf = _extract_yaw_orient(cN).to(x.dtype)
            else:
                neutral_pre = neutral_suf = None  # transl, betas

            # ── PREFIX ─────────────────────────────────────────────────────
            if neutral_pre is not None:
                x[:t1] = neutral_pre.unsqueeze(0).expand(t1, *x.shape[1:])
                lerp_start = max(0, t1 - n_frames)
                n_lerp = t1 - lerp_start
                for i in range(n_lerp):
                    w = (i + 1) / (n_lerp + 1)
                    x[lerp_start + i] = neutral_pre * (1.0 - w) + c0 * w
            else:
                x[:t1] = c0.unsqueeze(0).expand(t1, *x.shape[1:])

            # ── SUFFIX ─────────────────────────────────────────────────────
            if neutral_suf is not None:
                for i in range(n_suffix):
                    w = min(1.0, (i + 1) / (n_frames + 1))
                    x[t2 + i] = cN * (1.0 - w) + neutral_suf * w
            else:
                x[t2:] = cN.unsqueeze(0).expand(n_suffix, *x.shape[1:])

            params[key] = x
        pred[section] = params

    torch.save(pred, combined_pt)
    print(
        f"[Smooth] Prefix: orient/transl=core[0], neutral body_pose + lerp {n_frames}f | "
        f"Suffix: orient/transl=core[-1], lerp → neutral {n_frames}f"
    )


def process_video_sandwich(video_path: Path, video_name: str, output_dir: Path, args) -> Path:
    """
    Runs the sandwich pipeline:
        Prefix (stand still) + Core (video) + Suffix (stand still)

    Anchor frames are built by repeating the first/last frame of the core.
    smooth_transitions() overwrites those frames with the neutral pose
    (_make_standing_body_pose) and applies lerp at the edges.

    Returns the path to the combined hmr4d_results.pt.
    """
    import torch as _torch

    prefix_frames = args.anchor_frames
    suffix_frames = args.suffix_frames
    core_dir      = output_dir / "sandwich_core"
    combined_pt   = output_dir / "hmr4d_results.pt"

    print(f"\n{'#' * 60}")
    print(f"# [Sandwich] Stand still prefix : {prefix_frames} frames @ {ANCHOR_FPS} fps ({prefix_frames / ANCHOR_FPS:.1f} s)")
    print(f"# [Sandwich] Core               : {video_path.name}")
    print(f"# [Sandwich] Stand still suffix : {suffix_frames} frames @ {ANCHOR_FPS} fps ({suffix_frames / ANCHOR_FPS:.1f} s)")
    print(f"{'#' * 60}")

    if combined_pt.exists() and not args.force:
        print(f"[Sandwich] Result already exists: {combined_pt}")
        return combined_pt

    # Stage 1 — Core (video → SMPL)
    core_pt = run_genmo(video_path, video_name, core_dir, args)
    core    = _torch.load(core_pt, map_location="cpu")
    core_frames = core["smpl_params_global"]["transl"].shape[0]

    # Stage 2 — Build combined by repeating core edges as anchor placeholders.
    # smooth_transitions() will overwrite these frames with neutral pose + lerp.
    combined = {}
    for section in ("smpl_params_global", "smpl_params_incam"):
        combined[section] = {}
        for key, c in core[section].items():
            prefix = c[0:1].expand(prefix_frames, *c.shape[1:]).clone()
            suffix = c[-1:].expand(suffix_frames, *c.shape[1:]).clone()
            combined[section][key] = _torch.cat([prefix, c, suffix], dim=0)

    K = core.get("K_fullimg")
    if K is not None and K.ndim == 3 and K.shape[1:] == (3, 3):
        combined["K_fullimg"] = _torch.cat([
            K[0:1].expand(prefix_frames, 3, 3).clone(),
            K,
            K[-1:].expand(suffix_frames, 3, 3).clone(),
        ], dim=0)
    else:
        T_total = prefix_frames + core_frames + suffix_frames
        combined["K_fullimg"] = _torch.eye(3).unsqueeze(0).expand(T_total, 3, 3).clone()

    combined_pt.parent.mkdir(parents=True, exist_ok=True)
    _torch.save(combined, combined_pt)
    print(f"[Sandwich] SMPL assembled: {prefix_frames + core_frames + suffix_frames} frames → {combined_pt}")

    # Stage 3 — Neutral pose and transition smoothing
    if args.neutral_pose_path:
        import numpy as _np
        _p = Path(args.neutral_pose_path)
        neutral_body_pose = (
            _torch.from_numpy(_np.load(_p)).float()
            if _p.suffix == ".npy"
            else _torch.load(_p, map_location="cpu").float()
        )
        neutral_body_pose = neutral_body_pose.reshape(NEUTRAL_BODY_POSE_DIM)
    else:
        neutral_body_pose = _make_standing_body_pose()

    smooth_transitions(
        combined_pt,
        t1=prefix_frames,
        t2=prefix_frames + core_frames,
        n_frames=args.transition_frames,
        neutral_body_pose=neutral_body_pose,
    )

    return combined_pt


def run_gmr(hmr4d_results: Path, video_name: str, args):
    """
    Runs GMR (gvhmr_to_robot.py) from hmr4d_results.pt.
    PKL is saved to GMR/output/pkl/<robot>_<video_name>.pkl.
    """
    if args.save_path:
        save_path = Path(args.save_path)
    else:
        GMR_OUTPUT_PKL.mkdir(parents=True, exist_ok=True)
        save_path = GMR_OUTPUT_PKL / f"{args.robot}_{video_name}.pkl"

    GMR_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    video_save_path = GMR_VIDEOS_DIR / f"{args.robot}_{video_name}.mp4"

    cmd = [
        GMR_PYTHON,
        str(GMR_SCRIPT),
        f"--gvhmr_pred_file={hmr4d_results}",
        f"--robot={args.robot}",
        f"--save_path={save_path}",
        f"--video_save_path={video_save_path}",
    ]

    if args.record_video:
        cmd.append("--record_video")
    if args.rate_limit:
        cmd.append("--rate_limit")
    if args.loop:
        cmd.append("--loop")

    env = os.environ.copy()
    if args.headless:
        env["MUJOCO_GL"] = "egl"
        xvfb = shutil.which("xvfb-run")
        if xvfb:
            cmd = [xvfb, "-a", "--server-args=-screen 0 1024x768x24"] + cmd
        else:
            print("[WARN] xvfb-run not found. Install with: sudo apt-get install -y xvfb")

    print("\n" + "=" * 60)
    print(f"[GMR] Retargeting for {args.robot}: {video_name}")
    if args.headless:
        print("[GMR] Headless (xvfb-run + MUJOCO_GL=egl)" if shutil.which("xvfb-run") else "[GMR] Partial headless (MUJOCO_GL=egl only)")
    if args.record_video:
        print(f"[GMR] Video → {video_save_path}")
    print("=" * 60)

    result = subprocess.run(cmd, cwd=str(GMR_DIR), env=env)
    if result.returncode != 0:
        print(f"[ERROR] GMR failed for: {video_name}")
        sys.exit(result.returncode)

    print(f"[GMR] PKL saved to: {save_path}")
    return save_path


def run_pkl_to_csv(pkl_path: Path, video_name: str, robot: str):
    """
    Converts a single GMR .pkl file to CSV and saves it to GMR/output/csv/.
    """
    GMR_OUTPUT_CSV.mkdir(parents=True, exist_ok=True)
    csv_path = GMR_OUTPUT_CSV / f"{robot}_{video_name}.csv"

    print("\n" + "=" * 60)
    print(f"[CSV] Converting PKL → CSV: {pkl_path.name}")
    print(f"[CSV] Destination: {csv_path}")
    print("=" * 60)

    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_pkl = Path(tmp_dir) / pkl_path.name
        shutil.copy2(pkl_path, tmp_pkl)

        cmd = [
            GMR_PYTHON,
            str(GMR_PKL_TO_CSV),
            f"--folder={tmp_dir}",
        ]
        result = subprocess.run(cmd, cwd=str(GMR_DIR))
        if result.returncode != 0:
            print(f"[ERROR] PKL→CSV conversion failed for: {pkl_path.name}")
            return None

        generated_csv = Path(tmp_dir) / "csv" / pkl_path.with_suffix(".csv").name
        if generated_csv.exists():
            shutil.move(str(generated_csv), str(csv_path))
            print(f"[CSV] Saved to: {csv_path}")
        else:
            print(f"[WARN] Generated CSV not found: {generated_csv}")
            return None

    return csv_path


def process_video(video_path: Path, args, video_name: str = None):
    """Runs the full pipeline (GENMO + GMR) for a single video."""
    video_name = video_name or video_path.stem

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = GENMO_DIR / "outputs" / "demo" / video_name

    print(f"\n{'#' * 60}")
    print(f"# Video    : {video_path.name}")
    if video_name != video_path.stem:
        print(f"# Name     : {video_name}")
    print(f"# Output   : {output_dir}")
    print(f"# Robot    : {args.robot}")
    print(f"# Sandwich : {'active' if args.sandwich else 'disabled'}")
    print(f"{'#' * 60}")

    # Stage 1 — GENMO: video → SMPL (with or without sandwich)
    if args.sandwich:
        hmr4d_results = process_video_sandwich(video_path, video_name, output_dir, args)
    else:
        hmr4d_results = run_genmo(video_path, video_name, output_dir, args)

    # Stage 2 — GMR: SMPL → robot motion PKL
    save_path = run_gmr(hmr4d_results, video_name, args)

    # Stage 3 — Convert PKL → CSV
    run_pkl_to_csv(save_path, video_name, args.robot)

    return hmr4d_results, save_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CopyCat — Full pipeline: Video → SMPL (GENMO) → Robot (GMR)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Input ────────────────────────────────────────────────────────────────
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--video", "-v",
        help="Single .mp4 file to process.",
    )
    input_group.add_argument(
        "--videos_path",
        help="Folder with .mp4 videos (recursive search in subfolders). "
             "Ex: --videos_path fut_do_t1/",
    )

    # ── GENMO ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--video_name",
        default=None,
        help="Name used in the output folder (default: video file stem). "
             "Ignored when --video is a folder.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output folder for GENMO results "
             "(default: GENMO/outputs/demo/<video_name>).",
    )
    parser.add_argument(
        "--ckpt_path",
        default=str(DEFAULT_CKPT),
        help=f"GENMO model checkpoint (default: {DEFAULT_CKPT}).",
    )
    parser.add_argument(
        "--exp",
        default="genmo_lg",
        help="GENMO experiment config (default: genmo_lg).",
    )
    parser.add_argument(
        "--orig_fps",
        type=int,
        default=30,
        help="Original FPS of the input video (default: 30).",
    )
    parser.add_argument(
        "--static_cam",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Assume static camera (default: on). Use --no-static-cam for SLAM.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force reprocessing even if hmr4d_results.pt already exists.",
    )
    parser.add_argument(
        "--pose",
        action="store_true",
        default=False,
        help="Generate pose/YOLO debug PNGs in GENMO (off by default).",
    )

    # ── Sandwich (Mixed Conditions) ───────────────────────────────────────────
    parser.add_argument(
        "--sandwich",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable sandwich mode: 'stand still' prefix + video + 'stand still' suffix. "
             "Use for videos that start/end abruptly in poses the robot cannot safely hold. "
             "Disabled by default. Enable with --sandwich.",
    )
    parser.add_argument(
        "--anchor_frames",
        type=int,
        default=ANCHOR_FRAMES,
        help=f"Stand still frames in the prefix (default: {ANCHOR_FRAMES} = {ANCHOR_FRAMES / ANCHOR_FPS:.1f} s @ {ANCHOR_FPS} fps).",
    )
    parser.add_argument(
        "--suffix_frames",
        type=int,
        default=SUFFIX_FRAMES,
        help=f"Stand still frames in the suffix (default: {SUFFIX_FRAMES} = {SUFFIX_FRAMES / ANCHOR_FPS:.1f} s @ {ANCHOR_FPS} fps).",
    )
    parser.add_argument(
        "--neutral_pose_path",
        default=None,
        help="A .npy or .pt file with a (63,) body_pose tensor for the anchors "
             "(default: zeros = SMPL rest pose). Useful for adjusting arm position.",
    )
    parser.add_argument(
        "--transition_frames",
        type=int,
        default=15,
        help="Smoothing frames at each sandwich transition edge "
             "(default: 15 = 0.5 s at 30 fps). Uses lerp on SMPL parameters.",
    )

    # ── GMR ──────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--robot", "-r",
        choices=SUPPORTED_ROBOTS,
        default="unitree_g1",
        help="Target robot for retargeting (default: unitree_g1).",
    )
    parser.add_argument(
        "--save_path",
        default=None,
        help="Path to save the robot motion (.pkl). "
             "Default: GMR/output/pkl/<robot>_<video_name>.pkl",
    )
    parser.add_argument(
        "--record_video",
        action="store_true",
        default=False,
        help="Record a video of the GMR visualization.",
    )
    parser.add_argument(
        "--rate_limit",
        action="store_true",
        default=False,
        help="Limit playback rate to the human motion FPS.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        default=False,
        help="Loop the motion in the viewer indefinitely.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run GMR/MuJoCo in headless mode (no window). "
             "Sets MUJOCO_GL=egl. Useful for SSH or display-less servers.",
    )

    args = parser.parse_args()

    # ── Basic validation ──────────────────────────────────────────────────────
    if not GENMO_SCRIPT.exists():
        print(f"[ERROR] GENMO script not found: {GENMO_SCRIPT}")
        sys.exit(1)
    if not GMR_SCRIPT.exists():
        print(f"[ERROR] GMR script not found: {GMR_SCRIPT}")
        sys.exit(1)

    print(f"[Config] GENMO Python : {GENMO_PYTHON}")
    print(f"[Config] GMR   Python : {GMR_PYTHON}")
    if args.sandwich:
        print(f"[Config] Sandwich     : active")
        print(f"[Config] Prefix       : {args.anchor_frames} frames ({args.anchor_frames / ANCHOR_FPS:.1f} s @ {ANCHOR_FPS} fps)")
        print(f"[Config] Suffix       : {args.suffix_frames} frames ({args.suffix_frames / ANCHOR_FPS:.1f} s @ {ANCHOR_FPS} fps)")

    # ── Folder mode (--videos_path) ───────────────────────────────────────────
    if args.videos_path:
        folder = Path(args.videos_path).resolve()
        if not folder.is_dir():
            print(f"[ERROR] Folder not found: {folder}")
            sys.exit(1)
        videos = find_videos_in_folder(folder)
        if not videos:
            sys.exit(1)
        print(f"[INFO] {len(videos)} video(s) found in: {folder}")

        results = []
        for i, vp in enumerate(videos, 1):
            print(f"\n[{i}/{len(videos)}] {vp.name}")
            try:
                _, save = process_video(vp, args)
                results.append((vp, save, None))
            except SystemExit as e:
                print(f"[WARN] Skipping {vp.name} (error {e.code})")
                results.append((vp, None, e.code))

        print("\n" + "=" * 60)
        print(f"[DONE] {len(videos)} video(s) processed.")
        for vp, save, err in results:
            print(f"  {vp.name} → {str(save) if save else f'ERROR ({err})'}")
        return

    # ── Single file mode (--video) ────────────────────────────────────────────
    video_input = Path(args.video).resolve()
    if not video_input.exists():
        print(f"[ERROR] Path not found: {video_input}")
        sys.exit(1)
    if video_input.suffix.lower() != ".mp4":
        print(f"[WARN] File is not .mp4: {video_input}")

    process_video(video_input, args, video_name=args.video_name)

    print("\n[DONE] Pipeline finished.")


if __name__ == "__main__":
    main()
