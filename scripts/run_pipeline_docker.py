#!/usr/bin/env python3
"""
CopyCat — Full pipeline via Docker: Video → SMPL (GENMO) → Robot Motion (GMR)

Identical interface to run_pipeline_refpose.py, but runs GENMO and GMR
in separate containers via docker compose run.

Prerequisites:
  1. Docker with nvidia-container-toolkit installed
  2. Images built:
       docker compose build
  3. Data in place:
       GENMO/inputs/checkpoints/s050000.ckpt
       GMR/assets/booster_t1/  (or the desired robot)
       GMR/assets/body_models/smplx/

Minimal usage:
    python scripts/run_pipeline_docker.py --video /path/to/video.mp4 --robot booster_t1

Process an entire folder:
    python scripts/run_pipeline_docker.py --videos_path fut_do_t1/ --robot unitree_g1
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Host paths
# ─────────────────────────────────────────────────────────────────────────────
COPYCAT_DIR = Path(__file__).resolve().parent
GENMO_DIR   = COPYCAT_DIR / "GENMO"
GMR_DIR     = COPYCAT_DIR / "GMR"

# Host mount points (mirrors of volumes in docker-compose.yml)
GENMO_OUTPUTS_HOST = GENMO_DIR / "outputs"
GENMO_INPUTS_HOST  = GENMO_DIR / "inputs"
GMR_OUTPUT_HOST    = GMR_DIR / "output"
GMR_VIDEOS_HOST    = GMR_DIR / "videos"
GMR_ASSETS_HOST    = GMR_DIR / "assets"

# Container mount points (must match docker-compose.yml)
GENMO_OUTPUTS_CONT = "/app/outputs"
GENMO_INPUTS_CONT  = "/app/inputs"
GMR_OUTPUT_CONT    = "/app/output"
GMR_VIDEOS_CONT    = "/app/videos"
GMR_ASSETS_CONT    = "/app/assets"
GMR_GENMO_OUT_CONT = "/genmo_outputs"   # GENMO outputs read by GMR

# Scripts inside containers
GENMO_SCRIPT_CONT      = "scripts/demo/demo_text.py"
GENMO_SANDWICH_CONT    = "scripts/sandwich_runner.py"
GMR_SCRIPT_CONT        = "scripts/gvhmr_to_robot_docker.py"
GMR_PKL_TO_CSV_CONT    = "scripts/batch_gmr_pkl_to_csv.py"

DEFAULT_CKPT = GENMO_INPUTS_HOST / "checkpoints" / "s050000.ckpt"

ANCHOR_FPS    = 30
ANCHOR_FRAMES = 30
SUFFIX_FRAMES = 60

SUPPORTED_ROBOTS = [
    "unitree_g1", "unitree_g1_with_hands", "unitree_h1", "unitree_h1_2",
    "booster_t1", "booster_t1_29dof", "stanford_toddy", "fourier_n1",
    "engineai_pm01", "kuavo_s45", "hightorque_hi", "galaxea_r1pro",
    "berkeley_humanoid_lite", "booster_k1", "pnd_adam_lite", "openloong",
    "tienkung",
]


# ─────────────────────────────────────────────────────────────────────────────
# Host → container path mapping
# ─────────────────────────────────────────────────────────────────────────────

def _host_to_container(host_path: Path, mount_pairs: list[tuple[Path, str]]) -> tuple[str, str | None]:
    """
    Attempts to map host_path using the provided (host_base, container_base) pairs.
    Returns (container_path, extra_volume_spec_or_None).

    If the path is not under any already-mounted volume, creates an extra volume
    by mounting its parent directory at /extra_mount_N.
    """
    host_path = host_path.resolve()
    for host_base, cont_base in mount_pairs:
        try:
            rel = host_path.relative_to(host_base.resolve())
            return f"{cont_base}/{rel}".replace("\\", "/"), None
        except ValueError:
            continue

    # Path outside known volumes — mount parent directory dynamically
    parent = host_path.parent
    cont_point = f"/extra_vol_{abs(hash(str(parent))) % 100000}"
    extra_vol = f"{parent}:{cont_point}:ro"
    return f"{cont_point}/{host_path.name}", extra_vol


def _genmo_path(host_path: Path) -> tuple[str, str | None]:
    """Maps host_path to the GENMO container."""
    return _host_to_container(host_path, [
        (GENMO_OUTPUTS_HOST, GENMO_OUTPUTS_CONT),
        (GENMO_INPUTS_HOST,  GENMO_INPUTS_CONT),
    ])


def _gmr_path(host_path: Path) -> tuple[str, str | None]:
    """Maps host_path to the GMR container."""
    return _host_to_container(host_path, [
        (GENMO_OUTPUTS_HOST, GMR_GENMO_OUT_CONT),
        (GMR_OUTPUT_HOST,    GMR_OUTPUT_CONT),
        (GMR_VIDEOS_HOST,    GMR_VIDEOS_CONT),
        (GMR_ASSETS_HOST,    GMR_ASSETS_CONT),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# docker compose run executor
# ─────────────────────────────────────────────────────────────────────────────

def _docker_compose_run(service: str, command: list[str], extra_volumes: list[str] | None = None) -> int:
    """
    Runs a command in a docker compose service and waits for completion.
    Uses `docker compose run --rm --no-deps`.

    extra_volumes: list of "host_path:container_path[:ro]" strings
    """
    # Run with host UID/GID so files created inside the container
    # are owned by the correct user (avoids PermissionError on bind mounts).
    uid, gid = os.getuid(), os.getgid()

    cmd = [
        "docker", "compose",
        "--file", str(COPYCAT_DIR / "docker-compose.yml"),
        "run", "--rm", "--no-deps",
        "--user", f"{uid}:{gid}",
    ]

    for vol in (extra_volumes or []):
        cmd.extend(["-v", vol])

    cmd.append(service)
    cmd.extend(command)

    print("\n" + "=" * 60)
    print(f"[Docker] Service: {service}")
    print(f"[Docker] Command: {' '.join(cmd)}")
    print("=" * 60)

    result = subprocess.run(cmd, cwd=str(COPYCAT_DIR))
    return result.returncode


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline stages
# ─────────────────────────────────────────────────────────────────────────────

def find_videos_in_folder(folder: Path) -> list[Path]:
    videos = sorted(folder.rglob("*.mp4"))
    if not videos:
        print(f"[WARNING] No .mp4 files found in: {folder}")
    return videos


def run_genmo(video_path: Path, video_name: str, output_dir: Path, args) -> Path:
    """Runs GENMO in the container and returns the path to hmr4d_results.pt on the host."""
    hmr4d_results = output_dir / "hmr4d_results.pt"

    if hmr4d_results.exists() and not args.force:
        print(f"[GENMO] Skipping — result already exists: {hmr4d_results}")
        return hmr4d_results

    output_dir.mkdir(parents=True, exist_ok=True)

    # Video: mount parent directory as /input (read-only)
    video_host = video_path.resolve()
    container_video = f"/input/{video_host.name}"
    video_volume = f"{video_host.parent}:/input:ro"

    # Container paths
    container_output, extra_out = _genmo_path(output_dir)
    container_ckpt,   extra_ckpt = _genmo_path(Path(args.ckpt_path))

    extra_volumes = [video_volume]
    if extra_out:
        extra_volumes.append(extra_out)
    if extra_ckpt:
        extra_volumes.append(extra_ckpt)

    command = [
        "python", GENMO_SCRIPT_CONT,
        f"video1_path={container_video}",
        f"video1_name={video_name}",
        f"output_dir={container_output}",
        f"exp={args.exp}",
        f"ckpt_path={container_ckpt}",
        "rsync_ckpt=false",
        f"static_cam1={'true' if args.static_cam else 'false'}",
        f"orig_fps1={args.orig_fps}",
    ]
    if args.pose:
        command.append("--pose")

    rc = _docker_compose_run("genmo", command, extra_volumes)
    if rc != 0:
        print(f"[ERROR] GENMO failed (exit code {rc})")
        sys.exit(rc)

    if not hmr4d_results.exists():
        print(f"[ERROR] hmr4d_results.pt not found: {hmr4d_results}")
        sys.exit(1)

    return hmr4d_results


def run_sandwich(core_pt: Path, output_pt: Path, args) -> None:
    """
    Runs sandwich_runner.py inside the GENMO container (which has torch).
    Combines prefix + core + suffix and applies lerp at the boundaries.
    """
    container_core,   extra_core   = _genmo_path(core_pt)
    container_output, extra_output = _genmo_path(output_pt)

    extra_volumes = []
    if extra_core:
        extra_volumes.append(extra_core)
    if extra_output:
        extra_volumes.append(extra_output)

    command = [
        "python", GENMO_SANDWICH_CONT,
        f"--core_pt={container_core}",
        f"--output_pt={container_output}",
        f"--prefix_frames={args.anchor_frames}",
        f"--suffix_frames={args.suffix_frames}",
        f"--transition_frames={args.transition_frames}",
    ]

    if args.neutral_pose_path:
        neutral_path = Path(args.neutral_pose_path).resolve()
        container_neutral, extra_neutral = _genmo_path(neutral_path)
        if extra_neutral:
            extra_volumes.append(extra_neutral)
        command.append(f"--neutral_pose_path={container_neutral}")

    rc = _docker_compose_run("genmo", command, extra_volumes or None)
    if rc != 0:
        print(f"[ERROR] sandwich_runner failed (exit code {rc})")
        sys.exit(rc)


def process_video_sandwich(video_path: Path, video_name: str, output_dir: Path, args) -> Path:
    """Sandwich pipeline: Prefix + Core + Suffix → combined hmr4d_results.pt."""
    prefix_frames = args.anchor_frames
    suffix_frames = args.suffix_frames
    core_dir    = output_dir / "sandwich_core"
    combined_pt = output_dir / "hmr4d_results.pt"

    print(f"\n{'#' * 60}")
    print(f"# [Sandwich] Prefix : {prefix_frames} frames @ {ANCHOR_FPS} fps ({prefix_frames / ANCHOR_FPS:.1f} s)")
    print(f"# [Sandwich] Core   : {video_path.name}")
    print(f"# [Sandwich] Suffix : {suffix_frames} frames @ {ANCHOR_FPS} fps ({suffix_frames / ANCHOR_FPS:.1f} s)")
    print(f"{'#' * 60}")

    if combined_pt.exists() and not args.force:
        print(f"[Sandwich] Result already exists: {combined_pt}")
        return combined_pt

    # Stage 1: GENMO on core
    core_pt = run_genmo(video_path, video_name, core_dir, args)

    # Stage 2: Sandwich + smooth_transitions inside GENMO container
    run_sandwich(core_pt, combined_pt, args)

    if not combined_pt.exists():
        print(f"[ERROR] Combined tensor not found: {combined_pt}")
        sys.exit(1)

    return combined_pt


def run_gmr(hmr4d_results: Path, video_name: str, args) -> Path:
    """Runs GMR in the container and returns the path to the PKL on the host."""
    if args.save_path:
        save_path = Path(args.save_path).resolve()
    else:
        (GMR_OUTPUT_HOST / "pkl").mkdir(parents=True, exist_ok=True)
        save_path = GMR_OUTPUT_HOST / "pkl" / f"{args.robot}_{video_name}.pkl"

    GMR_VIDEOS_HOST.mkdir(parents=True, exist_ok=True)
    video_save_path = GMR_VIDEOS_HOST / f"{args.robot}_{video_name}.mp4"

    container_hmr4d,     extra_hmr4d     = _gmr_path(hmr4d_results)
    container_save,      extra_save      = _gmr_path(save_path)
    container_video_out, extra_video_out = _gmr_path(video_save_path)

    extra_volumes = []
    for ev in [extra_hmr4d, extra_save, extra_video_out]:
        if ev:
            extra_volumes.append(ev)

    # MUJOCO_GL=egl (set in compose) uses EGL with NVIDIA GPU for rendering.
    command = [
        "python", GMR_SCRIPT_CONT,
        f"--gvhmr_pred_file={container_hmr4d}",
        f"--robot={args.robot}",
        f"--save_path={container_save}",
        f"--video_save_path={container_video_out}",
    ]

    if args.record_video:
        command.append("--record_video")
    if args.rate_limit:
        command.append("--rate_limit")
    if args.loop:
        command.append("--loop")

    rc = _docker_compose_run("gmr", command, extra_volumes or None)
    if rc != 0:
        print(f"[ERROR] GMR failed for: {video_name} (exit code {rc})")
        sys.exit(rc)

    print(f"[GMR] PKL saved to: {save_path}")
    return save_path


def run_pkl_to_csv(pkl_path: Path, video_name: str, robot: str) -> Path | None:
    """Converts a GMR PKL to CSV using a temporary directory mounted in the container."""
    (GMR_OUTPUT_HOST / "csv").mkdir(parents=True, exist_ok=True)
    csv_dest = GMR_OUTPUT_HOST / "csv" / f"{robot}_{video_name}.csv"

    print("\n" + "=" * 60)
    print(f"[CSV] Converting: {pkl_path.name}")
    print(f"[CSV] Destination: {csv_dest}")
    print("=" * 60)

    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        tmp_pkl = tmp_dir_path / pkl_path.name
        shutil.copy2(pkl_path, tmp_pkl)

        # Mount the temporary directory as /tmp_work in the GMR container.
        # The script creates /tmp_work/csv/<name>.csv inside the container,
        # which appears as tmp_dir_path/csv/<name>.csv on the host (via bind mount).
        container_tmp = "/tmp_work"
        extra_volumes = [f"{tmp_dir}:{container_tmp}"]

        command = [
            "python", GMR_PKL_TO_CSV_CONT,
            f"--folder={container_tmp}",
        ]

        rc = _docker_compose_run("gmr", command, extra_volumes)
        if rc != 0:
            print(f"[ERROR] PKL→CSV failed for: {pkl_path.name}")
            return None

        generated_csv = tmp_dir_path / "csv" / pkl_path.with_suffix(".csv").name
        if generated_csv.exists():
            shutil.move(str(generated_csv), str(csv_dest))
            print(f"[CSV] Saved to: {csv_dest}")
            return csv_dest
        else:
            print(f"[WARNING] Generated CSV not found: {generated_csv}")
            return None


def process_video(video_path: Path, args, video_name: str = None):
    """Runs the full pipeline (GENMO + GMR) for a single video."""
    video_name = video_name or video_path.stem

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = GENMO_OUTPUTS_HOST / "demo" / video_name

    print(f"\n{'#' * 60}")
    print(f"# Video    : {video_path.name}")
    if video_name != video_path.stem:
        print(f"# Name     : {video_name}")
    print(f"# Output   : {output_dir}")
    print(f"# Robot    : {args.robot}")
    print(f"# Sandwich : {'enabled' if args.sandwich else 'disabled'}")
    print(f"{'#' * 60}")

    # Stage 1 — GENMO
    if args.sandwich:
        hmr4d_results = process_video_sandwich(video_path, video_name, output_dir, args)
    else:
        hmr4d_results = run_genmo(video_path, video_name, output_dir, args)

    # Stage 2 — GMR
    save_path = run_gmr(hmr4d_results, video_name, args)

    # Stage 3 — PKL → CSV
    run_pkl_to_csv(save_path, video_name, args.robot)

    return hmr4d_results, save_path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CopyCat via Docker — Pipeline: Video → SMPL (GENMO) → Robot (GMR)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Input ─────────────────────────────────────────────────────────────────
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--video", "-v", help="Single .mp4 file.")
    input_group.add_argument("--videos_path", help="Folder with .mp4 files (recursive search).")

    # ── GENMO ─────────────────────────────────────────────────────────────────
    parser.add_argument("--video_name", default=None,
                        help="Output folder name (default: file stem).")
    parser.add_argument("--output_dir", default=None,
                        help="GENMO output directory (default: GENMO/outputs/demo/<video_name>).")
    parser.add_argument("--ckpt_path", default=str(DEFAULT_CKPT),
                        help=f"GENMO checkpoint (default: {DEFAULT_CKPT}).")
    parser.add_argument("--exp", default="genmo_lg",
                        help="GENMO experiment config (default: genmo_lg).")
    parser.add_argument("--orig_fps", type=int, default=30,
                        help="Original video FPS (default: 30).")
    parser.add_argument("--static_cam", action=argparse.BooleanOptionalAction, default=True,
                        help="Static camera assumption (default: enabled).")
    parser.add_argument("--force", action="store_true", default=False,
                        help="Force reprocessing even if result already exists.")
    parser.add_argument("--pose", action="store_true", default=False,
                        help="Generate pose/YOLO debug PNGs.")

    # ── Sandwich ──────────────────────────────────────────────────────────────
    parser.add_argument("--sandwich", action=argparse.BooleanOptionalAction, default=True,
                        help="Sandwich mode: stand-still + video + stand-still (default: enabled).")
    parser.add_argument("--anchor_frames", type=int, default=ANCHOR_FRAMES,
                        help=f"Stand-still frames at the prefix (default: {ANCHOR_FRAMES}).")
    parser.add_argument("--suffix_frames", type=int, default=SUFFIX_FRAMES,
                        help=f"Stand-still frames at the suffix (default: {SUFFIX_FRAMES}).")
    parser.add_argument("--neutral_pose_path", default=None,
                        help="Path to .npy or .pt file with neutral body_pose (63,).")
    parser.add_argument("--transition_frames", type=int, default=15,
                        help="Lerp frames at each boundary (default: 15).")

    # ── GMR ───────────────────────────────────────────────────────────────────
    parser.add_argument("--robot", "-r", choices=SUPPORTED_ROBOTS, default="unitree_g1",
                        help="Target robot (default: unitree_g1).")
    parser.add_argument("--save_path", default=None,
                        help="Output path for the PKL (default: GMR/output/pkl/<robot>_<name>.pkl).")
    parser.add_argument("--record_video", action="store_true", default=False,
                        help="Record a GMR visualization video.")
    parser.add_argument("--rate_limit", action="store_true", default=False,
                        help="Limit playback rate to human FPS.")
    parser.add_argument("--loop", action="store_true", default=False,
                        help="Repeat motion indefinitely in the viewer.")
    parser.add_argument("--headless", action="store_true", default=False,
                        help="(Ignored in Docker — always headless via MUJOCO_GL=egl).")

    args = parser.parse_args()

    # ── Validation ────────────────────────────────────────────────────────────
    compose_file = COPYCAT_DIR / "docker-compose.yml"
    if not compose_file.exists():
        print(f"[ERROR] docker-compose.yml not found: {compose_file}")
        sys.exit(1)

    ckpt = Path(args.ckpt_path)
    if not ckpt.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt}")
        print(f"        Place the file at: GENMO/inputs/checkpoints/")
        sys.exit(1)

    if args.headless:
        print("[INFO] --headless is ignored in Docker (MUJOCO_GL=egl is always used in the GMR container).")

    print(f"[Config] Compose    : {compose_file}")
    print(f"[Config] Checkpoint : {ckpt}")
    print(f"[Config] Robot      : {args.robot}")
    if args.sandwich:
        print(f"[Config] Sandwich   : enabled")
        print(f"[Config] Prefix     : {args.anchor_frames} frames ({args.anchor_frames / ANCHOR_FPS:.1f} s @ {ANCHOR_FPS} fps)")
        print(f"[Config] Suffix     : {args.suffix_frames} frames ({args.suffix_frames / ANCHOR_FPS:.1f} s @ {ANCHOR_FPS} fps)")

    # ── Folder mode ───────────────────────────────────────────────────────────
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
                print(f"[WARNING] Skipping {vp.name} (error {e.code})")
                results.append((vp, None, e.code))

        print("\n" + "=" * 60)
        print(f"[DONE] {len(videos)} video(s) processed.")
        for vp, save, err in results:
            print(f"  {vp.name} → {str(save) if save else f'ERROR ({err})'}")
        return

    # ── Single file mode ──────────────────────────────────────────────────────
    video_input = Path(args.video).resolve()
    if not video_input.exists():
        print(f"[ERROR] File not found: {video_input}")
        sys.exit(1)
    if video_input.suffix.lower() != ".mp4":
        print(f"[WARNING] File is not .mp4: {video_input}")

    process_video(video_input, args, video_name=args.video_name)
    print("\n[DONE] Pipeline complete.")


if __name__ == "__main__":
    main()
