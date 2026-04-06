"""
demo_music.py — GENMO Music-Driven Dance Generation Demo
=========================================================

Generates a 3D human dance animation conditioned solely on an input audio file.
The generated motion is in global coordinates (ay-frame) and rendered as an
overlay-free 3D visualisation.

Usage (via Hydra):
    cd /path/to/GENMO
    python scripts/demo/demo_music.py \\
        music_path=wuthering_cut.wav \\
        music_duration=10 \\
        ckpt_path=inputs/checkpoints/genmo.ckpt

All output files are written to:
    outputs/demo_music/<audio_stem>/

Notes
-----
- This script is a standalone sibling of demo_text.py.
- It does NOT modify any text or video functionality.
- The model is conditioned on 35-dim music features at 30fps, matching the
  AIST++ training format (encoded_music_dim=35 in configs/pipeline/dual_mode.yaml).
- The output video is merged with the original audio so the result plays with
  the corresponding music.
"""

import os
import subprocess
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from einops import einsum
from hydra import compose, initialize_config_module
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Ensure repo root is in sys.path (works regardless of cwd)
# ---------------------------------------------------------------------------
workspace_root = Path(__file__).resolve().parents[2]
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

import open3d as o3d
from third_party.GVHMR.hmr4d.utils.geo.hmr_cam import (
    create_camera_sensor,
)
from genmo.utils.geo_transform import (
    apply_T_on_points,
    compute_cam_angvel,
    compute_cam_tvel,
    compute_T_ayfz2ay,
)
from genmo.utils.music_utils import load_music_embed
from genmo.utils.net_utils import detach_to_cpu, to_cuda
from genmo.utils.pylogger import Log
from genmo.utils.tools import find_last_version
from third_party.GVHMR.hmr4d.utils.smplx_utils import make_smplx
from genmo.utils.video_io_utils import (
    merge_videos_horizontal,
    save_video,
)
from genmo.utils.vis.o3d_render import Settings, create_meshes, get_ground
from genmo.utils.vis.renderer import (
    get_global_cameras_static_v2,
    get_ground_params_from_points,
)

CRF = 23  # video quality (17 = lossless, +6 halves size)


# ---------------------------------------------------------------------------
# Helper: build the data dict expected by model.predict()
# ---------------------------------------------------------------------------

def build_music_data(music_embed: torch.Tensor) -> dict:
    """Build the inference data dict from a music embedding tensor.

    Args:
        music_embed: Float tensor of shape ``(L, 35)``, 30 fps music features.

    Returns:
        Data dict compatible with ``GENMO.predict()``.
    """
    length = music_embed.shape[0]

    # Placeholder camera / visual fields (all zeros / identity — no video input)
    R_w2c = torch.eye(3).repeat(length, 1, 1)
    t_w2c = torch.zeros(length, 3)
    T_w2c = torch.eye(4).reshape(1, 4, 4).repeat(length, 1, 1)
    T_w2c[:, :3, :3] = R_w2c
    T_w2c[:, :3, 3] = t_w2c
    cam_angvel = compute_cam_angvel(R_w2c)
    cam_tvel = compute_cam_tvel(t_w2c)

    # Dummy K (1080p, approx.)
    K_fullimg = torch.tensor(
        [[1000.0, 0.0, 540.0],
         [0.0, 1000.0, 960.0],
         [0.0,    0.0,   1.0]]
    ).unsqueeze(0).repeat(length, 1, 1)

    data = {
        "meta": [
            {
                "vid": "music_demo",
                "caption": "",
                "eval_gen_only": True,
            }
        ],
        "length": torch.tensor(length),
        # Music conditioning — the key ingredient
        "music_embed": music_embed,          # (L, 35)
        # Placeholders (no image / 2D / camera signal)
        "bbx_xys": torch.zeros(length, 3),
        "K_fullimg": K_fullimg,
        "f_imgseq": torch.zeros(length, 1024),
        "kp2d": torch.zeros(length, 17, 3),
        "cam_angvel": cam_angvel,
        "cam_tvel": cam_tvel,
        "R_w2c": R_w2c,
        "T_w2c": T_w2c,
        "gt_T_w2c": T_w2c.clone(),
        "gender": "neutral",
        "caption": "",
        "has_text": torch.tensor([False]),
        "mask": {
            "valid":          torch.ones(length).bool(),
            "has_img_mask":   torch.zeros(length).bool(),
            "has_2d_mask":    torch.zeros(length).bool(),
            # No real camera available — set to False so the model does NOT use
            # the placeholder identity matrices as camera conditioning, which
            # would corrupt the global trajectory rollout.
            "has_cam_mask":   torch.zeros(length).bool(),
            "has_audio_mask": torch.zeros(length).bool(),
            "has_music_mask": torch.ones(length).bool(),   # music conditioning ON
        },
    }
    return data


# ---------------------------------------------------------------------------
# Rendering helpers (adapted from demo_text.py render_global_o3d)
# ---------------------------------------------------------------------------

def render_global_o3d(
    verts_glob_list: torch.Tensor,
    faces: torch.Tensor,
    output_path: Path,
    orig_fps: int = 30,
    width: int = 1280,
    height: int = 720,
):
    """Render the global 3-D motion using Open3D offscreen rendering.

    Args:
        verts_glob_list: SMPL vertices in global AY frame, shape ``(T, V, 3)``.
        faces:           SMPL faces tensor, shape ``(F, 3)``.
        output_path:     Path to write the output ``.mp4``.
        orig_fps:        Frame rate of the output video.
        width, height:   Resolution of the rendered video.
    """
    _, _, K = create_camera_sensor(width, height, 24)  # 24 mm lens
    device = verts_glob_list.device

    J_regressor = torch.load(
        "inputs/checkpoints/body_models/smpl_neutral_J_regressor.pt"
    ).to(device)
    joints_glob = einsum(J_regressor, verts_glob_list, "j v, l v i -> l j i")

    mat_settings = Settings()
    color_purple = torch.tensor([0.69019608, 0.39215686, 0.95686275]).to(device)
    color_green  = torch.tensor([0.46666667, 0.90196078, 0.74901961]).to(device)
    color_light_purple = torch.tensor([1.0, 0.65490196, 0.95294118]).to(device)

    T, V, _ = verts_glob_list.shape

    scale, cx, cz = get_ground_params_from_points(joints_glob[:, 0], verts_glob_list)
    scale = max(scale, 3)
    ground_geometry = get_ground(scale * 1.5, cx, cz)

    position, target, up = get_global_cameras_static_v2(
        verts_glob_list.cpu().clone(),
        beta=3.0,
        cam_height_degree=20,
        target_center_height=1.0,
    )

    colors = color_purple[None, :].repeat(T, 1)
    colors_trans = torch.zeros_like(colors)
    colors_trans[:, 0] = torch.linspace(color_green[0], color_light_purple[0], T)
    colors_trans[:, 1] = torch.linspace(color_green[1], color_light_purple[1], T)
    colors_trans[:, 2] = torch.linspace(color_green[2], color_light_purple[2], T)

    lit_mat = mat_settings._materials[Settings.LIT]

    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.camera.set_projection(
        K.cpu().double().numpy(), 0.1, 100.0, float(width), float(height)
    )
    camera = renderer.scene.camera
    camera.look_at(target[:, None], position[:, None], up[:, None])

    gv, gf, gc = ground_geometry
    ground_mesh = create_meshes(gv, gf, gc[..., :3])
    renderer.scene.add_geometry(
        "mesh_ground", ground_mesh, o3d.visualization.rendering.MaterialRecord()
    )

    frames_list = []
    for t, verts in tqdm(enumerate(verts_glob_list), total=T, desc="Rendering Global"):
        mesh = create_meshes(verts, faces, colors[t])
        if t > 0:
            renderer.scene.remove_geometry(f"mesh_{t - 1}")
        renderer.scene.add_geometry(f"mesh_{t}", mesh, lit_mat)
        img = renderer.render_to_image()
        frames_list.append(np.array(img))

    frames_array = np.array(frames_list).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_video(frames_array, output_path, fps=orig_fps, crf=CRF)
    Log.info(f"[Render] Saved to {output_path}")


def add_audio_to_video(video_path: Path, audio_path: Path, output_path: Path):
    """Merge a silent video with the original music audio using ffmpeg.

    Args:
        video_path:  Path to the rendered (silent) video.
        audio_path:  Path to the music audio file.
        output_path: Path for the final video with audio.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",           # trim to the shorter of video / audio
        str(output_path),
    ]
    Log.info(f"[Audio] Merging audio: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        Log.warn(f"[Audio] ffmpeg error:\n{result.stderr}")
    else:
        Log.info(f"[Audio] Saved with audio: {output_path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

@hydra.main(version_base="1.3", config_path="../../configs", config_name="demo")
def main(cfg):
    # ------------------------------------------------------------------
    # 1. Validate config
    # ------------------------------------------------------------------
    if not hasattr(cfg, "music_path") or cfg.music_path is None:
        raise ValueError(
            "music_path must be provided. "
            "Example: python demo_music.py music_path=wuthering_cut.wav"
        )
    music_path = Path(cfg.music_path)
    if not music_path.exists():
        raise FileNotFoundError(f"Audio file not found: {music_path}")

    duration: float | None = getattr(cfg, "music_duration", None)
    music_fps: int = getattr(cfg, "music_fps", 30)
    orig_fps: int = music_fps  # render at the same fps

    audio_stem = music_path.stem

    # Use the audio filename as the run name
    cfg.text1_video_name = audio_stem
    output_dir = Path(getattr(cfg, "output_root", "outputs/demo")) / "demo_music" / audio_stem
    output_dir.mkdir(parents=True, exist_ok=True)

    Log.info(f"[Music Demo] Audio : {music_path}")
    Log.info(f"[Music Demo] Duration: {duration}s" if duration else "[Music Demo] Duration: full")
    Log.info(f"[Music Demo] Output: {output_dir}")
    Log.info(f"[GPU]: {torch.cuda.get_device_name()}")

    # ------------------------------------------------------------------
    # 2. Extract music features  →  (L, 35)
    # ------------------------------------------------------------------
    Log.info("[Preprocess] Extracting music features…")
    music_embed = load_music_embed(music_path, target_fps=music_fps, duration=duration)
    length = music_embed.shape[0]
    Log.info(f"[Preprocess] music_embed shape: {music_embed.shape}  ({length / music_fps:.1f}s)")

    # ------------------------------------------------------------------
    # 3. Build data dict
    # ------------------------------------------------------------------
    data = build_music_data(music_embed)

    # ------------------------------------------------------------------
    # 4. Load model and run inference
    # ------------------------------------------------------------------
    results_path = output_dir / "hmr4d_results.pt"

    if results_path.exists():
        # Always re-run inference (in case fixes were applied)
        Log.info(f"[GENMO] Removing stale cache: {results_path}")
        results_path.unlink()

    if not results_path.exists():
        Log.info("[GENMO] Loading model…")
        import hydra as hydra_mod
        model = hydra_mod.utils.instantiate(cfg.model, _recursive_=False)

        # Resolve checkpoint path
        if cfg.get("ckpt_path") is not None and cfg.ckpt_path:
            ckpt_path = cfg.ckpt_path
        else:
            test_cp = cfg.get("test_checkpoint", "last")
            if cfg.version is None:
                version = find_last_version(cfg.ckpt_dir)
                if version is None:
                    raise FileNotFoundError(
                        f"No checkpoint found in {cfg.ckpt_dir}. "
                        "Specify ckpt_path= directly."
                    )
                ckpt_path = os.path.join(
                    cfg.ckpt_dir, f"version_{version}", "checkpoints", "last.ckpt"
                )
            else:
                ckpt_path = os.path.join(
                    cfg.ckpt_dir, f"version_{cfg.version}", "checkpoints", "last.ckpt"
                )

        Log.info(f"[GENMO] Checkpoint: {ckpt_path}")
        model.load_pretrained_model(ckpt_path)
        model = model.eval().cuda()

        tic = Log.sync_time()
        # static_cam=False → uses pp_static_joint (correct for camera-less
        # music-only generation). pp_static_joint_cam (static_cam=True) needs
        # a real camera to anchor the global trajectory and causes drift here.
        pred = model.predict(data, static_cam=False, postproc=True)
        pred = detach_to_cpu(pred)
        Log.info(
            f"[GENMO] Elapsed: {Log.sync_time() - tic:.2f}s "
            f"for {length / music_fps:.1f}s of audio"
        )
        torch.save(pred, results_path)
        Log.info(f"[GENMO] Saved predictions to {results_path}")
    else:
        Log.info(f"[GENMO] Loading cached predictions from {results_path}")
        pred = torch.load(results_path)

    # ------------------------------------------------------------------
    # 5. Post-process: move to origin, face -Z
    # ------------------------------------------------------------------
    smplx = make_smplx("supermotion").cuda()
    smplx2smpl = torch.load(
        "inputs/checkpoints/body_models/smplx2smpl_sparse.pt"
    ).cuda()
    faces_smpl = make_smplx("smpl").faces

    J_regressor = torch.load(
        "inputs/checkpoints/body_models/smpl_neutral_J_regressor.pt"
    ).cuda()

    smplx_out = smplx(**to_cuda(pred["smpl_params_global"]))
    pred_ay_verts = torch.stack(
        [torch.matmul(smplx2smpl, v_) for v_ in smplx_out.vertices]
    )

    def move_to_start_point_face_z(verts, J_reg):
        """XZ → origin, start from ground, face -Z."""
        verts = verts.clone()
        offset = einsum(J_reg, verts[0], "j v, v i -> j i")[0]  # (3,)
        offset[1] = verts[:, :, 1].min()
        verts -= offset
        T_ay2ayfz = compute_T_ayfz2ay(
            einsum(J_reg, verts[[0]], "j v, l v i -> l j i"), inverse=True
        )
        verts = apply_T_on_points(verts, T_ay2ayfz)
        return verts

    verts_glob = move_to_start_point_face_z(pred_ay_verts, J_regressor)
    faces_tensor = torch.from_numpy(faces_smpl.astype("int")).to(verts_glob.device)

    # ------------------------------------------------------------------
    # 6. Render
    # ------------------------------------------------------------------
    render_width, render_height = 1280, 720
    silent_video_path = output_dir / "2_global_silent.mp4"
    final_video_path  = output_dir / "2_global.mp4"

    if not final_video_path.exists():
        render_global_o3d(
            verts_glob_list=verts_glob,
            faces=faces_tensor,
            output_path=silent_video_path,
            orig_fps=orig_fps,
            width=render_width,
            height=render_height,
        )

        # Merge with audio
        add_audio_to_video(silent_video_path, music_path, final_video_path)

        # Clean up silent version if audio merge succeeded
        if final_video_path.exists():
            silent_video_path.unlink(missing_ok=True)
    else:
        Log.info(f"[Render] Video already exists: {final_video_path}")

    Log.info("=" * 60)
    Log.info(f"[Done] Music-driven dance generation complete!")
    Log.info(f"       Output: {final_video_path}")
    Log.info("=" * 60)


if __name__ == "__main__":
    main()
