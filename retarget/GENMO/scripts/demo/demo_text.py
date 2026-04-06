import argparse
import copy
import os
import subprocess
from glob import glob
from pathlib import Path

import cv2
import ffmpeg
import hydra
import imageio.v3 as iio
import numpy as np
import open3d as o3d
import torch
from einops import einsum
from hydra import compose, initialize_config_module
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from genmo.utils.tools import rsync_file_from_remote, find_last_version
import sys

workspace_root = Path(__file__).resolve().parents[2]
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from third_party.GVHMR.hmr4d.utils.geo.hmr_cam import (
    convert_K_to_K4,
    create_camera_sensor,
    estimate_K,
    get_bbx_xys_from_xyxy,
)
from genmo.utils.geo_transform import (
    apply_T_on_points,
    compute_cam_angvel,
    compute_cam_tvel,
    compute_T_ayfz2ay,
    normalize_T_w2c,
)
from genmo.utils.net_utils import detach_to_cpu, to_cuda
from third_party.GVHMR.hmr4d.utils.preproc import (
    Extractor,
    Tracker,
    VitPoseExtractor,
)
from genmo.utils.pylogger import Log
from third_party.GVHMR.hmr4d.utils.smplx_utils import make_smplx
from genmo.utils.video_io_utils import (
    concat_videos,
    get_video_lwh,
    get_video_reader,
    get_writer,
    merge_videos_horizontal,
    read_video_np,
    save_video,
)
from genmo.utils.vis.cv2_utils import (
    draw_bbx_xyxy_on_image_batch,
    draw_coco17_skeleton_batch,
)
from genmo.utils.vis.o3d_render import Settings, create_meshes, get_ground
from genmo.utils.vis.renderer import (
    Renderer,
    get_global_cameras,
    get_global_cameras_static,
    get_global_cameras_static_v2,
    get_ground_params_from_points,
)
from genmo.utils.rotation_conversions import quaternion_to_matrix

CRF = 23  # 17 is lossless, every +6 halves the mp4 size


def normalize_pose_flag_for_hydra():
    """Allow using '--pose' in CLI by translating it to a Hydra override."""
    if "--pose" not in sys.argv:
        return

    sys.argv = [arg for arg in sys.argv if arg != "--pose"]
    has_pose_override = any(
        arg.startswith("pose=") or arg.startswith("+pose=") for arg in sys.argv[1:]
    )
    if not has_pose_override:
        sys.argv.append("pose=true")


def create_text_video(
    output_path,
    text,
    fps=30,
    num_frames=90,
    width=1280,
    height=720,
    font_path="arial.ttf",
    font_size=60,
    text_color=(255, 255, 255),
):
    """
    Create a video with text centered on a black background.
    Text will automatically wrap if it's too long for the screen width.

    Args:
        output_path: Path to save the output video
        text: Text to display
        fps: Frames per second
        num_frames: Total number of frames
        width: Video width
        height: Video height
        font_path: Path to font file
        font_size: Font size
        text_color: RGB tuple for text color
    """
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # or 'XVID'
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Create a black frame with text
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        # Fallback to default font if specified font not found
        font = ImageFont.load_default(size=font_size)

    # Create a black image with text
    img = Image.new("RGB", (width, height), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Calculate max width for text (with some margin)
    max_text_width = width * 0.9

    # Wrap text
    lines = []
    words = text.split()
    current_line = words[0]

    for word in words[1:]:
        # Check if adding this word exceeds the max width
        test_line = current_line + " " + word
        test_width = draw.textbbox((0, 0), test_line, font=font)[2]

        if test_width <= max_text_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word

    lines.append(current_line)  # Add the last line

    # Calculate total text height
    line_height = font_size * 1.2  # Add some line spacing
    total_text_height = len(lines) * line_height

    # Calculate starting y position to center all text vertically
    y_position = (height - total_text_height) // 2

    # Draw each line of text centered horizontally
    for line in lines:
        line_width = draw.textbbox((0, 0), line, font=font)[2]
        x_position = (width - line_width) // 2
        draw.text((x_position, y_position), line, font=font, fill=text_color)
        y_position += line_height

    # Convert PIL Image to OpenCV format
    frame = np.array(img)
    # Convert RGB to BGR (OpenCV uses BGR)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Write the frame to video multiple times
    for _ in range(num_frames):
        video.write(frame)

    # Release the video writer
    video.release()
    print(f"Video saved to {output_path}")


def has_video_input(cfg):
    """
    Checks if there is video as input.
    
    Args:
        cfg: Hydra configuration
        
    Returns:
        bool: True if video1_path exists and is valid
    """
    if not hasattr(cfg, 'video1_path') or cfg.video1_path is None:
        return False
    video_path = Path(cfg.video1_path)
    return video_path.exists() and video_path.is_file()


def has_text_input(cfg):
    """
    Checks if there is text as input.
    
    Args:
        cfg: Hydra configuration
        
    Returns:
        bool: True if text1 is defined and not empty
    """
    if hasattr(cfg, 'text1_file') and cfg.text1_file is not None:
        return True
    if hasattr(cfg, 'text1') and cfg.text1 is not None and str(cfg.text1).strip():
        return True
    return False


def save_preprocess_debug_pngs(video_path, bbx_path, vitpose, output_dir, vid=1, frame_idx=0):
    """
    Save one YOLO bbox PNG and one ViTPose PNG for quick inspection.
    This is intentionally lightweight and runs once per preprocess call.
    """
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if frame_idx > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            Log.warn(f"[Debug PNG] Could not read frame {frame_idx} from video: {video_path}")
            return

        bbx_data = torch.load(bbx_path, map_location="cpu")
        bbx_xyxy = bbx_data.get("bbx_xyxy", None)

        if bbx_xyxy is not None:
            if isinstance(bbx_xyxy, torch.Tensor):
                bbx_xyxy = bbx_xyxy.cpu().numpy()
            frame_idx_bbx = min(frame_idx, len(bbx_xyxy) - 1)
            x1, y1, x2, y2 = bbx_xyxy[frame_idx_bbx].astype(int).tolist()

            yolo_img = frame.copy()
            cv2.rectangle(yolo_img, (x1, y1), (x2, y2), (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(
                yolo_img,
                "YOLO track",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            yolo_png = output_dir / f"yolo_debug_vid{vid}_f{frame_idx_bbx:04d}.png"
            cv2.imwrite(str(yolo_png), yolo_img)
            Log.info(f"[Debug PNG] Saved YOLO bbox image: {yolo_png}")
        else:
            Log.warn(f"[Debug PNG] bbx_xyxy not found in {bbx_path}; skipping YOLO PNG")

        if isinstance(vitpose, tuple):
            vitpose = vitpose[0]
        if not isinstance(vitpose, torch.Tensor):
            vitpose = torch.as_tensor(vitpose)

        if vitpose.ndim == 3 and vitpose.shape[0] > 0:
            frame_idx_pose = min(frame_idx, int(vitpose.shape[0]) - 1)
            kps = vitpose[frame_idx_pose].detach().cpu().numpy()  # (17, 3)

            pose_img = frame.copy()
            edges = [
                (0, 1), (0, 2), (1, 3), (2, 4),
                (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                (5, 11), (6, 12), (11, 12),
                (11, 13), (13, 15), (12, 14), (14, 16),
            ]

            for a, b in edges:
                xa, ya, ca = kps[a]
                xb, yb, cb = kps[b]
                if ca > 0.2 and cb > 0.2:
                    cv2.line(
                        pose_img,
                        (int(xa), int(ya)),
                        (int(xb), int(yb)),
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

            for x, y, c in kps:
                color = (0, 0, 255) if c > 0.2 else (100, 100, 100)
                cv2.circle(pose_img, (int(x), int(y)), 4, color, -1, cv2.LINE_AA)

            pose_png = output_dir / f"pose_debug_vid{vid}_f{frame_idx_pose:04d}.png"
            cv2.imwrite(str(pose_png), pose_img)
            Log.info(f"[Debug PNG] Saved pose image: {pose_png}")
        else:
            Log.warn("[Debug PNG] Unexpected vitpose format; skipping pose PNG")

    except Exception as e:
        Log.warn(f"[Debug PNG] Failed to export debug PNGs: {e}")


def load_video_data(cfg, vid=1):
    """
    Loads processed video data.
    
    Args:
        cfg: Hydra configuration
        vid: Video ID (1 or 2)
        
    Returns:
        dict: Dictionary with video data in the format expected by the model
    """
    paths = cfg.paths
    if vid == 1:
        video_path = cfg.video1_path
        bbx_path = paths.bbx1
        vitpose_path = paths.vitpose1
        vit_features_path = paths.vit_features1
        slam_path = paths.slam1
        static_cam = cfg.static_cam1
        video_name = getattr(cfg, 'video1_name', None) or Path(video_path).stem
    else:
        video_path = cfg.video2_path
        bbx_path = paths.bbx2
        vitpose_path = paths.vitpose2
        vit_features_path = paths.vit_features2
        slam_path = paths.slam2
        static_cam = cfg.static_cam2
        video_name = getattr(cfg, 'video2_name', None) or Path(video_path).stem
    
    # Load processed data
    length, width, height = get_video_lwh(video_path)
    
    # Load bounding boxes
    bbx_data = torch.load(bbx_path)
    bbx_xys = bbx_data["bbx_xys"]
    
    # Load 2D keypoints
    vitpose = torch.load(vitpose_path)
    if isinstance(vitpose, tuple):
        vitpose = vitpose[0]
    kp2d = vitpose
    
    # Load image features
    f_imgseq = torch.load(vit_features_path)
    
    # Load or create camera
    if static_cam:
        R_w2c = torch.eye(3).repeat(length, 1, 1)
        t_w2c = torch.zeros(length, 3)
    else:
        traj = torch.load(slam_path)
        if traj.shape[1] == 4:  # If it's a 4x4 transformation matrix
            R_w2c = torch.from_numpy(traj[:, :3, :3])
            t_w2c = torch.from_numpy(traj[:, :3, 3])
        else:  # If it's only a 3x3 rotation
            R_w2c = torch.from_numpy(traj[:, :3, :3])
            t_w2c = torch.zeros(length, 3)
    
    # Estimate K if necessary
    K_fullimg = estimate_K(width, height).repeat(length, 1, 1)
    
    # Calculate camera velocities
    cam_angvel = compute_cam_angvel(R_w2c)
    cam_tvel = compute_cam_tvel(t_w2c)
    
    # Create transformations
    T_w2c = torch.eye(4).reshape(1, 4, 4).repeat(length, 1, 1)
    T_w2c[:, :3, :3] = R_w2c
    T_w2c[:, :3, 3] = t_w2c
    gt_T_w2c = T_w2c.clone()
    
    # Determinar gênero (padrão: neutral)
    gender = getattr(cfg, 'gender', 'neutral')
    
    # Create data structure
    data = {
        "meta": [
            {
                "vid": video_name,
                "caption": getattr(cfg, 'text1', '') if has_text_input(cfg) else '',
            }
        ],
        "length": torch.tensor(length),
        "bbx_xys": bbx_xys,
        "K_fullimg": K_fullimg,
        "f_imgseq": f_imgseq,
        "kp2d": kp2d,
        "cam_angvel": cam_angvel,
        "cam_tvel": cam_tvel,
        "R_w2c": R_w2c,
        "T_w2c": T_w2c,
        "gt_T_w2c": gt_T_w2c,
        "gender": gender,
        "caption": str(getattr(cfg, 'text1', '') or ''),
        "has_text": torch.tensor([has_text_input(cfg)]),
        "mask": {
            "valid": torch.ones(length),
            "has_img_mask": torch.ones(length).bool(),
            "has_2d_mask": torch.ones(length).bool(),
            "has_cam_mask": torch.ones(length).bool(),
            "has_audio_mask": torch.zeros(length).bool(),
            "has_music_mask": torch.zeros(length).bool(),
        },
    }
    
    return data


@torch.no_grad()
def run_preprocess_text(cfg):
    Log.info("[Preprocess] Start text!")
    tic = Log.time()

    text_1 = cfg.text1

    text_length = cfg.text_length
    bbx_xys = torch.zeros(text_length, 3)
    
    return_data = {
        "meta": {
            "vid": "text",
            "caption": [text_1],
        },
        "length": torch.tensor(text_length),
        "bbx_xys": bbx_xys,
        "K_fullimg": torch.eye(3).repeat(text_length, 3, 3),
        "f_imgseq": torch.zeros(text_length, 1024),
        "kp2d": torch.zeros(text_length, 17, 3),
        # "cam_angvel": torch.zeros_like(
        #     compute_cam_angvel(torch.eye(3)[None].repeat(text_length, 1, 1))
        # ),
        "cam_angvel": compute_cam_angvel(torch.eye(3)[None].repeat(text_length, 1, 1)),
        "cam_tvel": torch.zeros(text_length, 3),
        "R_w2c": torch.eye(3).reshape(1, 3, 3).repeat(text_length, 1, 1),
        "T_w2c": torch.eye(4).reshape(1, 4, 4).repeat(text_length, 1, 1),
        "gt_T_w2c": torch.eye(4).reshape(1, 4, 4).repeat(text_length, 1, 1),
        "gender": "neutral",
        "caption": text_1,
        "has_text": torch.tensor([True]),
        "mask": {
            "valid": torch.ones(text_length),
            # "vitpose": False,
            # "bbx_xys": False,
            # "f_imgseq": False,
            # "spv_incam_only": False,
            "has_img_mask": torch.zeros(text_length).bool(),
            "has_2d_mask": torch.zeros(text_length).bool(),
            "has_cam_mask": torch.ones(text_length).bool(),
            "has_audio_mask": torch.zeros(text_length).bool(),
            "has_music_mask": torch.zeros(text_length).bool(),
        },
    }
    return return_data


@torch.no_grad()
def run_preprocess(cfg, vid=1):
    Log.info(f"[Preprocess] Start {vid}!")
    tic = Log.time()
    paths = cfg.paths
    
    # Garantir que o diretório preprocess existe
    Path(cfg.preprocess_dir).mkdir(parents=True, exist_ok=True)
    
    # video_path = cfg.video_path
    if vid == 1:
        video_path = cfg.video1_path
    else:
        video_path = cfg.video2_path
    bbx_path = paths.bbx1 if vid == 1 else paths.bbx2
    bbx_xyxy_video_overlay_path = (
        paths.bbx_xyxy_video_overlay1 if vid == 1 else paths.bbx_xyxy_video_overlay2
    )
    vitpose_path = paths.vitpose1 if vid == 1 else paths.vitpose2
    vitpose_video_overlay_path = (
        paths.vitpose_video_overlay1 if vid == 1 else paths.vitpose_video_overlay2
    )
    slam_path = paths.slam1 if vid == 1 else paths.slam2
    static_cam = cfg.static_cam1 if vid == 1 else cfg.static_cam2
    vimo_pred_path = paths.vimo_pred1 if vid == 1 else paths.vimo_pred2
    vit_features_path = paths.vit_features1 if vid == 1 else paths.vit_features2
    verbose = cfg.verbose

    # Get bbx tracking result
    if not Path(bbx_path).exists():
        # Garantir que o diretório do arquivo existe
        Path(bbx_path).parent.mkdir(parents=True, exist_ok=True)
        tracker = Tracker()
        bbx_xyxy = tracker.get_one_track(video_path).float()  # (L, 4)
        bbx_xys = get_bbx_xys_from_xyxy(
            bbx_xyxy, base_enlarge=1.2
        ).float()  # (L, 3) apply aspect ratio and enlarge
        torch.save({"bbx_xyxy": bbx_xyxy, "bbx_xys": bbx_xys}, bbx_path)
        del tracker
    else:
        bbx_xys = torch.load(bbx_path)["bbx_xys"]
        Log.info(f"[Preprocess] bbx (xyxy, xys) from {bbx_path}")
    if verbose:
        video = read_video_np(video_path)
        bbx_xyxy = torch.load(bbx_path)["bbx_xyxy"]
        video_overlay = draw_bbx_xyxy_on_image_batch(bbx_xyxy, video)
        save_video(video_overlay, bbx_xyxy_video_overlay_path)

    # Get VitPose
    if not Path(vitpose_path).exists():
        Path(vitpose_path).parent.mkdir(parents=True, exist_ok=True)
        Log.info(f"[Pose] Running VitPose extraction from video: {video_path}")
        vitpose_extractor = VitPoseExtractor()
        vitpose = vitpose_extractor.extract(video_path, bbx_xys)
        Log.info(f"[Pose] VitPose extraction finished. Raw type={type(vitpose)}")
        torch.save(vitpose, vitpose_path)
        Log.info(f"[Pose] Saved VitPose to: {vitpose_path}")
        del vitpose_extractor
    else:
        vitpose = torch.load(vitpose_path)
        Log.info(f"[Pose] Loaded cached VitPose from: {vitpose_path}")
    if verbose:
        video = read_video_np(video_path)
        video_overlay = draw_coco17_skeleton_batch(video, vitpose, 0.5)
        save_video(video_overlay, vitpose_video_overlay_path)

    if isinstance(vitpose, tuple):
        vitpose = vitpose[0]

    if isinstance(vitpose, torch.Tensor) and vitpose.ndim == 3 and vitpose.shape[0] > 0:
        kp0 = vitpose[0, 0].tolist()
        Log.info(
            "[Pose] Final pose tensor shape={} | frame0 kp0(x,y,conf)=({:.1f}, {:.1f}, {:.3f})".format(
                tuple(vitpose.shape), kp0[0], kp0[1], kp0[2]
            )
        )
    else:
        Log.warn(f"[Pose] Unexpected VitPose format: type={type(vitpose)}")

    # Export debug PNGs only when requested via pose flag.
    if getattr(cfg, "pose", False):
        save_preprocess_debug_pngs(
            video_path=video_path,
            bbx_path=bbx_path,
            vitpose=vitpose,
            output_dir=cfg.output_dir,
            vid=vid,
            frame_idx=0,
        )
    else:
        Log.info("[Debug PNG] Skipped. Use --pose to export YOLO/pose PNGs.")

    # Get DROID-SLAM results
    if not static_cam:  # use slam to get cam rotation
        if not Path(slam_path).exists():
            length, width, height = get_video_lwh(video_path)
            K_fullimg = estimate_K(width, height)
            intrinsics = convert_K_to_K4(K_fullimg)
            cam_int = [
                K_fullimg[0, 0],
                K_fullimg[1, 1],
                K_fullimg[0, 2],
                K_fullimg[1, 2],
            ]
            out_dir = os.path.dirname(slam_path)
            np.save(f"{out_dir}/cam_int.npy", cam_int)

            # parse video to frames
            video = read_video_np(video_path)
            img_dir = os.path.join(os.path.dirname(video_path), f"imgs_{vid}")
            os.makedirs(img_dir, exist_ok=True)
            for i, frame in enumerate(video):
                cv2.imwrite(f"{img_dir}/{i:06d}.jpg", frame[..., ::-1])
                i += 1

            cmd = f"python tools/estimate_camera_dir.py --img_dir {img_dir} --out_dir {out_dir}"
            Log.info(f"[DROID-SLAM] {cmd}")
            subprocess.run(cmd, shell=True)

        else:
            Log.info(f"[Preprocess] slam results from {slam_path}")
    else:
        length, width, height = get_video_lwh(video_path)
        K_fullimg = estimate_K(width, height)
        intrinsics = convert_K_to_K4(K_fullimg)
        cam_int = [
            K_fullimg[0, 0],
            K_fullimg[1, 1],
            K_fullimg[0, 2],
            K_fullimg[1, 2],
        ]
        out_dir = os.path.dirname(slam_path)
        np.save(f"{out_dir}/cam_int.npy", cam_int)

    # Get vit features
    if not Path(vit_features_path).exists():
        Path(vit_features_path).parent.mkdir(parents=True, exist_ok=True)
        extractor = Extractor()
        vit_features = extractor.extract_video_features(video_path, bbx_xys)
        torch.save(vit_features, vit_features_path)
        del extractor
    else:
        Log.info(f"[Preprocess] vit_features from {vit_features_path}")

    Log.info(f"[Preprocess] End. Time elapsed: {Log.time() - tic:.2f}s")


def render_incam(cfg, vid_slice, vid=1):
    incam_video_path = (
        Path(cfg.paths.incam_video1) if vid == 1 else Path(cfg.paths.incam_video2)
    )
    if incam_video_path.exists():
        Log.info(f"[Render Incam] Video already exists at {incam_video_path}")
        return

    pred_full = torch.load(cfg.paths.hmr4d_results)
    pred = {"smpl_params_incam": pred_full["smpl_params_incam"]}
    start_idx, end_idx = vid_slice[vid]
    for k in pred_full.keys():
        if k not in ["smpl_params_incam", "K_fullimg"]:
            continue
        if isinstance(pred_full[k], dict):
            pred[k] = {}
            for kk in pred_full[k].keys():
                pred[k][kk] = pred_full[k][kk][start_idx:end_idx]
        else:
            pred[k] = pred_full[k][start_idx:end_idx]

    smplx = make_smplx("supermotion").cuda()
    smplx2smpl = torch.load("inputs/checkpoints/body_models/smplx2smpl_sparse.pt").cuda()
    faces_smpl = make_smplx("smpl").faces  # Usar faces do SMPL (compatível com vértices convertidos)

    # smpl
    smplx_out = smplx(**to_cuda(pred["smpl_params_incam"]))
    pred_c_verts = torch.stack(
        [torch.matmul(smplx2smpl, v_) for v_ in smplx_out.vertices]
    )

    # -- rendering code -- #
    video_path = cfg.video1_path if vid == 1 else cfg.video2_path
    video_30fps_path = str(video_path).replace(".mp4", "_30fps.mp4")

    # convert to 30 fps
    fps = cfg.orig_fps1 if vid == 1 else cfg.orig_fps2
    if fps != 30:
        stream = ffmpeg.input(video_path).filter("setpts", f"{30.0 / fps}*PTS")
        output = ffmpeg.output(stream, video_30fps_path)
        ffmpeg.run(output, overwrite_output=True, quiet=True)
    else:
        video_30fps_path = video_path

    length, width, height = get_video_lwh(video_30fps_path)
    K = pred["K_fullimg"][0]

    # renderer
    renderer = Renderer(width, height, device="cuda", faces=faces_smpl, K=K)
    reader = get_video_reader(video_30fps_path)  # (F, H, W, 3), uint8, numpy
    bbx_path = cfg.paths.bbx1 if vid == 1 else cfg.paths.bbx2
    bbx_xys_render = torch.load(bbx_path)["bbx_xys"]
    color = torch.ones(3).float().cuda() * 0.8
    color_purple = torch.tensor([0.69019608, 0.39215686, 0.95686275]).cuda()
    color[0] = color_purple[0]
    color[1] = color_purple[1]
    color[2] = color_purple[2]

    # -- render mesh -- #
    verts_incam = pred_c_verts
    assert abs(get_video_lwh(video_30fps_path)[0] - len(verts_incam)) < 10, (
        f"Video length mismatch: {get_video_lwh(video_30fps_path)[0]} != {len(verts_incam)}"
    )
    
    # Acumular frames e escrever todos de uma vez (mais confiável que write_frame)
    frames_list = []
    for i, img_raw in tqdm(
        enumerate(reader),
        total=get_video_lwh(video_30fps_path)[0],
        desc=f"Rendering Incam",
    ):
        if i >= verts_incam.shape[0]:
            break
        img = renderer.render_mesh(
            verts_incam[i].cuda(), img_raw, [color[0], color[1], color[2]]
        )

        # # bbx
        # bbx_xys_ = bbx_xys_render[i].cpu().numpy()
        # lu_point = (bbx_xys_[:2] - bbx_xys_[2:] / 2).astype(int)
        # rd_point = (bbx_xys_[:2] + bbx_xys_[2:] / 2).astype(int)
        # img = cv2.rectangle(img, lu_point, rd_point, (255, 178, 102), 2)

        frames_list.append(img)
    reader.close()
    
    # Escrever todos os frames de uma vez
    frames_array = np.array(frames_list).astype(np.uint8)
    save_video(frames_array, incam_video_path, fps=30, crf=CRF)


def render_global_o3d(cfg, orig_fps):
    global_video_path = Path(cfg.paths.global_video)
    # if global_video_path.exists():
    #     Log.info(f"[Render Global] Video already exists at {global_video_path}")
    #     return

    debug_cam = False
    pred = torch.load(cfg.paths.hmr4d_results)
    smplx = make_smplx("supermotion").cuda()
    smplx2smpl = torch.load("inputs/checkpoints/body_models/smplx2smpl_sparse.pt").cuda()
    faces_smpl = make_smplx("smpl").faces  # Usar faces do SMPL (compatível com vértices convertidos)
    J_regressor = torch.load(
        "inputs/checkpoints/body_models/smpl_neutral_J_regressor.pt"
    ).cuda()

    # smpl
    smplx_out = smplx(**to_cuda(pred["smpl_params_global"]))
    pred_ay_verts = torch.stack(
        [torch.matmul(smplx2smpl, v_) for v_ in smplx_out.vertices]
    )

    def move_to_start_point_face_z(verts, J_regressor):
        "XZ to origin, Start from the ground, Face-Z"
        # position
        verts = verts.clone()  # (L, V, 3)
        offset = einsum(J_regressor, verts[0], "j v, v i -> j i")[0]  # (3)
        offset[1] = verts[:, :, [1]].min()
        verts = verts - offset
        # face direction
        T_ay2ayfz = compute_T_ayfz2ay(
            einsum(J_regressor, verts[[0]], "j v, l v i -> l j i"), inverse=True
        )
        verts = apply_T_on_points(verts, T_ay2ayfz)
        return verts

    verts_glob_list = move_to_start_point_face_z(pred_ay_verts, J_regressor)
    joints_glob_list = einsum(J_regressor, verts_glob_list, "j v, l v i -> l j i")
    length = verts_glob_list.shape[0]

    # -- rendering code -- #
    video_path = cfg.text1_video_path
    # orig_fps = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS)
    length, width, height = get_video_lwh(video_path)
    _, _, K = create_camera_sensor(width, height, 24)  # render as 24mm lens
    device = verts_glob_list.device
    # global_video_path = f"out/{vis_type}_video/{fname}.mp4"
    os.makedirs(os.path.dirname(global_video_path), exist_ok=True)

    mat_settings = Settings()

    color_purple = torch.tensor([0.69019608, 0.39215686, 0.95686275]).to(device)
    color_green = torch.tensor([0.46666667, 0.90196078, 0.74901961]).to(device)
    color_light_purple = torch.tensor([1.0, 0.65490196, 0.95294118]).to(device)

    # -- render mesh -- #
    scale, cx, cz = get_ground_params_from_points(
        joints_glob_list[:, 0], verts_glob_list
    )
    scale = max(scale, 3)
    ground_geometry = get_ground(scale * 1.5, cx, cz)
    # color = torch.ones(3).float().cuda() * 0.8

    T, V, _ = verts_glob_list.shape

    position, target, up = get_global_cameras_static_v2(
        # verts_list[0].cpu(),
        verts_glob_list.cpu().clone(),
        beta=3.0,
        # beta=4.0,
        cam_height_degree=20,
        target_center_height=1.0,
    )

    trans_mat_box = mat_settings._materials[Settings.Transparency]
    lit_mat_box = mat_settings._materials[Settings.LIT]

    colors = color_purple[None, :].repeat(T, 1)
    # colors[:, 0] = torch.linspace(color_green[0], color_purple[0], T)
    # colors[:, 1] = torch.linspace(color_green[1], color_purple[1], T)
    # colors[:, 2] = torch.linspace(color_green[2], color_purple[2], T)

    colors_trans = torch.zeros_like(colors)
    colors_trans[:, 0] = torch.linspace(color_green[0], color_light_purple[0], T)
    colors_trans[:, 1] = torch.linspace(color_green[1], color_light_purple[1], T)
    colors_trans[:, 2] = torch.linspace(color_green[2], color_light_purple[2], T)
    faces = torch.from_numpy(faces_smpl.astype("int")).to(device)

    # colors = torch.stack([torch.from_numpy(color_rgb[i % len(color_rgb)]).float().cuda() for i in range(len(verts_glob_list))], dim=0)
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    # renderer.scene.camera.set_projection(K[0, 0], K[1, 1], K[0, 2], K[1, 2], width, height, 0.1, 100.0)
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
    # faces_list = list(torch.unbind(faces, dim=0))  # + [gf]

    T_c2w = camera.get_view_matrix()
    R_w2c = T_c2w[:3, :3].T
    
    # Acumular frames e escrever todos de uma vez (mais confiável que write_frame)
    frames_list = []
    for t, verts in tqdm(enumerate(verts_glob_list)):
        verts = verts_glob_list[t]  # + [gv]
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.base_color = [0.9, 0.9, 0.9, 0.3 + t * 0.7 / T]
        mat.shader = Settings.Transparency
        # mat.opacity = (i + 1) / N
        mat.thickness = 1.0
        mat.transmission = 1.0
        mat.absorption_distance = 10
        mat.absorption_color = [0.5, 0.5, 0.5]

        mesh = create_meshes(verts, faces, colors[t])
        if t > 0:
            renderer.scene.remove_geometry(f"mesh_{t - 1}")
        # img = renderer.render_to_image()

        renderer.scene.add_geometry(f"mesh_{t}", mesh, lit_mat_box)
        # mesh = create_meshes(verts[i], faces_list[i], colors[t])
        # renderer.scene.add_geometry(f"mesh_{i}_{t}", mesh, mat)

        img = renderer.render_to_image()
        # import ipdb; ipdb.set_trace()
        # o3d.io.write_image("out/tmp.png", img)
        # img = cv2.imread("out/tmp.png")
        frames_list.append(np.array(img))
    
    # Escrever todos os frames de uma vez
    frames_array = np.array(frames_list).astype(np.uint8)
    save_video(frames_array, global_video_path, fps=orig_fps, crf=CRF)
    print(f"Saved to {global_video_path}")


def render_global(cfg):
    global_video_path = Path(cfg.paths.global_video)
    if global_video_path.exists():
        Log.info(f"[Render Global] Video already exists at {global_video_path}")
        return

    debug_cam = False
    pred = torch.load(cfg.paths.hmr4d_results)
    smplx = make_smplx("supermotion").cuda()
    smplx2smpl = torch.load("inputs/checkpoints/body_models/smplx2smpl_sparse.pt").cuda()
    faces_smpl = make_smplx("smpl").faces  # Usar faces do SMPL (compatível com vértices convertidos)
    J_regressor = torch.load(
        "inputs/checkpoints/body_models/smpl_neutral_J_regressor.pt"
    ).cuda()

    # smpl
    smplx_out = smplx(**to_cuda(pred["smpl_params_global"]))
    pred_ay_verts = torch.stack(
        [torch.matmul(smplx2smpl, v_) for v_ in smplx_out.vertices]
    )

    def move_to_start_point_face_z(verts):
        "XZ to origin, Start from the ground, Face-Z"
        # position
        verts = verts.clone()  # (L, V, 3)
        offset = einsum(J_regressor, verts[0], "j v, v i -> j i")[0]  # (3)
        offset[1] = verts[:, :, [1]].min()
        verts = verts - offset
        # face direction
        T_ay2ayfz = compute_T_ayfz2ay(
            einsum(J_regressor, verts[[0]], "j v, l v i -> l j i"), inverse=True
        )
        verts = apply_T_on_points(verts, T_ay2ayfz)
        return verts, T_ay2ayfz

    verts_glob, T_ay2ayfz = move_to_start_point_face_z(pred_ay_verts)
    joints_glob = einsum(J_regressor, verts_glob, "j v, l v i -> l j i")  # (L, J, 3)
    # global_R, global_T, global_lights = get_global_cameras_static(
    #     verts_glob.cpu(),
    #     beta=2.0,
    #     cam_height_degree=20,
    #     target_center_height=1.0,
    # )
    global_R, global_T, global_lights = get_global_cameras(
        verts_glob.cpu(),
    )
    # pred_T_w2c = pred["net_outputs"]["pred_T_w2c"].to(T_ay2ayfz)
    # T_w2c = (T_ay2ayfz @ pred_T_w2c.inverse()).inverse()

    # # pred_T_c2w = pred_T_w2c.inverse()
    # global_R = T_w2c[:, :3, :3]
    # global_T = T_w2c[:, :3, 3]

    # -- rendering code -- #
    # Determine which video to use for rendering
    video_path = None
    if hasattr(cfg, 'text1_video_path') and cfg.text1_video_path and Path(cfg.text1_video_path).exists():
        video_path = cfg.text1_video_path
    elif hasattr(cfg, 'video1_path') and cfg.video1_path and Path(cfg.video1_path).exists():
        video_path = cfg.video1_path
    elif hasattr(cfg, 'video_path') and cfg.video_path and Path(cfg.video_path).exists():
        video_path = cfg.video_path
    
    if video_path and Path(video_path).exists():
        length, width, height = get_video_lwh(video_path)
    else:
        # If there's no video, use default dimensions and length from pred
        length = len(pred["smpl_params_global"]["transl"])
        width, height = 1280, 720
    
    _, _, K = create_camera_sensor(width, height, 24)  # render as 24mm lens

    # renderer
    renderer = Renderer(width, height, device="cuda", faces=faces_smpl, K=K, bin_size=0)
    # renderer = Renderer(width, height, device="cuda", faces=faces_smpl, K=K, bin_size=0)

    # -- render mesh -- #
    scale, cx, cz = get_ground_params_from_points(joints_glob[:, 0], verts_glob)
    renderer.set_ground(scale * 1.5, cx, cz)
    color = torch.ones(3).float().cuda() * 0.8

    # Garantir que não tentamos renderizar mais frames do que existem no pred
    pred_length = len(verts_glob)
    render_length = min(length, pred_length) if not debug_cam else min(8, pred_length)
    if render_length > pred_length:
        Log.info(f"[Render Global] Warning: Video has {length} frames but pred has only {pred_length}. Rendering {pred_length} frames.")
        render_length = pred_length
    
    # Acumular frames e escrever todos de uma vez (mais confiável que write_frame)
    frames_list = []
    for i in tqdm(range(render_length), desc=f"Rendering Global"):
        cameras = renderer.create_camera(global_R[i], global_T[i])
        img = renderer.render_with_ground(
            verts_glob[[i]], color[None], cameras, global_lights
        )
        frames_list.append(img)
    
    # Escrever todos os frames de uma vez
    frames_array = np.array(frames_list).astype(np.uint8)
    save_video(frames_array, global_video_path, fps=30, crf=CRF)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="demo")
def main(cfg):
    # Parse args with proper Hydra override support
    if cfg.text1_file is not None:
        text_file = open(cfg.text1_file, "r")
        cfg.text1 = text_file.read().strip()

    # Detectar tipo de input
    has_video = has_video_input(cfg)
    has_text = has_text_input(cfg)
    
    if not has_video and not has_text:
        raise ValueError("Must provide at least text (text1) or video (video1_path)")
    
    # Determine operation mode
    if has_video and has_text:
        mode = "both"
        Log.info("[Mode] Text + Video")
    elif has_video:
        mode = "video"
        Log.info("[Mode] Video Only")
    else:
        mode = "text"
        Log.info("[Mode] Text Only")
    
    # Configurar output_dir baseado no modo
    if mode == "text":
        cfg.text1_video_name = cfg.text1.replace(" ", "_").replace(".", "")
        cfg.text1_video_path = os.path.join(cfg.output_dir, cfg.text1.replace(" ", "_").replace(".", "") + ".mp4")
    elif mode == "video":
        video_name = getattr(cfg, 'video1_name', None) or Path(cfg.video1_path).stem
        if not hasattr(cfg, 'text1_video_name') or cfg.text1_video_name is None:
            cfg.text1_video_name = video_name
        if not hasattr(cfg, 'text1_video_path') or cfg.text1_video_path is None:
            cfg.text1_video_path = os.path.join(cfg.output_dir, video_name + ".mp4")
    else:  # both
        if has_text:
            cfg.text1_video_name = cfg.text1.replace(" ", "_").replace(".", "")
        else:
            video_name = getattr(cfg, 'video1_name', None) or Path(cfg.video1_path).stem
            cfg.text1_video_name = video_name
        cfg.text1_video_path = os.path.join(cfg.output_dir, cfg.text1_video_name + ".mp4")

    # Output
    Log.info(f"[Output Dir]: {cfg.output_dir}")
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    paths = cfg.paths
    Log.info(f"[GPU]: {torch.cuda.get_device_name()}")
    Log.info(f"[GPU]: {torch.cuda.get_device_properties('cuda')}")

    # ===== Preprocess and save to disk ===== #
    data_list = []
    
    # Process text if available
    if has_text:
        Log.info("[Preprocess] Processing text...")
        data_text = run_preprocess_text(cfg)
        length = cfg.text_length
        width, height = 1280, 720

        # generate text video
        text_video_path = Path(cfg.text1_video_path)
        if not text_video_path.exists() or True:
            Log.info("[Generate Text Video]")
            create_text_video(
                str(text_video_path),
                cfg.text1,
                fps=30,
                num_frames=cfg.text_length,
                width=width,
                height=height,
                font_size=int(min(width, height) * 0.1),
            )
        
        data_text["meta"] = [
            {
                "vid": "text1",
                "caption": cfg.text1,
            }
        ]
        data_list.append(data_text)
    
    # Process video if available
    if has_video:
        Log.info("[Preprocess] Processing video...")
        # Ensure video was processed
        run_preprocess(cfg, vid=1)
        # Load processed data
        data_video = load_video_data(cfg, vid=1)
        data_list.append(data_video)
    
    # Combine data
    if len(data_list) == 1:
        data = data_list[0]
    else:
        # Combine multiple inputs (text + video)
        # Identify which is video and which is text based on insertion order
        # data_list[0] = text (if has_text)
        # data_list[1] = video (if has_video)
        data_text = data_list[0] if has_text else None
        data_video = data_list[1] if has_video and len(data_list) > 1 else (data_list[0] if has_video else None)
        
        # Use video data as base (has more visual information)
        if data_video is not None:
            data = copy.deepcopy(data_video)
            
            # If there's text, add as conditioning
            if data_text is not None:
                # Add text to video caption
                caption_text = data_text.get("caption", "")
                data["caption"] = str(caption_text) if caption_text is not None else ""
                if "meta" in data and "meta" in data_text and len(data["meta"]) > 0 and len(data_text["meta"]) > 0:
                    data["meta"][0]["caption"] = data_text["meta"][0].get("caption", data["caption"])
                data["has_text"] = torch.tensor([True])
                Log.info("[Data] Combining video + text: using video data with text as conditioning")
            else:
                Log.info("[Data] Using video data")
        else:
            # Fallback: use first input
            data = data_list[0]
            Log.info("[Data] Using first available input")

    debug = False
    if debug:
        data = data_text
        data["meta"] = [
            {
                "vid1": cfg.video1_name,
                "caption": cfg.text1,
                "eval_gen_only": True,
                # "multi_text_data": multi_text_data,
            }
        ]
    # ===== HMR4D ===== #
    if not Path(paths.hmr4d_results).exists():
        Log.info("[GENMO] Predicting")
        model = hydra.utils.instantiate(cfg.model, _recursive_=False)

        # Se ckpt_path está especificado diretamente, usar ele
        if cfg.get("ckpt_path") is not None and cfg.ckpt_path:
            ckpt_path = cfg.ckpt_path
            Log.info(f"[GENMO] Using checkpoint from: {ckpt_path}")
        else:
            test_cp = cfg.get("test_checkpoint", "last")
            if cfg.version is None:
                version = find_last_version(cfg.ckpt_dir)
                if version is None:
                    # If there's no local version and rsync is disabled, try to fetch from remote only to get the version
                    if cfg.get("rsync_ckpt", False):
                        remote_ckpt_dir = os.path.join(cfg.remote_results_path, cfg.data_name, cfg.exp_name)
                        version = find_last_version(remote_ckpt_dir, cp=test_cp)
                        ckpt_path = os.path.join("outputs", cfg.data_name, cfg.exp_name, f"version_{version}", "checkpoints", "last.ckpt")
                        print(f"rsyncing from remote {remote_ckpt_dir} to {ckpt_path}")
                        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                        rsync_file_from_remote(
                            ckpt_path,
                            remote_ckpt_dir,
                            cfg.ckpt_dir,
                            hostname="cs-oci-ord-dc-03",
                        )
                    else:
                        # Se rsync está desabilitado e não há checkpoint local, lançar erro
                        raise FileNotFoundError(
                            f"No checkpoint found locally in {cfg.ckpt_dir} and rsync_ckpt is disabled. "
                            f"Please either: 1) Enable rsync_ckpt=true, 2) Ensure checkpoints exist locally, "
                            f"or 3) Specify ckpt_path directly."
                        )
                else:
                    ckpt_path = os.path.join(cfg.ckpt_dir, f"version_{version}", "checkpoints", "last.ckpt")
            else:
                # Se cfg.version está definido, usar diretamente
                ckpt_path = os.path.join(cfg.ckpt_dir, f"version_{cfg.version}", "checkpoints", "last.ckpt")

        model.load_pretrained_model(ckpt_path)
        model = model.eval().cuda()
        tic = Log.sync_time()
        pred = model.predict(data, static_cam=False, postproc=True)
        pred = detach_to_cpu(pred)
        data_time = data["length"] / 30
        Log.info(
            f"[GENMO] Elapsed: {Log.sync_time() - tic:.2f}s for data-length={data_time:.1f}s"
        )
        torch.save(pred, paths.hmr4d_results)

    # ===== Render ===== #
    render_global(cfg)
    
    # Determine which video to use for merge
    if not Path(paths.incam_global_horiz_video).exists():
        Log.info("[Merge Videos]")
        video_for_merge = None
        if has_video and Path(cfg.video1_path).exists():
            video_for_merge = cfg.video1_path
        elif has_text and Path(cfg.text1_video_path).exists():
            video_for_merge = cfg.text1_video_path
        
        if video_for_merge and Path(video_for_merge).exists():
            merge_videos_horizontal(
                [video_for_merge, paths.global_video], paths.incam_global_horiz_video
            )
        else:
            Log.warn("[Merge Videos] Input video not found, skipping merge")


if __name__ == "__main__":
    normalize_pose_flag_for_hydra()
    main()
