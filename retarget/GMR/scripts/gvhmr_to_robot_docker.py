import argparse
import os
import pathlib

import numpy as np
import mujoco as mj

from general_motion_retargeting import (
    GeneralMotionRetargeting as GMR,
    ROBOT_XML_DICT,
    ROBOT_BASE_DICT,
    VIEWER_CAM_DISTANCE_DICT,
)
from general_motion_retargeting.utils.smpl import load_gvhmr_pred_file, get_gvhmr_data_offline_fast

from rich import print

if __name__ == "__main__":

    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gvhmr_pred_file",
        help="SMPLX motion file to load.",
        type=str,
        default="/home/yanjieze/projects/g1_wbc/GMR/GVHMR/outputs/demo/tennis/hmr4d_results.pt",
    )
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands", "unitree_h1", "unitree_h1_2",
                 "booster_t1", "booster_t1_29dof", "stanford_toddy", "fourier_n1",
                 "engineai_pm01", "kuavo_s45", "hightorque_hi", "galaxea_r1pro",
                 "berkeley_humanoid_lite", "booster_k1", "pnd_adam_lite", "openloong", "tienkung"],
        default="unitree_g1",
    )
    parser.add_argument("--save_path",       default=None, help="Path to save the robot motion (.pkl).")
    parser.add_argument("--video_save_path", default=None, help="Path to save the visualization video (.mp4).")
    parser.add_argument("--record_video",    default=False, action="store_true", help="Record visualization video (offscreen).")
    # Mantidos por compatibilidade de API — ignorados no modo headless
    parser.add_argument("--loop",       default=False, action="store_true", help="(ignorado em headless)")
    parser.add_argument("--rate_limit", default=False, action="store_true", help="(ignorado em headless)")

    args = parser.parse_args()

    SMPLX_FOLDER = HERE / ".." / "assets" / "body_models"

    # ── Carregar dados SMPL-X ─────────────────────────────────────────────────
    smplx_data, body_model, smplx_output, actual_human_height = load_gvhmr_pred_file(
        args.gvhmr_pred_file, SMPLX_FOLDER
    )

    tgt_fps = 30
    smplx_data_frames, aligned_fps = get_gvhmr_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=tgt_fps
    )

    # ── Inicializar retargeting ───────────────────────────────────────────────
    retarget = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=args.robot,
    )

    # ── Carregar modelo MuJoCo (sem viewer) ──────────────────────────────────
    xml_path = ROBOT_XML_DICT[args.robot]
    model = mj.MjModel.from_xml_path(str(xml_path))
    data  = mj.MjData(model)
    mj.mj_step(model, data)

    # ── Configurar gravação de vídeo offscreen (sem GLFW) ─────────────────────
    mp4_writer = None
    renderer   = None
    if args.record_video:
        import imageio

        _default_video = f"videos/{args.robot}_{pathlib.Path(args.gvhmr_pred_file).stem}.mp4"
        video_path = args.video_save_path if args.video_save_path else _default_video
        video_dir  = os.path.dirname(video_path)
        if video_dir:
            os.makedirs(video_dir, exist_ok=True)

        renderer   = mj.Renderer(model, height=480, width=640)
        mp4_writer = imageio.get_writer(video_path, fps=int(aligned_fps))

        # Câmera que segue o robô (mesma lógica do RobotMotionViewer)
        cam               = mj.MjvCamera()
        cam.type          = mj.mjtCamera.mjCAMERA_FREE
        cam.distance      = VIEWER_CAM_DISTANCE_DICT[args.robot]
        cam.elevation     = -10
        robot_base_id     = model.body(ROBOT_BASE_DICT[args.robot]).id

        print(f"[Video] Gravando em: {video_path}")

    # ── Preparar saída PKL ────────────────────────────────────────────────────
    qpos_list = [] if args.save_path is not None else None
    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    # ── Loop principal: retarget + render ─────────────────────────────────────
    for i in range(1, len(smplx_data_frames)):
        smplx_frame = smplx_data_frames[i]
        qpos = retarget.retarget(smplx_frame)

        if qpos_list is not None:
            qpos_list.append(qpos)

        if renderer is not None:
            data.qpos[:3]  = qpos[:3]    # posição da base
            data.qpos[3:7] = qpos[3:7]   # rotação da base (wxyz)
            data.qpos[7:]  = qpos[7:]    # ângulos das juntas
            mj.mj_forward(model, data)

            # Atualizar câmera para seguir o robô
            cam.lookat[:] = data.xpos[robot_base_id]
            renderer.update_scene(data, camera=cam)
            mp4_writer.append_data(renderer.render())

    # ── Finalizar vídeo ───────────────────────────────────────────────────────
    if mp4_writer is not None:
        mp4_writer.close()
        renderer.close()
        print(f"[Video] Salvo.")

    # ── Salvar PKL ────────────────────────────────────────────────────────────
    if qpos_list is not None:
        import pickle

        root_pos = np.array([q[:3]            for q in qpos_list])
        root_rot = np.array([q[3:7][[1,2,3,0]] for q in qpos_list])  # wxyz → xyzw
        dof_pos  = np.array([q[7:]             for q in qpos_list])

        motion_data = {
            "fps":            aligned_fps,
            "root_pos":       root_pos,
            "root_rot":       root_rot,
            "dof_pos":        dof_pos,
            "local_body_pos": None,
            "link_body_list": None,
        }
        with open(args.save_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"Saved to {args.save_path}")
