#!/usr/bin/env python3
"""
CopyCat — Pipeline completa: Vídeo → SMPL (GENMO) → Movimento de Robô (GMR)

Uso mínimo:
    python run_pipeline.py --video /caminho/video.mp4 --robot booster_t1

Modo sandwich (padrão): gera prefixo "stand still" + núcleo (vídeo) + sufixo "stand still"
antes de passar para o GMR, garantindo transição estável para o T1.

Processar uma pasta inteira de vídeos:
    python run_pipeline.py --video /caminho/para/pasta/ --robot unitree_g1
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Caminhos relativos à raiz do CopyCat
# ---------------------------------------------------------------------------
COPYCAT_DIR = Path(__file__).resolve().parent
GENMO_DIR   = COPYCAT_DIR / "GENMO"
GMR_DIR     = COPYCAT_DIR / "GMR"

GENMO_SCRIPT      = GENMO_DIR / "scripts" / "demo" / "demo_text.py"
GMR_SCRIPT        = GMR_DIR / "scripts" / "gvhmr_to_robot.py"
GMR_PKL_TO_CSV    = GMR_DIR / "scripts" / "batch_gmr_pkl_to_csv.py"

# Pastas de saída do GMR
GMR_OUTPUT_PKL = GMR_DIR / "output" / "pkl"
GMR_OUTPUT_CSV = GMR_DIR / "output" / "csv"
GMR_VIDEOS_DIR = GMR_DIR / "videos"

# Checkpoint padrão do GENMO
DEFAULT_CKPT = GENMO_DIR / "inputs" / "checkpoints" / "s050000.ckpt"

# Parâmetros de âncora (sandwich)
ANCHOR_FPS    = 30          # FPS fixo — compatibilidade IsaacLab / controlador T1
ANCHOR_FRAMES = 30          # prefixo: 1 s × 30 fps
SUFFIX_FRAMES = 60          # sufixo:  2 s × 30 fps

# Pose corporal neutra usada nas âncoras de stand still.
# SMPL-X body_pose: 21 juntas × 3 (axis-angle) = 63 valores.
# Joint map (body_pose, excluindo root/pelvis):
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
    Recebe um axis-angle (3,) de global_orient e retorna outro axis-angle
    com apenas o componente de yaw (rotação em torno do eixo Y) preservado.
    Pitch e roll são zerados → robô upright na mesma direção do vídeo.
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

    # Extrair yaw: projetar o vetor "forward" (coluna Z de R) no plano XZ
    fwd = R[:, 2]
    yaw = torch.atan2(fwd[0], fwd[2])

    # Construir matriz de rotação pura em Y (yaw only)
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
    Cria um tensor (63,) com pose ereta e braços ao longo do corpo.
    Ajuste os valores abaixo com --neutral_pose_path se necessário.
    """
    import torch as _t
    bp = _t.zeros(NEUTRAL_BODY_POSE_DIM)
    # Collars: leve rotação interna para posicionar os ombros naturalmente
    bp[36:39] = _t.tensor([0.0,  0.0, -0.4])   # left_collar
    bp[39:42] = _t.tensor([0.0,  0.0,  0.4])   # right_collar
    # Ombros: ~70° para trazer os braços da horizontal para ao lado do corpo
    bp[45:48] = _t.tensor([0.0,  0.0, -0.8])   # left_shoulder
    bp[48:51] = _t.tensor([0.0,  0.0,  0.8])   # right_shoulder
    return bp


# ---------------------------------------------------------------------------
# Python executável de cada ferramenta
# ---------------------------------------------------------------------------
def _find_python(venv_dirs: list[Path]) -> str:
    """Retorna o primeiro python encontrado dentre os venvs candidatos."""
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
        print(f"[AVISO] Nenhum arquivo .mp4 encontrado em: {folder}")
    return videos


def _run_genmo_cmd(cmd: list, label: str, output_pt: Path):
    """Executa um comando GENMO e valida que hmr4d_results.pt foi gerado."""
    print("\n" + "=" * 60)
    print(f"[GENMO] {label}")
    print("=" * 60)
    print("Comando:", " ".join(cmd))

    result = subprocess.run(cmd, cwd=str(GENMO_DIR))
    if result.returncode != 0:
        print(f"[ERRO] GENMO falhou: {label}")
        sys.exit(result.returncode)

    if not output_pt.exists():
        print(f"[ERRO] hmr4d_results.pt não encontrado em: {output_pt}")
        sys.exit(1)

    return output_pt


def run_genmo(video_path: Path, video_name: str, output_dir: Path, args) -> Path:
    """
    Executa o GENMO em modo vídeo e retorna o caminho para hmr4d_results.pt.
    """
    hmr4d_results = output_dir / "hmr4d_results.pt"

    if hmr4d_results.exists() and not args.force:
        print(f"[GENMO] Pulando vídeo — resultado já existe: {hmr4d_results}")
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

    return _run_genmo_cmd(cmd, f"Processando vídeo: {video_path.name}", hmr4d_results)


def smooth_transitions(
    combined_pt: Path,
    t1: int,
    t2: int,
    n_frames: int,
    neutral_body_pose,  # torch.Tensor (63,) — zeros ou pose customizada
) -> None:
    """
    Constrói âncoras de stand still antes e depois do núcleo de movimento.

    PREFIXO [0 : t1]
      · transl        = core[0].transl        — robô na posição inicial
      · global_orient = core[0].global_orient — mesma orientação do início
      · body_pose     = neutro (zeros/custom), lerp → core[0] nos últimos n_frames
      · betas         = core[0].betas

    SUFIXO [t2 : fim]
      · transl        = core[-1].transl        — sem teleporte
      · global_orient = core[-1].global_orient — mesma orientação do fim
      · body_pose     = lerp core[-1] → neutro nos primeiros n_frames, depois neutro
      · betas         = core[-1].betas
    """
    import torch

    pred = torch.load(combined_pt, map_location="cpu")
    nbp  = neutral_body_pose  # alias curto

    for section in ("smpl_params_global", "smpl_params_incam"):
        params = pred[section]

        for key in params:
            x        = params[key].float()
            T        = x.shape[0]
            n_suffix = T - t2
            c0       = x[t1].clone()       # primeiro frame do núcleo
            cN       = x[t2 - 1].clone()   # último  frame do núcleo

            # Valor neutro para este campo
            if key == "body_pose":
                neutral_pre = nbp.to(x.dtype)
                neutral_suf = nbp.to(x.dtype)
            elif key == "global_orient":
                # Yaw do core preservado; pitch/roll zerados → upright na direção certa
                neutral_pre = _extract_yaw_orient(c0).to(x.dtype)
                neutral_suf = _extract_yaw_orient(cN).to(x.dtype)
            else:
                neutral_pre = neutral_suf = None  # transl, betas

            # ── PREFIXO ────────────────────────────────────────────────────
            if neutral_pre is not None:
                x[:t1] = neutral_pre.unsqueeze(0).expand(t1, *x.shape[1:])
                lerp_start = max(0, t1 - n_frames)
                n_lerp = t1 - lerp_start
                for i in range(n_lerp):
                    w = (i + 1) / (n_lerp + 1)
                    x[lerp_start + i] = neutral_pre * (1.0 - w) + c0 * w
            else:
                x[:t1] = c0.unsqueeze(0).expand(t1, *x.shape[1:])

            # ── SUFIXO ─────────────────────────────────────────────────────
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
        f"[Smooth] Prefixo: orient/transl=core[0], body_pose neutro + lerp {n_frames}f | "
        f"Sufixo: orient/transl=core[-1], lerp → neutro {n_frames}f"
    )


def process_video_sandwich(video_path: Path, video_name: str, output_dir: Path, args) -> Path:
    """
    Executa o pipeline sandwich:
        Prefixo (stand still) + Núcleo (vídeo) + Sufixo (stand still)

    Os frames de âncora são construídos diretamente repetindo o primeiro/último
    frame do núcleo. smooth_transitions() sobrescreve esses frames com a pose
    neutra (_make_standing_body_pose) e aplica lerp nas bordas.

    Retorna o caminho para hmr4d_results.pt combinado.
    """
    import torch as _torch

    prefix_frames = args.anchor_frames
    suffix_frames = args.suffix_frames
    core_dir      = output_dir / "sandwich_core"
    combined_pt   = output_dir / "hmr4d_results.pt"

    print(f"\n{'#' * 60}")
    print(f"# [Sandwich] Prefixo stand still : {prefix_frames} frames @ {ANCHOR_FPS} fps ({prefix_frames / ANCHOR_FPS:.1f} s)")
    print(f"# [Sandwich] Núcleo              : {video_path.name}")
    print(f"# [Sandwich] Sufixo stand still  : {suffix_frames} frames @ {ANCHOR_FPS} fps ({suffix_frames / ANCHOR_FPS:.1f} s)")
    print(f"{'#' * 60}")

    if combined_pt.exists() and not args.force:
        print(f"[Sandwich] Resultado já existe: {combined_pt}")
        return combined_pt

    # Etapa 1 — Núcleo (vídeo → SMPL)
    core_pt = run_genmo(video_path, video_name, core_dir, args)
    core    = _torch.load(core_pt, map_location="cpu")
    core_frames = core["smpl_params_global"]["transl"].shape[0]

    # Etapa 2 — Montar combined repetindo bordas do núcleo como placeholder de âncora.
    # smooth_transitions() sobrescreve esses frames com a pose neutra + lerp.
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
    print(f"[Sandwich] SMPL montado: {prefix_frames + core_frames + suffix_frames} frames → {combined_pt}")

    # Etapa 3 — Pose neutra e suavização das transições
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
    Executa o GMR (gvhmr_to_robot.py) a partir do hmr4d_results.pt.
    O PKL é salvo em GMR/output/pkl/<robot>_<video_name>.pkl.
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
            print("[AVISO] xvfb-run não encontrado. Instale com: sudo apt-get install -y xvfb")

    print("\n" + "=" * 60)
    print(f"[GMR] Retargeting para {args.robot}: {video_name}")
    if args.headless:
        print("[GMR] Headless (xvfb-run + MUJOCO_GL=egl)" if shutil.which("xvfb-run") else "[GMR] Headless parcial (MUJOCO_GL=egl apenas)")
    if args.record_video:
        print(f"[GMR] Vídeo → {video_save_path}")
    print("=" * 60)

    result = subprocess.run(cmd, cwd=str(GMR_DIR), env=env)
    if result.returncode != 0:
        print(f"[ERRO] GMR falhou para: {video_name}")
        sys.exit(result.returncode)

    print(f"[GMR] PKL salvo em: {save_path}")
    return save_path


def run_pkl_to_csv(pkl_path: Path, video_name: str, robot: str):
    """
    Converte um único arquivo .pkl do GMR em CSV e salva em GMR/output/csv/.
    """
    GMR_OUTPUT_CSV.mkdir(parents=True, exist_ok=True)
    csv_path = GMR_OUTPUT_CSV / f"{robot}_{video_name}.csv"

    print("\n" + "=" * 60)
    print(f"[CSV] Convertendo PKL → CSV: {pkl_path.name}")
    print(f"[CSV] Destino: {csv_path}")
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
            print(f"[ERRO] Conversão PKL→CSV falhou para: {pkl_path.name}")
            return None

        generated_csv = Path(tmp_dir) / "csv" / pkl_path.with_suffix(".csv").name
        if generated_csv.exists():
            shutil.move(str(generated_csv), str(csv_path))
            print(f"[CSV] Salvo em: {csv_path}")
        else:
            print(f"[AVISO] CSV gerado não encontrado: {generated_csv}")
            return None

    return csv_path


def process_video(video_path: Path, args, video_name: str = None):
    """Roda a pipeline completa (GENMO + GMR) para um único vídeo."""
    video_name = video_name or video_path.stem

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = GENMO_DIR / "outputs" / "demo" / video_name

    print(f"\n{'#' * 60}")
    print(f"# Vídeo  : {video_path.name}")
    if video_name != video_path.stem:
        print(f"# Nome   : {video_name}")
    print(f"# Saída  : {output_dir}")
    print(f"# Robô   : {args.robot}")
    print(f"# Sandwich: {'ativo' if args.sandwich else 'desativado'}")
    print(f"{'#' * 60}")

    # Etapa 1 — GENMO: vídeo → SMPL (com ou sem sandwich)
    if args.sandwich:
        hmr4d_results = process_video_sandwich(video_path, video_name, output_dir, args)
    else:
        hmr4d_results = run_genmo(video_path, video_name, output_dir, args)

    # Etapa 2 — GMR: SMPL → PKL do movimento do robô
    save_path = run_gmr(hmr4d_results, video_name, args)

    # Etapa 3 — Converter PKL → CSV
    run_pkl_to_csv(save_path, video_name, args.robot)

    return hmr4d_results, save_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CopyCat — Pipeline completa: Vídeo → SMPL (GENMO) → Robô (GMR)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Entrada ──────────────────────────────────────────────────────────────
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--video", "-v",
        help="Arquivo .mp4 único para processar.",
    )
    input_group.add_argument(
        "--videos_path",
        help="Pasta com vídeos .mp4 (busca recursiva em subpastas). "
             "Ex: --videos_path fut_do_t1/",
    )

    # ── GENMO ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--video_name",
        default=None,
        help="Nome usado na pasta de saída (padrão: stem do arquivo de vídeo). "
             "Ignorado quando --video é uma pasta.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Pasta de saída para os resultados do GENMO "
             "(padrão: GENMO/outputs/demo/<video_name>).",
    )
    parser.add_argument(
        "--ckpt_path",
        default=str(DEFAULT_CKPT),
        help=f"Checkpoint do modelo GENMO (padrão: {DEFAULT_CKPT}).",
    )
    parser.add_argument(
        "--exp",
        default="genmo_lg",
        help="Configuração de experimento do GENMO (padrão: genmo_lg).",
    )
    parser.add_argument(
        "--orig_fps",
        type=int,
        default=30,
        help="FPS original do vídeo de entrada (padrão: 30).",
    )
    parser.add_argument(
        "--static_cam",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Assume câmera estática (padrão: ativo). Use --no-static-cam para SLAM.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Força reprocessamento mesmo que hmr4d_results.pt já exista.",
    )
    parser.add_argument(
        "--pose",
        action="store_true",
        default=False,
        help="Gera PNGs de debug de pose/YOLO no GENMO (desligado por padrão).",
    )

    # ── Sandwich (Mixed Conditions) ───────────────────────────────────────────
    parser.add_argument(
        "--sandwich",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ativa o modo sandwich: prefixo 'stand still' + vídeo + sufixo 'stand still'. "
             "Garante transição estável para o T1 (padrão: ativo). "
             "Use --no-sandwich para desativar.",
    )
    parser.add_argument(
        "--anchor_frames",
        type=int,
        default=ANCHOR_FRAMES,
        help=f"Frames de stand still no prefixo (padrão: {ANCHOR_FRAMES} = {ANCHOR_FRAMES / ANCHOR_FPS:.1f} s @ {ANCHOR_FPS} fps).",
    )
    parser.add_argument(
        "--suffix_frames",
        type=int,
        default=SUFFIX_FRAMES,
        help=f"Frames de stand still no sufixo (padrão: {SUFFIX_FRAMES} = {SUFFIX_FRAMES / ANCHOR_FPS:.1f} s @ {ANCHOR_FPS} fps).",
    )
    parser.add_argument(
        "--neutral_pose_path",
        default=None,
        help="Arquivo .npy ou .pt com tensor (63,) de body_pose neutra para as âncoras "
             "(padrão: zeros = SMPL rest pose). Útil para ajustar posição dos braços.",
    )
    parser.add_argument(
        "--transition_frames",
        type=int,
        default=15,
        help="Frames de suavização em cada borda de transição do sandwich "
             "(padrão: 15 = 0.5 s a 30 fps). Usa lerp nos parâmetros SMPL.",
    )

    # ── GMR ──────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--robot", "-r",
        choices=SUPPORTED_ROBOTS,
        default="unitree_g1",
        help="Robô-alvo para o retargeting (padrão: unitree_g1).",
    )
    parser.add_argument(
        "--save_path",
        default=None,
        help="Caminho para salvar o movimento do robô (.pkl). "
             "Padrão: GMR/output/pkl/<robot>_<video_name>.pkl",
    )
    parser.add_argument(
        "--record_video",
        action="store_true",
        default=False,
        help="Grava um vídeo da visualização do GMR.",
    )
    parser.add_argument(
        "--rate_limit",
        action="store_true",
        default=False,
        help="Limita a taxa de reprodução ao FPS do movimento humano.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        default=False,
        help="Repete o movimento no viewer indefinidamente.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Roda o GMR/MuJoCo em modo headless (sem abrir janela). "
             "Define MUJOCO_GL=egl. Útil para SSH ou servidores sem display.",
    )

    args = parser.parse_args()

    # ── Validações básicas ────────────────────────────────────────────────────
    if not GENMO_SCRIPT.exists():
        print(f"[ERRO] Script GENMO não encontrado: {GENMO_SCRIPT}")
        sys.exit(1)
    if not GMR_SCRIPT.exists():
        print(f"[ERRO] Script GMR não encontrado: {GMR_SCRIPT}")
        sys.exit(1)

    print(f"[Config] GENMO Python : {GENMO_PYTHON}")
    print(f"[Config] GMR   Python : {GMR_PYTHON}")
    if args.sandwich:
        print(f"[Config] Sandwich     : ativo")
        print(f"[Config] Prefixo      : {args.anchor_frames} frames ({args.anchor_frames / ANCHOR_FPS:.1f} s @ {ANCHOR_FPS} fps)")
        print(f"[Config] Sufixo       : {args.suffix_frames} frames ({args.suffix_frames / ANCHOR_FPS:.1f} s @ {ANCHOR_FPS} fps)")

    # ── Modo pasta (--videos_path) ────────────────────────────────────────────
    if args.videos_path:
        folder = Path(args.videos_path).resolve()
        if not folder.is_dir():
            print(f"[ERRO] Pasta não encontrada: {folder}")
            sys.exit(1)
        videos = find_videos_in_folder(folder)
        if not videos:
            sys.exit(1)
        print(f"[INFO] {len(videos)} vídeo(s) encontrado(s) em: {folder}")

        results = []
        for i, vp in enumerate(videos, 1):
            print(f"\n[{i}/{len(videos)}] {vp.name}")
            try:
                _, save = process_video(vp, args)
                results.append((vp, save, None))
            except SystemExit as e:
                print(f"[AVISO] Pulando {vp.name} (erro {e.code})")
                results.append((vp, None, e.code))

        print("\n" + "=" * 60)
        print(f"[CONCLUÍDO] {len(videos)} vídeo(s) processado(s).")
        for vp, save, err in results:
            print(f"  {vp.name} → {str(save) if save else f'ERRO ({err})'}")
        return

    # ── Modo arquivo único (--video) ──────────────────────────────────────────
    video_input = Path(args.video).resolve()
    if not video_input.exists():
        print(f"[ERRO] Caminho não encontrado: {video_input}")
        sys.exit(1)
    if video_input.suffix.lower() != ".mp4":
        print(f"[AVISO] O arquivo não é .mp4: {video_input}")

    process_video(video_input, args, video_name=args.video_name)

    print("\n[CONCLUÍDO] Pipeline finalizada.")


if __name__ == "__main__":
    main()
