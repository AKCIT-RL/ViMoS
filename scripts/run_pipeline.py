#!/usr/bin/env python3
"""
CopyCat — Pipeline completa: Vídeo → SMPL (GENMO) → Movimento de Robô (GMR)

Uso mínimo:
    python run_pipeline.py --video /caminho/video.mp4 --robot booster_t1

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

GENMO_SCRIPT = GENMO_DIR / "scripts" / "demo" / "demo_text.py"
GMR_SCRIPT        = GMR_DIR / "scripts" / "gvhmr_to_robot.py"
GMR_PKL_TO_CSV    = GMR_DIR / "scripts" / "batch_gmr_pkl_to_csv.py"

# Pastas de saída do GMR
GMR_OUTPUT_PKL = GMR_DIR / "output" / "pkl"
GMR_OUTPUT_CSV = GMR_DIR / "output" / "csv"
GMR_VIDEOS_DIR = GMR_DIR / "videos"

# Checkpoint padrão do GENMO
DEFAULT_CKPT = GENMO_DIR / "inputs" / "checkpoints" / "s050000.ckpt"

# ---------------------------------------------------------------------------
# Python executável de cada ferramenta
# Cada sub-script roda com o venv do seu próprio projeto, igual a quando
# você os executava manualmente.
# ---------------------------------------------------------------------------
def _find_python(venv_dirs: list[Path]) -> str:
    """Retorna o primeiro python encontrado dentre os venvs candidatos."""
    for venv in venv_dirs:
        candidate = venv / "bin" / "python"
        if candidate.exists():
            return str(candidate)
    # fallback: Python atual
    return sys.executable

# Busca o Python correto para cada sub-projeto.
# GENMO → CopyCat/GENMO/.venv
# GMR   → CopyCat/GMR/.venv
GENMO_PYTHON = _find_python([
    GENMO_DIR / ".venv",                        # CopyCat/GENMO/.venv
    COPYCAT_DIR.parent / "GENMO" / ".venv",    # motion/GENMO/.venv  (legado)
])
GMR_PYTHON = _find_python([
    GMR_DIR / ".venv",                          # CopyCat/GMR/.venv   ← atual
    COPYCAT_DIR / ".venv",                      # CopyCat/.venv       (legado)
    COPYCAT_DIR.parent / "GMR" / ".venv",
])

print(f"[Config] GENMO Python : {GENMO_PYTHON}")
print(f"[Config] GMR   Python : {GMR_PYTHON}")

# Robôs suportados pelo GMR
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
    """Busca todos os .mp4 dentro de uma pasta e subpastas (recursivo)."""
    videos = sorted(folder.rglob("*.mp4"))
    if not videos:
        print(f"[AVISO] Nenhum arquivo .mp4 encontrado em: {folder}")
    return videos


def run_genmo(video_path: Path, video_name: str, output_dir: Path, args) -> Path:
    """
    Executa o GENMO (demo_text.py) e retorna o caminho para hmr4d_results.pt.
    """
    hmr4d_results = output_dir / "hmr4d_results.pt"

    if hmr4d_results.exists() and not args.force:
        print(f"[GENMO] Pulando — resultado já existe: {hmr4d_results}")
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

    print("\n" + "=" * 60)
    print(f"[GENMO] Processando: {video_path.name}")
    print("=" * 60)
    print("Comando:", " ".join(cmd))

    result = subprocess.run(cmd, cwd=str(GENMO_DIR))
    if result.returncode != 0:
        print(f"[ERRO] GENMO falhou para: {video_path.name}")
        sys.exit(result.returncode)

    if not hmr4d_results.exists():
        print(f"[ERRO] hmr4d_results.pt não encontrado em: {hmr4d_results}")
        sys.exit(1)

    return hmr4d_results


def run_gmr(hmr4d_results: Path, video_name: str, args):
    """
    Executa o GMR (gvhmr_to_robot.py) a partir do hmr4d_results.pt gerado pelo GENMO.
    O PKL é salvo em GMR/output/pkl/<video_name>.pkl.
    """
    if args.save_path:
        save_path = Path(args.save_path)
    else:
        # Salva em GMR/output/pkl/ com nome {robot}_{video_name}.pkl
        GMR_OUTPUT_PKL.mkdir(parents=True, exist_ok=True)
        save_path = GMR_OUTPUT_PKL / f"{args.robot}_{video_name}.pkl"

    # Vídeo de visualização: GMR/videos/<robot>_<video_name>.mp4
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
        # xvfb-run fornece display virtual → launch_passive não abre janela real
        xvfb = shutil.which("xvfb-run")
        if xvfb:
            cmd = [xvfb, "-a", "--server-args=-screen 0 1024x768x24"] + cmd
        else:
            print("[AVISO] xvfb-run não encontrado. Instale com: sudo apt-get install -y xvfb")
            print("[AVISO] O MuJoCo pode abrir uma janela mesmo assim.")

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
    O CSV terá o nome {robot}_{video_name}.csv.
    """
    GMR_OUTPUT_CSV.mkdir(parents=True, exist_ok=True)
    csv_path = GMR_OUTPUT_CSV / f"{robot}_{video_name}.csv"

    print("\n" + "=" * 60)
    print(f"[CSV] Convertendo PKL → CSV: {pkl_path.name}")
    print(f"[CSV] Destino: {csv_path}")
    print("=" * 60)

    # batch_gmr_pkl_to_csv lê todos os .pkl de uma pasta e salva em <pasta>/csv/
    # Passamos a pasta do pkl e depois movemos o CSV gerado para GMR/output/csv/
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

        # O script salva em <tmp_dir>/csv/<stem>.csv
        generated_csv = Path(tmp_dir) / "csv" / pkl_path.with_suffix(".csv").name
        if generated_csv.exists():
            shutil.move(str(generated_csv), str(csv_path))
            print(f"[CSV] Salvo em: {csv_path}")
        else:
            print(f"[AVISO] CSV gerado não encontrado: {generated_csv}")
            return None

    return csv_path


def process_video(video_path: Path, args):
    """Roda a pipeline completa (GENMO + GMR) para um único vídeo."""
    video_name = video_path.stem

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = GENMO_DIR / "outputs" / "demo" / video_name

    print(f"\n{'#' * 60}")
    print(f"# Vídeo : {video_path.name}")
    print(f"# Saída : {output_dir}")
    print(f"# Robô  : {args.robot}")
    print(f"{'#' * 60}")

    # Etapa 1 — GENMO: vídeo → SMPL
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
             "Padrão: <output_dir>/robot_motion_<robot>.pkl",
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
                hmr, save = process_video(vp, args)
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

    if args.video_name:
        _process_single(video_input, args.video_name, args)
    else:
        process_video(video_input, args)

    print("\n[CONCLUÍDO] Pipeline finalizada.")


def _process_single(video_path: Path, video_name: str, args):
    """Roda a pipeline completa para um único vídeo com nome customizado."""
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = GENMO_DIR / "outputs" / "demo" / video_name

    print(f"\n{'#' * 60}")
    print(f"# Vídeo : {video_path.name}")
    print(f"# Nome  : {video_name}")
    print(f"# Saída : {output_dir}")
    print(f"# Robô  : {args.robot}")
    print(f"{'#' * 60}")

    hmr4d_results = run_genmo(video_path, video_name, output_dir, args)
    save_path = run_gmr(hmr4d_results, video_name, args)
    run_pkl_to_csv(save_path, video_name, args.robot)
    return hmr4d_results, save_path


if __name__ == "__main__":
    main()
