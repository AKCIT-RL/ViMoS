#!/usr/bin/env python3
"""
ViMoS — Asset & checkpoint installer.

This script places every model/checkpoint the pipeline needs into the exact
directory each stage expects. It is idempotent: already-installed files are
skipped, so it is safe to re-run.

It handles two classes of assets:

  1. Freely redistributable checkpoints (GENMO video backbone: GVHMR, ViTPose,
     HMR2). These are downloaded automatically from public HuggingFace mirrors.

  2. License-gated body models (SMPL, SMPL-X) and the GENMO motion checkpoint
     (s050000.ckpt). These CANNOT be redistributed, so you must download them
     once (registration required) and drop the archives in the repo root (or
     point to them with the flags below). This script then extracts and places
     them correctly for both GENMO and GMR.

------------------------------------------------------------------------------
What you need to download manually first (put these in the repo root):
------------------------------------------------------------------------------
  • s050000.ckpt                  GENMO motion model.
        https://drive.google.com/file/d/1b1E84G7S0h2n5o0RmrcmKOhRKukOjgsJ/view
        (a name like "s050000(1).ckpt" is auto-detected too)

  • models_smplx_v1_1.zip         SMPL-X body models (npz + pkl).
        https://smpl-x.is.tue.mpg.de/  (register, accept license)

  • SMPL_python_v.1.1.0.zip       SMPL body model.
        https://smpl.is.tue.mpg.de/    (register, accept license)

Then simply run:
    python scripts/download_assets.py

Everything else (GVHMR / ViTPose / HMR2) is fetched automatically.
"""

import argparse
import shutil
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen

REPO_ROOT = Path(__file__).resolve().parent.parent
GENMO_DIR = REPO_ROOT / "retarget" / "GENMO"
GMR_DIR = REPO_ROOT / "retarget" / "GMR"

# GENMO expects every checkpoint under inputs/checkpoints/
GENMO_CKPT_DIR = GENMO_DIR / "inputs" / "checkpoints"
GENMO_BODY_DIR = GENMO_CKPT_DIR / "body_models"
GENMO_SMPLX_DIR = GENMO_BODY_DIR / "smplx"
GENMO_SMPL_DIR = GENMO_BODY_DIR / "smpl"

# GMR expects body models under assets/body_models/
GMR_SMPLX_DIR = GMR_DIR / "assets" / "body_models" / "smplx"

# Auxiliary tensors shipped inside the GVHMR submodule
GVHMR_BODY_DIR = (
    GENMO_DIR / "third_party" / "GVHMR" / "hmr4d" / "utils" / "body_model"
)

# Freely redistributable checkpoints (public HuggingFace mirror, no auth).
# All live under inputs/checkpoints/. GVHMR reaches YOLO/HMR2 via an absolute
# PROJ_ROOT path; docker-compose mounts this same dir at both locations.
HF_BASE = "https://huggingface.co/camenduru/GVHMR/resolve/main"
FREE_CHECKPOINTS = [
    (f"{HF_BASE}/vitpose/vitpose-h-multi-coco.pth",
     GENMO_CKPT_DIR / "vitpose" / "vitpose-h-multi-coco.pth"),
    (f"{HF_BASE}/gvhmr/gvhmr_siga24_release.ckpt",
     GENMO_CKPT_DIR / "gvhmr" / "gvhmr_siga24_release.ckpt"),
    (f"{HF_BASE}/hmr2/epoch%3D10-step%3D25000.ckpt",
     GENMO_CKPT_DIR / "hmr2" / "epoch=10-step=25000.ckpt"),
    (f"{HF_BASE}/yolo/yolov8x.pt",
     GENMO_CKPT_DIR / "yolo" / "yolov8x.pt"),
]


def info(msg: str) -> None:
    print(f"[assets] {msg}")


def ok(msg: str) -> None:
    print(f"[assets] \033[92m✓\033[0m {msg}")


def warn(msg: str) -> None:
    print(f"[assets] \033[93m!\033[0m {msg}")


def _find(root: Path, *patterns: str) -> Path | None:
    """Return the first existing file in root matching any glob pattern."""
    for pattern in patterns:
        matches = sorted(root.glob(pattern))
        if matches:
            return matches[0]
    return None


def _extract_member(zf: zipfile.ZipFile, member: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(member) as src, open(dest, "wb") as out:
        shutil.copyfileobj(src, out)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    info(f"downloading {dest.name} ...")
    with urlopen(url) as resp:  # noqa: S310 (trusted HF mirror)
        total = int(resp.headers.get("Content-Length", 0))
        read = 0
        with open(tmp, "wb") as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                read += len(chunk)
                if total:
                    pct = read * 100 // total
                    print(f"\r[assets]   {dest.name}: {pct:3d}% "
                          f"({read >> 20}/{total >> 20} MiB)", end="")
        if total:
            print()
    tmp.replace(dest)


# ─────────────────────────────────────────────────────────────────────────────
# Steps
# ─────────────────────────────────────────────────────────────────────────────

def install_genmo_ckpt(args) -> None:
    dest = GENMO_CKPT_DIR / "s050000.ckpt"
    if dest.exists():
        ok(f"GENMO checkpoint already in place: {dest.relative_to(REPO_ROOT)}")
        return
    src = Path(args.genmo_ckpt) if args.genmo_ckpt else _find(
        REPO_ROOT, "s050000.ckpt", "s050000*.ckpt")
    if not src or not src.exists():
        warn("GENMO checkpoint s050000.ckpt not found. Download it from "
             "https://drive.google.com/file/d/1b1E84G7S0h2n5o0RmrcmKOhRKukOjgsJ/view "
             "and place it in the repo root, or pass --genmo-ckpt PATH.")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    info(f"copying {src.name} -> {dest.relative_to(REPO_ROOT)}")
    shutil.copy2(src, dest)
    ok("GENMO checkpoint installed.")


def install_smplx(args) -> None:
    targets = {
        "SMPLX_NEUTRAL.npz": [GENMO_SMPLX_DIR, GMR_SMPLX_DIR],
        "SMPLX_MALE.npz": [GENMO_SMPLX_DIR, GMR_SMPLX_DIR],
        "SMPLX_FEMALE.npz": [GENMO_SMPLX_DIR, GMR_SMPLX_DIR],
    }
    if all((d / name).exists() for name, dirs in targets.items() for d in dirs):
        ok("SMPL-X body models already in place (GENMO + GMR).")
        return
    zip_path = Path(args.smplx_zip) if args.smplx_zip else _find(
        REPO_ROOT, "models_smplx_v1_1.zip", "*smplx*v1_1.zip", "*smplx*.zip")
    if not zip_path or not zip_path.exists():
        warn("SMPL-X archive not found. Download models_smplx_v1_1.zip from "
             "https://smpl-x.is.tue.mpg.de/ and place it in the repo root, "
             "or pass --smplx-zip PATH.")
        return
    info(f"extracting SMPL-X from {zip_path.name}")
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        for fname, dirs in targets.items():
            member = next((n for n in names if n.endswith(fname)), None)
            if not member:
                warn(f"  {fname} not found inside archive — skipped.")
                continue
            for d in dirs:
                dest = d / fname
                if dest.exists():
                    continue
                _extract_member(zf, member, dest)
                info(f"  -> {dest.relative_to(REPO_ROOT)}")
    ok("SMPL-X body models installed.")


def install_smpl(args) -> None:
    dest = GENMO_SMPL_DIR / "SMPL_NEUTRAL.pkl"
    if dest.exists():
        ok("SMPL body model already in place.")
        return
    zip_path = Path(args.smpl_zip) if args.smpl_zip else _find(
        REPO_ROOT, "SMPL_python_v.1.1.0.zip", "SMPL_python*.zip")
    if not zip_path or not zip_path.exists():
        warn("SMPL archive not found. Download SMPL_python_v.1.1.0.zip from "
             "https://smpl.is.tue.mpg.de/ and place it in the repo root, "
             "or pass --smpl-zip PATH.")
        return
    info(f"extracting SMPL from {zip_path.name}")
    with zipfile.ZipFile(zip_path) as zf:
        member = next((n for n in zf.namelist()
                       if n.endswith("basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl")),
                      None)
        if not member:
            warn("  neutral SMPL model not found inside archive — skipped.")
            return
        _extract_member(zf, member, dest)
        info(f"  -> {dest.relative_to(REPO_ROOT)}")
    ok("SMPL body model installed.")


def install_aux_tensors() -> None:
    aux = ["smplx2smpl_sparse.pt", "smpl_neutral_J_regressor.pt"]
    if all((GENMO_BODY_DIR / f).exists() for f in aux):
        ok("Auxiliary body-model tensors already in place.")
        return
    if not GVHMR_BODY_DIR.exists():
        warn("GVHMR submodule not initialised — run "
             "'git submodule update --init --recursive' first. Skipping aux tensors.")
        return
    GENMO_BODY_DIR.mkdir(parents=True, exist_ok=True)
    for fname in aux:
        src = GVHMR_BODY_DIR / fname
        dest = GENMO_BODY_DIR / fname
        if dest.exists():
            continue
        if not src.exists():
            warn(f"  {fname} missing in GVHMR submodule — skipped.")
            continue
        shutil.copy2(src, dest)
        info(f"  -> {dest.relative_to(REPO_ROOT)}")
    ok("Auxiliary body-model tensors installed.")


def install_free_checkpoints(args) -> None:
    if args.skip_download:
        warn("Skipping HuggingFace downloads (--skip-download).")
        return
    for url, dest in FREE_CHECKPOINTS:
        if dest.exists():
            ok(f"already present: {dest.relative_to(REPO_ROOT)}")
            continue
        try:
            _download(url, dest)
            ok(f"installed {dest.relative_to(REPO_ROOT)}")
        except Exception as exc:  # noqa: BLE001
            warn(f"failed to download {dest.name}: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Install ViMoS checkpoints & body models into the right dirs.")
    parser.add_argument("--genmo-ckpt", help="Path to s050000.ckpt")
    parser.add_argument("--smplx-zip", help="Path to models_smplx_v1_1.zip")
    parser.add_argument("--smpl-zip", help="Path to SMPL_python_v.1.1.0.zip")
    parser.add_argument("--skip-download", action="store_true",
                        help="Do not download GVHMR/ViTPose/HMR2 from HuggingFace")
    args = parser.parse_args()

    if not GENMO_DIR.exists() or not GMR_DIR.exists():
        warn("Submodules not found. Run "
             "'git submodule update --init --recursive' first.")
        return 1

    info("Installing ViMoS assets ...")
    install_genmo_ckpt(args)
    install_smplx(args)
    install_smpl(args)
    install_aux_tensors()
    install_free_checkpoints(args)

    info("Done. Verify the tree under retarget/GENMO/inputs/checkpoints/ "
         "and retarget/GMR/assets/body_models/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
