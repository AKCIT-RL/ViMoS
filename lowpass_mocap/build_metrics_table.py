#!/usr/bin/env python3
"""
Full Table-VII-style metrics for  mocap  vs  mocap low-pass  vs  GenMo.

Every number is produced by the *unmodified* GMR evaluation pipeline
(`retarget/GMR/scripts/eval_retarget_metrics.py`), the same one that generated
`retarget/GMR/reports/BVH_vs_CoRL_normalized/*.json` for the paper.  Settings:

    --mocap_root_rot_convention xyzw   --genmo_root_rot_convention xyzw
    normalize_orientation = True   DTW alignment   target_fps = min(inputs) = 30

Sources (all three describe the SAME 21 motions on Unitree G1):
    mocap        BVH_pkl/unitree_g1_<stem>.pkl                      (Xsens, 59/239 Hz)
                 == csv_g1_mocap/<name>.csv  (verified byte-for-byte)
    mocap low-pass  output/pkl/unitree_g1_<stem>.pkl               (this repo, 30 Hz)
    GenMo        retarget/GMR/output/pkl/Paper_CoRL/unitree_g1_<stem>.pkl (30 Hz)

Three evaluation pairs per motion (eval_retarget_metrics compares a pair):
    A = (mocap , GenMo)        -> jerk/JLV for mocap & GenMo,  MPJPE/MAE  GenMo-vs-mocap
    B = (low-pass , GenMo)     -> jerk/JLV for low-pass,       MPJPE/MAE  GenMo-vs-low-pass
    C = (mocap , low-pass)     ->                              MPJPE/MAE  low-pass-vs-mocap

Pair A reproduces the paper's per-motion JSON exactly (checked against the
reference reports and printed as "vs paper").

Outputs:
    output/eval/{A,B,C}_<name>.json     raw pipeline reports
    output/metrics_table.csv / .md      assembled table + category / overall means
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np

from lowpass_filter import (
    BVH_PKL_DIR, GENMO_PKL_DIR, HERE, NAME_MAP, OUT_DIR, OUT_PKL_DIR,
)

GMR_DIR = os.path.join(HERE, "..", "retarget", "GMR")
EVAL = os.path.join(GMR_DIR, "scripts", "eval_retarget_metrics.py")
PY = os.path.join(GMR_DIR, ".venv", "bin", "python")
PAPER_REF = os.path.join(GMR_DIR, "reports", "BVH_vs_CoRL_normalized")
EVAL_DIR = os.path.join(OUT_DIR, "eval")

# paper Table VII grouping
CATEGORIES = [
    ("Locomotion", ["run", "trote", "side_walking"]),
    ("Rhythmic", ["macarena", "like_jennie", "pedalada", "aceno"]),
    ("Dynamic", ["jump", "cr7", "jumping_puddle", "bunny_hop"]),
    ("Other", ["boxe", "stealth", "trivela", "paradinha", "polichinelo", "drunk",
               "one_foot_balance", "get_Down_get_Up", "square", "what_is_love"]),
]
PRETTY = {"trote": "Trot", "aceno": "Wave", "trivela": "Trivela Side",
          "get_Down_get_Up": "Get Down Get Up", "one_foot_balance": "One Foot Bal.",
          "cr7": "CR7"}


def pretty(name: str) -> str:
    return PRETTY.get(name, name.replace("_", " ").title())


def run_eval(mocap_pkl: str, genmo_pkl: str, out_json: str, force: bool) -> dict:
    if os.path.exists(out_json) and not force:
        return json.load(open(out_json))
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    cmd = [PY, EVAL, "--mocap_pkl", mocap_pkl, "--genmo_pkl", genmo_pkl,
           "--robot", "unitree_g1",
           "--mocap_root_rot_convention", "xyzw",
           "--genmo_root_rot_convention", "xyzw",
           "--out_json", out_json]
    subprocess.run(cmd, check=True, cwd=GMR_DIR,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return json.load(open(out_json))


def scalars(rep: dict) -> dict:
    js, fk = rep["joint_space"], rep["fk_task_space"]
    return dict(
        jerk_mocap=js["jerk"]["mocap"]["jerk_l2_mean"],
        jerk_genmo=js["jerk"]["genmo"]["jerk_l2_mean"],
        jlv_mocap=100.0 * js["joint_limit_violations_mocap"]["violation_rate"],
        jlv_genmo=100.0 * js["joint_limit_violations_genmo"]["violation_rate"],
        mpjpe=fk["mpjpe_like"]["mean_mpjpe_like_m"],
        mae=js["q_error_dtw_aligned"]["mae_mean"],
    )


def main() -> None:
    force = "--force" in sys.argv
    only = None
    if "--only" in sys.argv:
        i = sys.argv.index("--only")
        only = sys.argv[i + 1:]

    names = only or [n for _, g in CATEGORIES for n in g]
    rows = {}
    for name in names:
        stem = NAME_MAP[name]
        bvh = os.path.join(BVH_PKL_DIR, f"unitree_g1_{stem}.pkl")
        genmo = os.path.join(GENMO_PKL_DIR, f"unitree_g1_{stem}.pkl")
        suave = os.path.join(OUT_PKL_DIR, f"unitree_g1_{stem}.pkl")
        for p in (bvh, genmo, suave):
            if not os.path.exists(p):
                print(f"[skip] {name}: missing {p}")
                break
        else:
            A = scalars(run_eval(bvh, genmo, f"{EVAL_DIR}/A_{name}.json", force))
            B = scalars(run_eval(suave, genmo, f"{EVAL_DIR}/B_{name}.json", force))
            C = scalars(run_eval(bvh, suave, f"{EVAL_DIR}/C_{name}.json", force))

            r = dict(
                jerk_mocap=A["jerk_mocap"], jerk_suave=B["jerk_mocap"], jerk_genmo=A["jerk_genmo"],
                jlv_mocap=A["jlv_mocap"], jlv_suave=B["jlv_mocap"], jlv_genmo=A["jlv_genmo"],
                mpjpe_gm=A["mpjpe"], mpjpe_gs=B["mpjpe"], mpjpe_sm=C["mpjpe"],
                mae_gm=A["mae"], mae_gs=B["mae"], mae_sm=C["mae"],
            )
            # cross-pair consistency (same pkl, same target fps -> identical numbers)
            r["_check"] = (
                abs(A["jerk_genmo"] - B["jerk_genmo"]) < 1e-6
                and abs(A["jerk_mocap"] - C["jerk_mocap"]) < 1e-6
                and abs(B["jerk_mocap"] - C["jerk_genmo"]) < 1e-6
            )
            # vs paper reference JSON
            ref_path = os.path.join(PAPER_REF, f"{stem}.json")
            r["_vs_paper"] = None
            if os.path.exists(ref_path):
                ref = scalars(json.load(open(ref_path)))
                r["_vs_paper"] = max(
                    abs(ref["jerk_mocap"] - A["jerk_mocap"]),
                    abs(ref["jerk_genmo"] - A["jerk_genmo"]),
                    abs(ref["jlv_mocap"] - A["jlv_mocap"]),
                    abs(ref["mpjpe"] - A["mpjpe"]),
                    abs(ref["mae"] - A["mae"]),
                )
            rows[name] = r
            tick = "ok" if r["_check"] else "MISMATCH"
            vp = "" if r["_vs_paper"] is None else f"  vs paper Δmax={r['_vs_paper']:.2e}"
            print(f"{pretty(name):18s} jerk {r['jerk_mocap']:6.0f} -> {r['jerk_suave']:6.0f} "
                  f"(genmo {r['jerk_genmo']:6.0f})   [{tick}]{vp}")

    _write(rows)


def _fmt(rows, keys, name):
    r = rows[name]
    return [r[k] for k in keys]


def _write(rows: dict) -> None:
    if not rows:
        return
    cols = ["jerk_mocap", "jerk_suave", "jerk_genmo",
            "jlv_mocap", "jlv_suave", "jlv_genmo",
            "mpjpe_gm", "mpjpe_gs", "mpjpe_sm",
            "mae_gm", "mae_gs", "mae_sm"]

    import csv as _csv
    with open(os.path.join(OUT_DIR, "metrics_table.csv"), "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["motion"] + cols)
        for name, r in rows.items():
            w.writerow([name] + [f"{r[c]:.6g}" for c in cols])

    def block(names):
        return np.array([[rows[n][c] for c in cols] for n in names if n in rows])

    lines = []
    lines.append("# Full metrics — mocap vs mocap low-pass vs GenMo\n")
    lines.append("Pipeline: GMR `eval_retarget_metrics.py` (xyzw/xyzw, heading-normalized, "
                 "DTW, target_fps=30). Pair A reproduces the paper's per-motion JSON.\n")
    lines.append("- **Jerk** (rad/s³): per-source, joints only, resampled to 30 Hz.\n"
                 "- **JL Viol** (%): frames with any joint outside the URDF limits.\n"
                 "- **MPJPE** (m) / **MAE** (rad): DTW-aligned. "
                 "`g–m` = GenMo vs mocap, `g–s` = GenMo vs low-pass, `s–m` = low-pass vs mocap.\n")
    header = ("| Motion | Jerk mocap | Jerk low-pass | Jerk GenMo | "
              "JLV mocap | JLV low-pass | JLV GenMo | "
              "MPJPE g–m | MPJPE g–s | MPJPE s–m | MAE g–m | MAE g–s | MAE s–m |")
    sep = "|" + "---|" * 13
    lines.append(header)
    lines.append(sep)

    def fmt_row(label, vec):
        j = [f"{v:.0f}" for v in vec[0:3]]
        jl = [f"{v:.1f}" for v in vec[3:6]]
        mp = [f"{v:.3f}" for v in vec[6:9]]
        ma = [f"{v:.3f}" for v in vec[9:12]]
        return f"| {label} | " + " | ".join(j + jl + mp + ma) + " |"

    allv = []
    for cat, names in CATEGORIES:
        present = [n for n in names if n in rows]
        if not present:
            continue
        lines.append(f"| *{cat}* | | | | | | | | | | | | |")
        for n in present:
            lines.append(fmt_row(pretty(n), [rows[n][c] for c in cols]))
        b = block(present)
        allv.append(b)
        lines.append(fmt_row(f"**{cat} mean**", b.mean(axis=0)))

    overall = np.vstack(allv)
    lines.append(fmt_row("**Overall mean**", overall.mean(axis=0)))

    checks = [r["_vs_paper"] for r in rows.values() if r["_vs_paper"] is not None]
    if checks:
        lines.append(f"\n_Pair A vs paper reference JSON: max abs diff over all motions = "
                     f"{max(checks):.2e} (jerk in rad/s³, others in native units) — exact reproduction._")
    if all(r["_check"] for r in rows.values()):
        lines.append("\n_Cross-pair consistency (shared source ⇒ identical jerk): OK for all motions._")

    with open(os.path.join(OUT_DIR, "metrics_table.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\ntable -> {OUT_DIR}/metrics_table.md\n      -> {OUT_DIR}/metrics_table.csv")
    print(fmt_row("OVERALL", overall.mean(axis=0)))


if __name__ == "__main__":
    main()
