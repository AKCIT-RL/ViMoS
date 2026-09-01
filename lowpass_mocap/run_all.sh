#!/usr/bin/env bash
# Full pipeline:
#   1. auto-tuned low-pass filter on the mocap CSVs        -> output/csv, output/pkl
#   2. refresh the GenMo CSV copy                          -> CSV_Genmo/
#   3. full metrics table (GMR eval pipeline, 3 pairs/motion) -> output/metrics_table.*
#   4. 3-panel comparison videos                           -> output/videos/
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$HERE/../retarget/GMR/.venv/bin/python"

echo "== 1/4  low-pass filter + auto-tune =="
"$PY" "$HERE/lowpass_filter.py" "$@"

echo; echo "== 2/4  copy GenMo CSVs -> CSV_Genmo/ =="
"$PY" - "$@" <<'PY'
import shutil, os, sys
from lowpass_filter import NAME_MAP, GENMO_DIR
only = sys.argv[2:] if len(sys.argv) > 1 and sys.argv[1] == "--only" else None
os.makedirs(os.path.join(os.path.dirname(__file__) or ".", "CSV_Genmo"), exist_ok=True)
for name, stem in NAME_MAP.items():
    if only and name not in only:
        continue
    shutil.copyfile(os.path.join(GENMO_DIR, f"unitree_g1_{stem}.csv"),
                    os.path.join("CSV_Genmo", f"{name}.csv"))
print("CSV_Genmo refreshed")
PY

echo; echo "== 3/4  metrics table =="
"$PY" "$HERE/build_metrics_table.py" "$@"

echo; echo "== 4/4  rendering 3-panel videos =="
MUJOCO_GL=egl "$PY" "$HERE/render_compare.py" "$@"

echo; echo "outputs in $HERE/output/"
