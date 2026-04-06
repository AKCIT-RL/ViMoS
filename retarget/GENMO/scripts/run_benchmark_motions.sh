#!/usr/bin/env bash
# GENMO motion benchmark: runs all 5 phases sequentially (text and/or video).
# Usage: ./scripts/run_benchmark_motions.sh
# Or:    VIDEO_DIR=/path/to/videos OUTPUT_ROOT=outputs/demo/my_bench ./scripts/run_benchmark_motions.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

# Configurable variables (can be set via environment)
CKPT_PATH="${CKPT_PATH:-inputs/checkpoints/s050000.ckpt}"
EXP="${EXP:-genmo_lg}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/demo/benchmark}"
VIDEO_DIR="${VIDEO_DIR:-}"

run_text() {
  local name="$1"
  local prompt="$2"
  local length="${3:-300}"
  echo "[run_text] $name"
  python scripts/demo/demo_text.py \
    text1="$prompt" \
    text_length="$length" \
    output_dir="$OUTPUT_ROOT/$name" \
    exp="$EXP" \
    ckpt_path="$CKPT_PATH" \
    rsync_ckpt=false \
    pipeline.args.return_mid=false
}

run_video() {
  local name="$1"
  local video_path="$2"
  echo "[run_video] $name"
  python scripts/demo/demo_text.py \
    video1_path="$video_path" \
    video1_name="$name" \
    output_dir="$OUTPUT_ROOT/$name" \
    exp="$EXP" \
    ckpt_path="$CKPT_PATH" \
    rsync_ckpt=false \
    pipeline.args.return_mid=false \
    static_cam1=true \
    orig_fps1=30
}

# Phase 5: if video exists in VIDEO_DIR, use video; otherwise use text only
run_text_or_video() {
  local name="$1"
  local prompt="$2"
  local video_basename="$3"
  if [ -n "$VIDEO_DIR" ] && [ -f "$VIDEO_DIR/$video_basename" ]; then
    run_video "$name" "$VIDEO_DIR/$video_basename"
  else
    run_text "$name" "$prompt"
  fi
}

# ---------------------------------------------------------------------------
# Phase 1: Sanity Test
# ---------------------------------------------------------------------------
echo "=== Phase 1: Sanity Test ==="
#run_text "phase1_walk_straight" "A person walks in a straight line then stops completely."
#run_text "phase1_walk_to_run" "Smooth transition from walking to running."
#run_text "phase1_dance_spins" "Simple dance choreography with spins on the spot."
#run_text "phase1_squat" "Squat down and stand back up."
#run_text "phase1_jumps" "Jump several times in place."
#run_text "phase1_one_leg" "Stand on one leg balancing."

# ---------------------------------------------------------------------------
# Phase 2: Physical Load and Inertia
# ---------------------------------------------------------------------------
echo "=== Phase 2: Physical Load and Inertia ==="
run_text "phase2_run_180" "Run at full speed then make a sharp 180 degree turn."
run_text "phase2_lift_heavy_box" "Pick up a visibly heavy object like a 30kg box and lift it."

# ---------------------------------------------------------------------------
# Phase 3: Severe Interaction and Occlusions
# ---------------------------------------------------------------------------
echo "=== Phase 3: Severe Interaction and Occlusions ==="
run_text "phase3_trip_fall_get_up" "Trip fall forward on the ground then quickly get up."
run_text "phase3_crawl_under_obstacle" "Crawl under a low obstacle."
run_text "phase3_climb_wall" "Climb over a wall or chest-high obstacle."

# ---------------------------------------------------------------------------
# Phase 4: Multi-Agent
# ---------------------------------------------------------------------------
echo "=== Phase 4: Multi-Agent ==="
run_text "phase4_two_people_hug" "Two people run toward each other and hug strongly."
run_text "phase4_body_collision" "Intense collision strong lateral push or body-to-body block."

# ---------------------------------------------------------------------------
# Phase 5: Articular Precision and Fine Kinematics (Video vs. Text)
# ---------------------------------------------------------------------------
echo "=== Phase 5: Articular Precision (video if VIDEO_DIR has files) ==="
run_text_or_video "phase5_wrist_screwdriver" \
  "Rotate the wrist repeatedly and quickly left to right like using a screwdriver." \
  "wrist_screwdriver.mp4"
run_text_or_video "phase5_tiptoe_walk" \
  "Walk cautiously on tiptoes taking three slow steps forward." \
  "tiptoe_walk.mp4"
run_text_or_video "phase5_neck_shoulder_nape" \
  "Turn the neck to look over the shoulder while the opposite arm goes behind the head to touch the back of the neck." \
  "neck_shoulder_nape.mp4"
run_text_or_video "phase5_deep_squat_knees" \
  "Deep squat to the ground hugging the knees to the chest and curling up completely." \
  "deep_squat_knees.mp4"

echo "=== Benchmark complete. Outputs in: $OUTPUT_ROOT ==="
