#!/bin/bash
# ============================================================
# Video-MME (Full) — Tree Search + Visual Judge Pipeline
#
# Flow:
#   Query decompose → Filtered tree → Budget context
#   → VisualJudge (needs_visual? → frame captioning → rejudge)
#   → Navigate/Answer
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="config/video_mme_tree_search_visual.yaml"
OUTPUT_DIR="./output/video_mme_tree_search_visual"

echo "============================================================"
echo "  Video-MME (Full) — Tree Search + Visual Judge"
echo "  Config: $CONFIG"
echo "  Output: $OUTPUT_DIR"
echo "============================================================"
echo ""

# Auto-versioning: solver.py가 자동으로 v1, v2, ... 폴더 생성

# Run
python solver.py --config "$CONFIG" --cached 1

echo ""
echo "============================================================"
echo "  Done! Results in: $OUTPUT_DIR/vN (auto-versioned)"
echo "============================================================"
