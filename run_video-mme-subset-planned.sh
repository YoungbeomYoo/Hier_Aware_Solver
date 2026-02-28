#!/bin/bash
# ============================================================
# Video-MME Subset — Planned Tree Search + Visual Judge Pipeline
# 10 videos, 30 questions
#
# Flow:
#   0. TreePlanner: upper-level overview → LLM selects focus regions → prune tree
#   1. Query decompose → Filtered tree → Budget context
#   2. VisualJudge (needs_visual? → frame captioning → rejudge)
#   3. Navigate/Answer
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="config/video_mme_subset_tree_search_planned.yaml"
OUTPUT_DIR="./output/video_mme_subset_tree_search_planned"

echo "============================================================"
echo "  Video-MME Subset — Planned Tree Search + Visual Judge"
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
