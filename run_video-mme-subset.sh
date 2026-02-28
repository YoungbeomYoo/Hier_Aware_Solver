#!/bin/bash
# ============================================================
# Video-MME Subset — Tree Search Pipeline
# 10 videos, 30 questions
#
# Flow:
#   Query decompose → Filtered tree → Budget context
#   → Judge (answerable?) → Navigate/Visual → Answer
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="config/video_mme_subset_tree_search.yaml"
OUTPUT_DIR="./output/video_mme_subset_tree_search"

echo "============================================================"
echo "  Video-MME Subset — Tree Search Pipeline"
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
