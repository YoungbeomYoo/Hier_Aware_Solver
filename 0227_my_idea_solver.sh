#!/bin/bash
# ============================================================
# 2026.02.27 — Semantic Matching + Tree Search Solver
# Qwen3-Embedding-0.6B 기반 시맨틱 매칭 실험
#
# Usage:
#   # Single GPU (기본)
#   bash 0227_my_idea_solver.sh video_mme
#   bash 0227_my_idea_solver.sh lvbench
#
#   # Multi-GPU 병렬 (비디오를 GPU 수로 분할)
#   bash 0227_my_idea_solver.sh video_mme 0,1,2,3
#   bash 0227_my_idea_solver.sh lvbench 0,1
#
#   # 특정 문제만
#   bash 0227_my_idea_solver.sh question 503
#   bash 0227_my_idea_solver.sh question 503 video_mme
#
#   # 특정 비디오만
#   bash 0227_my_idea_solver.sh video 2sriHX3PbXw
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

TARGET="${1:-video_mme}"
GPUS="${2:-}"

VMME_CONFIG="config/video_mme_subset_tree_search_semantic.yaml"
LV_CONFIG="config/lvbench_tree_search_semantic.yaml"

run_video_mme() {
    echo "============================================================"
    echo "  Video-MME Subset — Semantic Matching + Visual Judge"
    echo "============================================================"
    if [ -n "$GPUS" ]; then
        echo "  Multi-GPU mode: GPUs=$GPUS"
        python solver.py --config "$VMME_CONFIG" --gpus "$GPUS"
    else
        python solver.py --config "$VMME_CONFIG"
    fi
}

run_lvbench() {
    echo "============================================================"
    echo "  LVBench — Semantic Matching + Visual Judge"
    echo "============================================================"
    if [ -n "$GPUS" ]; then
        echo "  Multi-GPU mode: GPUs=$GPUS"
        python solver.py --config "$LV_CONFIG" --gpus "$GPUS"
    else
        python solver.py --config "$LV_CONFIG"
    fi
}

case "$TARGET" in
    video_mme|vmme)
        run_video_mme
        ;;
    lvbench|lv)
        run_lvbench
        ;;
    all)
        run_video_mme
        echo ""
        run_lvbench
        ;;
    question|q)
        # 특정 문제 하나만: bash 0227_my_idea_solver.sh question 503 [dataset]
        QID="${2:?Question ID required}"
        DATASET="${3:-video_mme}"
        if [ "$DATASET" = "lvbench" ] || [ "$DATASET" = "lv" ]; then
            CONFIG="$LV_CONFIG"
        else
            CONFIG="$VMME_CONFIG"
        fi
        echo "  Single question mode: Q=$QID (dataset=$DATASET)"
        python solver.py --config "$CONFIG" --question_id "$QID"
        ;;
    video|v)
        # 특정 비디오만: bash 0227_my_idea_solver.sh video 2sriHX3PbXw [dataset]
        VID="${2:?Video ID required}"
        DATASET="${3:-video_mme}"
        if [ "$DATASET" = "lvbench" ] || [ "$DATASET" = "lv" ]; then
            CONFIG="$LV_CONFIG"
        else
            CONFIG="$VMME_CONFIG"
        fi
        echo "  Single video mode: V=$VID (dataset=$DATASET)"
        python solver.py --config "$CONFIG" --video_id "$VID"
        ;;
    list)
        # 문제/비디오 목록 출력
        DATASET="${2:-video_mme}"
        if [ "$DATASET" = "lvbench" ] || [ "$DATASET" = "lv" ]; then
            CONFIG="$LV_CONFIG"
        else
            CONFIG="$VMME_CONFIG"
        fi
        echo "=== Videos ==="
        python solver.py --config "$CONFIG" --list_videos --dry_run 2>/dev/null || true
        echo ""
        echo "=== Questions (qid  video_id) ==="
        python solver.py --config "$CONFIG" --list_questions --dry_run 2>/dev/null || true
        ;;
    *)
        echo "Usage:"
        echo "  bash 0227_my_idea_solver.sh {video_mme|lvbench|all} [gpu_ids]"
        echo "  bash 0227_my_idea_solver.sh question <qid> [dataset]"
        echo "  bash 0227_my_idea_solver.sh video <vid> [dataset]"
        echo "  bash 0227_my_idea_solver.sh list [dataset]"
        echo ""
        echo "Examples:"
        echo "  bash 0227_my_idea_solver.sh video_mme          # single GPU"
        echo "  bash 0227_my_idea_solver.sh lvbench 0,1,2,3    # 4 GPUs parallel"
        echo "  bash 0227_my_idea_solver.sh question 503       # 문제 하나만"
        echo "  bash 0227_my_idea_solver.sh video 2sriHX3PbXw  # 비디오 하나만"
        exit 1
        ;;
esac

echo ""
echo "Done! Check output/ for results."
