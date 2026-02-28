#!/bin/bash
# ============================================================
# 2026.02.27 — Full-Set 4개 실험 SLURM 제출
#
# Video-MME (full) x 2 modes + LVBench (full) x 2 modes
#   1) Video-MME — Visual (exact match)
#   2) Video-MME — Semantic matching
#   3) LVBench   — Visual (exact match)
#   4) LVBench   — Semantic matching
#
# Usage:
#   bash 0227_run_all_full.sh              # 4개 모두 제출
#   bash 0227_run_all_full.sh video_mme    # Video-MME만 (visual + semantic)
#   bash 0227_run_all_full.sh lvbench      # LVBench만 (visual + semantic)
#   bash 0227_run_all_full.sh status       # 진행 상황 확인
#   bash 0227_run_all_full.sh aggregate    # 전부 aggregate
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

TARGET="${1:-all}"
PARTITION="${2:-gigabyte_a6000}"

echo "============================================================"
echo "  [$(date)] Full-Set Experiments — $TARGET"
echo "  Partition: $PARTITION"
echo "============================================================"
echo ""

case "$TARGET" in
    all)
        echo ">>> [1/4] Video-MME — Visual (exact match)"
        echo "------------------------------------------------------------"
        bash 0227_slurm_video_mme.sh all "$PARTITION" visual
        echo ""

        echo ">>> [2/4] Video-MME — Semantic matching"
        echo "------------------------------------------------------------"
        bash 0227_slurm_video_mme.sh all "$PARTITION" semantic
        echo ""

        echo ">>> [3/4] LVBench — Visual (exact match)"
        echo "------------------------------------------------------------"
        bash 0227_slurm_lvbench.sh all "$PARTITION" visual
        echo ""

        echo ">>> [4/4] LVBench — Semantic matching"
        echo "------------------------------------------------------------"
        bash 0227_slurm_lvbench.sh all "$PARTITION" semantic
        echo ""

        echo "============================================================"
        echo "  All 4 experiments submitted!"
        echo ""
        echo "  Monitor:    squeue -u \$USER"
        echo "  Status:     bash 0227_run_all_full.sh status"
        echo "  Aggregate:  bash 0227_run_all_full.sh aggregate"
        echo "============================================================"
        ;;

    video_mme|vmme)
        echo ">>> [1/2] Video-MME — Visual (exact match)"
        echo "------------------------------------------------------------"
        bash 0227_slurm_video_mme.sh all "$PARTITION" visual
        echo ""

        echo ">>> [2/2] Video-MME — Semantic matching"
        echo "------------------------------------------------------------"
        bash 0227_slurm_video_mme.sh all "$PARTITION" semantic
        echo ""

        echo "  Both Video-MME jobs submitted!"
        ;;

    lvbench|lv)
        echo ">>> [1/2] LVBench — Visual (exact match)"
        echo "------------------------------------------------------------"
        bash 0227_slurm_lvbench.sh all "$PARTITION" visual
        echo ""

        echo ">>> [2/2] LVBench — Semantic matching"
        echo "------------------------------------------------------------"
        bash 0227_slurm_lvbench.sh all "$PARTITION" semantic
        echo ""

        echo "  Both LVBench jobs submitted!"
        ;;

    status)
        echo "=== SLURM Job Status ==="
        squeue -u $USER -o "%.10i %.9P %.16j %.8T %.10M %.6D %R" 2>/dev/null || echo "  No jobs found"
        echo ""

        echo "=== Result Progress ==="
        for DIR in output/video_mme_full_visual output/video_mme_full_semantic output/lvbench_full_visual output/lvbench_full_semantic; do
            if [ -d "$DIR" ]; then
                LATEST=$(ls -d ${DIR}/v* 2>/dev/null | sort -t'v' -k2 -n | tail -1)
                if [ -n "$LATEST" ] && [ -d "$LATEST/by_qid" ]; then
                    DONE=$(ls "$LATEST/by_qid/"*.json 2>/dev/null | wc -l)
                    TOTAL=$(wc -l < "$LATEST/question_list.tsv" 2>/dev/null || echo "?")
                    echo "  $DIR: $DONE / $TOTAL"
                fi
            fi
        done
        ;;

    aggregate)
        echo "=== Aggregating all results ==="
        for MODE in visual semantic; do
            echo ""
            echo "--- Video-MME ($MODE) ---"
            bash 0227_slurm_video_mme.sh aggregate "$PARTITION" "$MODE" 2>/dev/null || echo "  (no results yet)"
            echo ""
            echo "--- LVBench ($MODE) ---"
            bash 0227_slurm_lvbench.sh aggregate "$PARTITION" "$MODE" 2>/dev/null || echo "  (no results yet)"
        done
        ;;

    *)
        echo "Usage: bash 0227_run_all_full.sh {all|video_mme|lvbench|status|aggregate} [partition]"
        echo ""
        echo "  all        — 4개 실험 전부 submit"
        echo "  video_mme  — Video-MME visual + semantic"
        echo "  lvbench    — LVBench visual + semantic"
        echo "  status     — 진행 상황"
        echo "  aggregate  — 결과 합치기"
        echo ""
        echo "Partition: gigabyte_a6000 (default)"
        exit 1
        ;;
esac
