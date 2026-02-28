#!/bin/bash
# ============================================================
# 2026.02.27 — 두 실험 순차 실행 (sbatch로 SLURM job 제출)
#
# 1) Video-MME Subset — 기존 visual judge (exact match)
# 2) Video-MME Subset — semantic matching + visual judge (NEW)
#
# Usage:
#   sbatch 0227_run_both.sh
#   bash 0227_run_both.sh    # interactive node에서
# ============================================================
############ Settings for sbatch #############
#SBATCH --job-name=0227_both
#SBATCH --partition=gigabyte_a6000
#SBATCH --qos=big_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=/lustre/youngbeom/find_solver_please/output/logs/0227_both_%j.out
#SBATCH --error=/lustre/youngbeom/find_solver_please/output/logs/0227_both_%j.err
######### End of settings for sbatch #########

set -Eeuo pipefail
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=0
export NCCL_P2P_LEVEL=NVL
export NCCL_TIMEOUT=14400

ml purge 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

SIMG="/scratch2/youngbeom/simg/acl2026.simg"

mkdir -p output/logs

echo "============================================================"
echo "  [$(date)] Starting 0227 experiments"
echo "  Node: $(hostname)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "  Container: $SIMG"
echo "============================================================"
echo ""

# ============================================================
# Experiment 1: 기존 Visual Judge (exact match baseline)
# ============================================================
echo "============================================================"
echo "  [1/2] Video-MME Subset — Visual Judge (exact match)"
echo "  Config: config/video_mme_subset_tree_search_visual.yaml"
echo "============================================================"
echo ""

START1=$(date +%s)
singularity exec --nv \
    --writable-tmpfs \
    -B /scratch2/youngbeom:/scratch2/youngbeom \
    -B /scratch2/youngbeom/slocal/acl2026:$HOME/.local \
    -B /lustre/youngbeom:/lustre/youngbeom \
    $SIMG \
    bash -c "cd $SCRIPT_DIR && python solver.py --config config/video_mme_subset_tree_search_visual.yaml --cached 0"
END1=$(date +%s)
echo ""
echo "  [1/2] Done in $((END1 - START1))s"
echo ""

# ============================================================
# Experiment 2: Semantic Matching + Visual Judge (NEW)
# ============================================================
echo "============================================================"
echo "  [2/2] Video-MME Subset — Semantic Matching + Visual Judge"
echo "  Config: config/video_mme_subset_tree_search_semantic.yaml"
echo "============================================================"
echo ""

START2=$(date +%s)
singularity exec --nv \
    --writable-tmpfs \
    -B /scratch2/youngbeom:/scratch2/youngbeom \
    -B /scratch2/youngbeom/slocal/acl2026:$HOME/.local \
    -B /lustre/youngbeom:/lustre/youngbeom \
    $SIMG \
    bash -c "cd $SCRIPT_DIR && python solver.py --config config/video_mme_subset_tree_search_semantic.yaml --cached 0"
END2=$(date +%s)
echo ""
echo "  [2/2] Done in $((END2 - START2))s"
echo ""

# ============================================================
# Summary
# ============================================================
TOTAL=$((END2 - START1))
echo "============================================================"
echo "  [$(date)] All experiments done!"
echo "  Experiment 1 (visual):   $((END1 - START1))s"
echo "  Experiment 2 (semantic): $((END2 - START2))s"
echo "  Total: ${TOTAL}s"
echo ""
echo "  Results:"
echo "    Visual:   output/video_mme_subset_tree_search_visual3/"
echo "    Semantic: output/video_mme_subset_tree_search_semantic/"
echo "============================================================"
