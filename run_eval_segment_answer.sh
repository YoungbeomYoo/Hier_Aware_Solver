#!/bin/bash
# ============================================================
# Segment Selection → Raw Caption → Answer (End-to-End)
# A5000에서 모델 1회 로딩, caption + summary 두 모드 순차 실행
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

QID_LIST="output/lvbench_mini_qid_list.tsv"
SEG_DIR="./output/eval_segment_selection/by_qid"
SIMG="/scratch2/youngbeom/simg/acl2026.simg"

# Caption mode
OUT_CAP="./output/eval_seg_answer_caption"
# Summary mode
OUT_SUM="./output/eval_seg_answer_summary"

echo "============================================"
echo "  Segment → Answer (caption + summary)"
echo "============================================"

mkdir -p "${OUT_CAP}/logs" "${OUT_CAP}/by_qid"
mkdir -p "${OUT_SUM}/logs" "${OUT_SUM}/by_qid"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=sga
#SBATCH --partition=gigabyte_a5000,asus_a5000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --output=${OUT_CAP}/logs/slurm_%j.out
#SBATCH --error=${OUT_CAP}/logs/slurm_%j.err

set -Eeuo pipefail
export NCCL_P2P_DISABLE=1

ml purge

singularity exec --nv \\
    --writable-tmpfs \\
    -B /scratch2/youngbeom:/scratch2/youngbeom \\
    -B /scratch2/youngbeom/slocal/acl2026:\$HOME/.local \\
    -B /lustre/youngbeom:/lustre/youngbeom \\
    ${SIMG} \\
    bash <<-INNER
cd ${SCRIPT_DIR}

echo "=== Mode: caption ==="
python eval_segment_answer.py \\
    --qid_list ${QID_LIST} \\
    --seg_result_dir ${SEG_DIR} \\
    --output_dir ${OUT_CAP} \\
    --mode caption

echo ""
echo "=== Mode: summary ==="
python eval_segment_answer.py \\
    --qid_list ${QID_LIST} \\
    --seg_result_dir ${SEG_DIR} \\
    --output_dir ${OUT_SUM} \\
    --mode summary

INNER

echo "=== All Done ==="
EOF

echo "  → Job submitted!"
