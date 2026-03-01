#!/bin/bash
# ============================================================
# Segment Selection Eval — LVBench mini (60 questions)
# 단일 GPU에서 전체 순차 처리 (모델 1회 로딩)
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

OUTPUT_DIR="./output/eval_segment_selection"
QID_LIST="output/lvbench_mini_qid_list.tsv"
SIMG="/scratch2/youngbeom/simg/acl2026.simg"

echo "============================================"
echo "  Segment Selection Eval"
echo "  Output: ${OUTPUT_DIR}"
echo "============================================"

mkdir -p "${OUTPUT_DIR}/logs" "${OUTPUT_DIR}/by_qid"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=seg
#SBATCH --partition=gigabyte_a5000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=${OUTPUT_DIR}/logs/slurm_%j.out
#SBATCH --error=${OUTPUT_DIR}/logs/slurm_%j.err

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
python eval_segment_selection.py \\
    --qid_list ${QID_LIST} \\
    --output_dir ${OUTPUT_DIR} \\
    --n_select 5
INNER

echo "=== Done ==="
EOF

echo "  → Job submitted!"
