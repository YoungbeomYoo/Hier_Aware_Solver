#!/bin/bash
# ============================================================
# LVBench — Subset (81 videos, 1193 questions with memory)
# Best config: A1+B1+C3+D0+F1+G1+A2v3
# MaxArraySize=1001 이므로 2배치로 분할 제출
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="config/full_run/lvbench_best.yaml"
OUTPUT_DIR="./output/lvbench_best"
QID_LIST="output/lvbench_subset_qid_list.tsv"
SIMG="/scratch2/youngbeom/simg/acl2026.simg"

if [ ! -f "$QID_LIST" ]; then
    echo "[ERROR] QID list not found: $QID_LIST"
    exit 1
fi

N_QUESTIONS=$(wc -l < "$QID_LIST")
BATCH_SIZE=1000

echo "============================================"
echo "  LVBench Best Config"
echo "  Questions: ${N_QUESTIONS}"
echo "  Config:    ${CONFIG}"
echo "  Output:    ${OUTPUT_DIR}"
echo "  Batch size: ${BATCH_SIZE}"
echo "============================================"

mkdir -p "${OUTPUT_DIR}/logs" "${OUTPUT_DIR}/by_qid"

# Submit in batches of BATCH_SIZE
BATCH_START=1
BATCH_NUM=1
while [ $BATCH_START -le $N_QUESTIONS ]; do
    BATCH_END=$((BATCH_START + BATCH_SIZE - 1))
    if [ $BATCH_END -gt $N_QUESTIONS ]; then
        BATCH_END=$N_QUESTIONS
    fi

    echo "  Batch ${BATCH_NUM}: tasks ${BATCH_START}-${BATCH_END}"

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=lvb${BATCH_NUM}
#SBATCH --partition=suma_a6000,gigabyte_a6000,tyan_a6000,asus_6000ada
#SBATCH --qos=big_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --array=${BATCH_START}-${BATCH_END}
#SBATCH --output=${OUTPUT_DIR}/logs/slurm_%A_%a.out
#SBATCH --error=${OUTPUT_DIR}/logs/slurm_%A_%a.err

set -Eeuo pipefail
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=0
export NCCL_P2P_LEVEL=NVL
export NCCL_TIMEOUT=14400

ml purge

LINE=\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" ${SCRIPT_DIR}/${QID_LIST})
QID=\$(echo "\$LINE" | cut -f1)
VID=\$(echo "\$LINE" | cut -f2)

echo "=== lvb | Q=\${QID} V=\${VID} | task \${SLURM_ARRAY_TASK_ID}/${N_QUESTIONS} ==="

singularity exec --nv \\
    --writable-tmpfs \\
    -B /scratch2/youngbeom:/scratch2/youngbeom \\
    -B /scratch2/youngbeom/slocal/acl2026:\$HOME/.local \\
    -B /lustre/youngbeom:/lustre/youngbeom \\
    ${SIMG} \\
    bash <<-INNER
cd ${SCRIPT_DIR}
python solver.py \\
    --config ${CONFIG} \\
    --question_id "\${QID}" \\
    --video_id "\${VID}" \\
    --output_dir ${OUTPUT_DIR} \\
    --cached 1
INNER

echo "=== Done: \${QID} ==="
EOF

    BATCH_START=$((BATCH_END + 1))
    BATCH_NUM=$((BATCH_NUM + 1))
done

echo "  → All batches submitted! (${N_QUESTIONS} tasks total)"
