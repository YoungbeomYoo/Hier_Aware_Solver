#!/bin/bash
# ============================================================
# Full VideoMME Long — Best C3+D0+F1+G1 (Phase 0 coarse_first)
# 900 questions, per-question array job
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="config/full_run/best_c3d0f1_g1.yaml"
OUTPUT_DIR="./output/full_best_c3d0f1_g1/v1"
QID_LIST="output/full_best_c3d0_qid_list.tsv"
SIMG="/scratch2/youngbeom/simg/acl2026.simg"

if [ ! -f "$QID_LIST" ]; then
    echo "[ERROR] QID list not found: $QID_LIST"
    exit 1
fi

N_QUESTIONS=$(wc -l < "$QID_LIST")

echo "============================================"
echo "  Full VideoMME Long — Best C3+D0+F1+G1"
echo "  Questions: ${N_QUESTIONS}"
echo "  Config:    ${CONFIG}"
echo "  Output:    ${OUTPUT_DIR}"
echo "============================================"

mkdir -p "${OUTPUT_DIR}/logs" "${OUTPUT_DIR}/by_qid"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=full_g1
#SBATCH --partition=suma_a6000,gigabyte_a6000,tyan_a6000,asus_6000ada
#SBATCH --qos=big_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --array=1-${N_QUESTIONS}
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

echo "=== full_g1 | Q=\${QID} V=\${VID} | task \${SLURM_ARRAY_TASK_ID}/${N_QUESTIONS} ==="

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

echo "  → Array job submitted! (${N_QUESTIONS} tasks)"
