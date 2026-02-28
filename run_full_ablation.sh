#!/bin/bash
# ============================================================
# Full Ablation Study (900Q)
# Best config = A1+B1+C3+D0+F1+G1 (53.2%)
# 각 기능을 하나씩 제거하여 개별 기여도 확인
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

QID_LIST="output/full_best_c3d0_qid_list.tsv"
SIMG="/scratch2/youngbeom/simg/acl2026.simg"

if [ ! -f "$QID_LIST" ]; then
    echo "[ERROR] QID list not found: $QID_LIST"
    exit 1
fi

N_QUESTIONS=$(wc -l < "$QID_LIST")

# ── Ablation experiments ──
# Already done: -G1 = full_best_c3d0f1 (47.4%)
declare -A EXPERIMENTS
EXPERIMENTS[minus_f1]="config/full_run/ablation_minus_f1.yaml"
EXPERIMENTS[minus_c3_c0]="config/full_run/ablation_minus_c3_to_c0.yaml"
EXPERIMENTS[minus_c3_c2]="config/full_run/ablation_minus_c3_to_c2.yaml"
EXPERIMENTS[minus_a1]="config/full_run/ablation_minus_a1.yaml"
EXPERIMENTS[minus_b1]="config/full_run/ablation_minus_b1.yaml"

echo "============================================"
echo "  Full Ablation Study (900Q)"
echo "  Best: A1+B1+C3+D0+F1+G1 = 53.2%"
echo "  Questions: ${N_QUESTIONS}"
echo "============================================"

for EXP_NAME in "${!EXPERIMENTS[@]}"; do
    CONFIG="${EXPERIMENTS[$EXP_NAME]}"
    OUTPUT_DIR="./output/full_ablation_${EXP_NAME}/v1"

    echo ""
    echo "── Submitting: ${EXP_NAME}"
    echo "   Config: ${CONFIG}"
    echo "   Output: ${OUTPUT_DIR}"

    mkdir -p "${OUTPUT_DIR}/logs" "${OUTPUT_DIR}/by_qid"

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=abl_${EXP_NAME}
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

echo "=== abl_${EXP_NAME} | Q=\${QID} V=\${VID} | task \${SLURM_ARRAY_TASK_ID}/${N_QUESTIONS} ==="

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

    echo "   → Submitted!"
done

echo ""
echo "============================================"
echo "  All ${#EXPERIMENTS[@]} ablation jobs submitted!"
echo "  (${N_QUESTIONS} tasks each)"
echo "============================================"
