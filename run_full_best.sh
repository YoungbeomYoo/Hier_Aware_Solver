#!/bin/bash
# ============================================================
# Full VideoMME Long — Best ablation combination (C3+D0)
# 900 questions, per-question array job
#
# Usage:
#   bash run_full_best.sh submit    # Submit SLURM array job
#   bash run_full_best.sh status    # Check progress
#   bash run_full_best.sh aggregate # Aggregate results
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="config/full_run/best_c3d0.yaml"
OUTPUT_DIR="./output/full_best_c3d0/v1"
QID_LIST="output/full_best_c3d0_qid_list.tsv"
SIMG="/scratch2/youngbeom/simg/acl2026.simg"

# ============================================================
# Generate full question list (qid → vid mapping)
# ============================================================
prepare() {
    echo "=== Generating full question list ==="

    # Use existing full question list if available, otherwise generate
    EXISTING_QID_TSV="output/video_mme_full_visual/v1/question_list.tsv"
    if [ -f "$EXISTING_QID_TSV" ]; then
        cp "$EXISTING_QID_TSV" "$QID_LIST"
        echo "  Copied from existing: $EXISTING_QID_TSV"
    else
        echo "  Generating from config..."
        singularity exec --nv \
            --writable-tmpfs \
            -B /scratch2/youngbeom:/scratch2/youngbeom \
            -B /scratch2/youngbeom/slocal/acl2026:$HOME/.local \
            -B /lustre/youngbeom:/lustre/youngbeom \
            ${SIMG} \
            python solver.py --config ${CONFIG} --dry_run --list_questions > "$QID_LIST"
    fi

    N_QUESTIONS=$(wc -l < "$QID_LIST")
    echo "  Total: ${N_QUESTIONS} questions ready"
    echo ""
}

# ============================================================
# Submit SLURM array job
# ============================================================
submit() {
    if [ ! -f "$QID_LIST" ]; then
        prepare
    fi

    local N_QUESTIONS
    N_QUESTIONS=$(wc -l < "$QID_LIST")

    echo "============================================"
    echo "  Full VideoMME Long — Best C3+D0"
    echo "  Questions: ${N_QUESTIONS}"
    echo "  Config:    ${CONFIG}"
    echo "  Output:    ${OUTPUT_DIR}"
    echo "  Container: ${SIMG}"
    echo "============================================"
    echo ""

    mkdir -p "${OUTPUT_DIR}/logs" "${OUTPUT_DIR}/by_qid"

    sbatch <<EOF
#!/bin/bash
############ Settings for sbatch #############
#SBATCH --job-name=full_best
#SBATCH --partition=suma_a6000,gigabyte_a6000,tyan_a6000,asus_6000ada
#SBATCH --qos=big_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --array=1-${N_QUESTIONS}
#SBATCH --output=${OUTPUT_DIR}/logs/slurm_%A_%a.out
#SBATCH --error=${OUTPUT_DIR}/logs/slurm_%A_%a.err
######### End of settings for sbatch #########

set -Eeuo pipefail
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=0
export NCCL_P2P_LEVEL=NVL
export NCCL_TIMEOUT=14400

ml purge

# N번째 줄에서 qid와 video_id 파싱
LINE=\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" ${SCRIPT_DIR}/${QID_LIST})
QID=\$(echo "\$LINE" | cut -f1)
VID=\$(echo "\$LINE" | cut -f2)

echo "CUDA_VISIBLE_DEVICES: \$CUDA_VISIBLE_DEVICES"
echo "SLURM_NODELIST: \$SLURM_NODELIST"
echo "=== full_best | Q=\${QID} V=\${VID} | task \${SLURM_ARRAY_TASK_ID}/${N_QUESTIONS} ==="

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
    echo ""
}

# ============================================================
# Status check
# ============================================================
status() {
    echo "=== Full Best C3+D0 — Status ==="
    squeue -u $USER -n "full_best" \
        -o "%.10i %.9P %.16j %.8T %.10M %.6D %R" 2>/dev/null || true
    echo ""

    local dir="${OUTPUT_DIR}/by_qid"
    if [ -d "$dir" ]; then
        local done=$(ls "$dir"/*.json 2>/dev/null | wc -l)
        local total=$(wc -l < "$QID_LIST" 2>/dev/null || echo "?")
        echo "  Progress: ${done}/${total}"
    else
        echo "  (no results yet)"
    fi
}

# ============================================================
# Aggregate results
# ============================================================
aggregate() {
    echo "=== Aggregating results ==="
    python aggregate.py --output_dir "$OUTPUT_DIR"
}

# ============================================================
# Main
# ============================================================
case "${1:-submit}" in
    prepare)
        prepare
        ;;
    submit)
        submit
        ;;
    status)
        status
        ;;
    aggregate)
        aggregate
        ;;
    *)
        echo "Usage: bash run_full_best.sh {prepare|submit|status|aggregate}"
        exit 1
        ;;
esac
