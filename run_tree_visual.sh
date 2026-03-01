#!/bin/bash
# ============================================================
# Tree-Guided Visual Search (TGVS) Pipeline
# 94Q hop-hard subset experiments
#
# Usage:
#   ./run_tree_visual.sh e1_baseline       # single experiment
#   ./run_tree_visual.sh submit_all        # submit all 6
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

QID_LIST="output/a2_subset_qid_list.tsv"
SIMG="/scratch2/youngbeom/simg/acl2026.simg"

if [ ! -f "$QID_LIST" ]; then
    echo "[ERROR] QID list not found: $QID_LIST"
    exit 1
fi

N_QUESTIONS=$(wc -l < "$QID_LIST")

ALL_EXPERIMENTS="e1_baseline e2_flat_coarse e3_llm_select e4_caption e5_multi_hop e6_heavy"

submit_one() {
    local EXP_NAME="$1"
    local CONFIG="config/tree_visual/${EXP_NAME}.yaml"
    local OUTPUT_DIR="./output/tree_visual_${EXP_NAME}"

    if [ ! -f "$CONFIG" ]; then
        echo "[ERROR] Config not found: $CONFIG"
        return 1
    fi

    echo "============================================"
    echo "  TGVS Experiment: ${EXP_NAME}"
    echo "  Questions: ${N_QUESTIONS}"
    echo "  Config:    ${CONFIG}"
    echo "  Output:    ${OUTPUT_DIR}"
    echo "============================================"

    mkdir -p "${OUTPUT_DIR}/logs" "${OUTPUT_DIR}/by_qid"

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=tv_${EXP_NAME}
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

echo "=== TGVS ${EXP_NAME} | Q=\${QID} V=\${VID} | task \${SLURM_ARRAY_TASK_ID}/${N_QUESTIONS} ==="

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

    echo "  → Array job submitted for ${EXP_NAME}! (${N_QUESTIONS} tasks)"
}

# Main
case "${1:-}" in
    submit_all)
        for exp in $ALL_EXPERIMENTS; do
            submit_one "$exp"
            sleep 1
        done
        echo ""
        echo "All 6 experiments submitted!"
        ;;
    "")
        echo "Usage: $0 <experiment_name|submit_all>"
        echo "Available experiments: $ALL_EXPERIMENTS"
        exit 1
        ;;
    *)
        submit_one "$1"
        ;;
esac
