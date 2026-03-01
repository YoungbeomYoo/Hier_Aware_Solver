#!/bin/bash
# ============================================================
# Budget-Constrained Leaf Selection Ablation
# K=20,30,50 × uniform/sequential/hierarchy + baseline(all)
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

ALL_EXPERIMENTS="b0_all b20_uniform b20_sequential b20_hierarchy b30_uniform b30_sequential b30_hierarchy b50_uniform b50_sequential b50_hierarchy"

submit_one() {
    local EXP_NAME="$1"
    local CONFIG="config/budget_ablation/${EXP_NAME}.yaml"
    local OUTPUT_DIR="./output/budget_${EXP_NAME}"

    if [ ! -f "$CONFIG" ]; then
        echo "[ERROR] Config not found: $CONFIG"
        return 1
    fi

    echo "============================================"
    echo "  Budget Ablation: ${EXP_NAME}"
    echo "  Questions: ${N_QUESTIONS}"
    echo "  Config:    ${CONFIG}"
    echo "  Output:    ${OUTPUT_DIR}"
    echo "============================================"

    mkdir -p "${OUTPUT_DIR}/logs" "${OUTPUT_DIR}/by_qid"

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=bg_${EXP_NAME:0:8}
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

echo "=== budget_${EXP_NAME} | Q=\${QID} V=\${VID} | task \${SLURM_ARRAY_TASK_ID}/${N_QUESTIONS} ==="

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

    echo "  → submitted (${N_QUESTIONS} tasks)"
}

check_status() {
    echo "============================================"
    echo "  Budget Ablation Results"
    echo "============================================"
    printf "  %-25s %s\n" "Experiment" "Result"
    echo "  -----------------------------------------"
    for exp in $ALL_EXPERIMENTS; do
        dir="./output/budget_${exp}/by_qid"
        if [ -d "$dir" ]; then
            result=$(python3 -c "
import json, glob
c=0; t=0
for f in glob.glob('$dir/*.json'):
    d=json.load(open(f))
    t+=1
    if d.get('correct'): c+=1
print(f'{c}/{t} = {c/t*100:.1f}%' if t > 0 else '0/0')
" 2>/dev/null)
            printf "  %-25s %s\n" "$exp" "$result"
        else
            printf "  %-25s %s\n" "$exp" "not started"
        fi
    done
}

case "${1:-}" in
    submit_all)
        for exp in $ALL_EXPERIMENTS; do
            submit_one "$exp"
            sleep 1
        done
        echo ""
        echo "All ${#ALL_EXPERIMENTS[@]} experiments submitted!"
        ;;
    status)
        check_status
        ;;
    "")
        echo "Usage: $0 <experiment_name|submit_all|status>"
        echo "Available: $ALL_EXPERIMENTS"
        exit 1
        ;;
    *)
        submit_one "$1"
        ;;
esac
