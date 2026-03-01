#!/bin/bash
# ============================================================
# LVBench Budget-Constrained Ablation
# K=20,30,50 × uniform/sequential/hierarchy + baseline(all)
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

SIMG="/scratch2/youngbeom/simg/acl2026.simg"
QID_P1="output/lvbench_full_qid_part1.tsv"
QID_P2="output/lvbench_full_qid_part2.tsv"

N1=$(wc -l < "$QID_P1")
N2=$(wc -l < "$QID_P2")

ALL_EXPERIMENTS="lv_b0_all lv_b20_uniform lv_b20_sequential lv_b20_hierarchy lv_b30_uniform lv_b30_sequential lv_b30_hierarchy lv_b50_uniform lv_b50_sequential lv_b50_hierarchy lv_b50_verified"

submit_one() {
    local EXP_NAME="$1"
    local CONFIG="config/budget_ablation/${EXP_NAME}.yaml"
    local OUTPUT_DIR="./output/budget_${EXP_NAME}"

    if [ ! -f "$CONFIG" ]; then
        echo "[ERROR] Config not found: $CONFIG"
        return 1
    fi

    echo "============================================"
    echo "  LVBench Budget: ${EXP_NAME}"
    echo "  Part1: ${N1}, Part2: ${N2}"
    echo "  Config: ${CONFIG}"
    echo "  Output: ${OUTPUT_DIR}"
    echo "============================================"

    mkdir -p "${OUTPUT_DIR}/logs" "${OUTPUT_DIR}/by_qid"

    # Part 1
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=lb_${EXP_NAME:3:7}1
#SBATCH --partition=suma_a6000,gigabyte_a6000,tyan_a6000,asus_6000ada
#SBATCH --qos=big_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --array=1-${N1}
#SBATCH --output=${OUTPUT_DIR}/logs/slurm_p1_%A_%a.out
#SBATCH --error=${OUTPUT_DIR}/logs/slurm_p1_%A_%a.err

set -Eeuo pipefail
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=0
export NCCL_P2P_LEVEL=NVL
export NCCL_TIMEOUT=14400

ml purge

LINE=\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" ${SCRIPT_DIR}/${QID_P1})
QID=\$(echo "\$LINE" | cut -f1)
VID=\$(echo "\$LINE" | cut -f2)

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
EOF

    # Part 2
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=lb_${EXP_NAME:3:7}2
#SBATCH --partition=suma_a6000,gigabyte_a6000,tyan_a6000,asus_6000ada
#SBATCH --qos=big_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --array=1-${N2}
#SBATCH --output=${OUTPUT_DIR}/logs/slurm_p2_%A_%a.out
#SBATCH --error=${OUTPUT_DIR}/logs/slurm_p2_%A_%a.err

set -Eeuo pipefail
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=0
export NCCL_P2P_LEVEL=NVL
export NCCL_TIMEOUT=14400

ml purge

LINE=\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" ${SCRIPT_DIR}/${QID_P2})
QID=\$(echo "\$LINE" | cut -f1)
VID=\$(echo "\$LINE" | cut -f2)

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
EOF

    echo "  → ${EXP_NAME} submitted (${N1}+${N2} tasks)"
}

check_status() {
    echo "============================================"
    echo "  LVBench Budget Ablation Results"
    echo "============================================"
    printf "  %-25s %-15s %s\n" "Experiment" "Accuracy" "Progress"
    echo "  -------------------------------------------------"
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
acc = f'{c/t*100:.1f}%' if t > 0 else '0%'
print(f'{c}/{t} = {acc}  {t}/1549')
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
            sleep 2
        done
        echo ""
        echo "All experiments submitted!"
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
