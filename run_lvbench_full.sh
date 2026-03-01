#!/bin/bash
# ============================================================
# LVBench Full — Best Config (1549문제, 2파트)
# A1+B1+C3+D0+F1+G1+A2v3
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="config/full_run/lvbench_best.yaml"
OUTPUT_DIR="./output/lvbench_full_best"
SIMG="/scratch2/youngbeom/simg/acl2026.simg"

QID_P1="output/lvbench_full_qid_part1.tsv"
QID_P2="output/lvbench_full_qid_part2.tsv"

N1=$(wc -l < "$QID_P1")
N2=$(wc -l < "$QID_P2")

echo "============================================"
echo "  LVBench Full Best Config"
echo "  Part1: ${N1}, Part2: ${N2}"
echo "  Config: ${CONFIG}"
echo "  Output: ${OUTPUT_DIR}"
echo "============================================"

mkdir -p "${OUTPUT_DIR}/logs" "${OUTPUT_DIR}/by_qid"

# Part 1
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=lvb1
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

echo "=== lvb1 | Q=\${QID} V=\${VID} | task \${SLURM_ARRAY_TASK_ID}/${N1} ==="

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

echo "  → Part 1 submitted (${N1} tasks)"

# Part 2
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=lvb2
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

echo "=== lvb2 | Q=\${QID} V=\${VID} | task \${SLURM_ARRAY_TASK_ID}/${N2} ==="

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

echo "  → Part 2 submitted (${N2} tasks)"
echo "  → Total: $((N1 + N2)) questions"
