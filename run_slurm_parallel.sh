#!/bin/bash
# ============================================================
# SLURM Parallel Solver — 비디오별 병렬 처리
#
# Usage:
#   # Step 1: 비디오 목록 생성
#   bash run_slurm_parallel.sh prepare <config.yaml>
#
#   # Step 2: SLURM array job 제출
#   bash run_slurm_parallel.sh submit <config.yaml>
#
#   # Step 3: 모든 job 완료 후 결과 합치기
#   bash run_slurm_parallel.sh aggregate <config.yaml>
#
# 또는 한번에:
#   bash run_slurm_parallel.sh all <config.yaml>
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

ACTION="${1:-all}"
CONFIG="${2:-config/video_mme_tree_search_visual.yaml}"

# Config에서 output_dir 추출
OUTPUT_DIR=$(python3 -c "
import yaml, sys
with open('$CONFIG') as f:
    c = yaml.safe_load(f)
print(c.get('paths', {}).get('output_dir', './output/default'))
")

VIDEO_LIST="${OUTPUT_DIR}/video_list.txt"

prepare() {
    echo "=== Preparing video list ==="
    mkdir -p "$OUTPUT_DIR"
    python solver.py --config "$CONFIG" --list_videos > "$VIDEO_LIST"
    N_VIDEOS=$(wc -l < "$VIDEO_LIST")
    echo "  Found $N_VIDEOS videos"
    echo "  Video list saved to: $VIDEO_LIST"
    echo ""
    echo "  To submit SLURM jobs:"
    echo "    bash run_slurm_parallel.sh submit $CONFIG"
}

submit() {
    if [ ! -f "$VIDEO_LIST" ]; then
        echo "Video list not found. Running prepare first..."
        prepare
    fi

    N_VIDEOS=$(wc -l < "$VIDEO_LIST")
    echo "=== Submitting SLURM array job ==="
    echo "  Config: $CONFIG"
    echo "  Videos: $N_VIDEOS"
    echo "  Output: $OUTPUT_DIR"
    echo ""

    # SLURM array job 제출
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=solver
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=1-${N_VIDEOS}
#SBATCH --output=${OUTPUT_DIR}/logs/slurm_%A_%a.out
#SBATCH --error=${OUTPUT_DIR}/logs/slurm_%A_%a.err

mkdir -p ${OUTPUT_DIR}/logs

cd ${SCRIPT_DIR}

# SLURM_ARRAY_TASK_ID 번째 비디오 ID 가져오기
VIDEO_ID=\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" ${VIDEO_LIST})

echo "=== Task \${SLURM_ARRAY_TASK_ID}: Video \${VIDEO_ID} ==="
echo "Config: ${CONFIG}"
echo "Output: ${OUTPUT_DIR}"
echo ""

python solver.py --config ${CONFIG} --video_id "\${VIDEO_ID}" --cached 1

echo ""
echo "=== Task \${SLURM_ARRAY_TASK_ID} done ==="
EOF

    echo "  SLURM array job submitted!"
    echo ""
    echo "  Monitor: squeue -u \$USER"
    echo "  After all done: bash run_slurm_parallel.sh aggregate $CONFIG"
}

aggregate_results() {
    echo "=== Aggregating results ==="
    python aggregate.py --output_dir "$OUTPUT_DIR"
}

case "$ACTION" in
    prepare)
        prepare
        ;;
    submit)
        submit
        ;;
    aggregate)
        aggregate_results
        ;;
    all)
        prepare
        submit
        echo ""
        echo "Jobs submitted. After all complete, run:"
        echo "  bash run_slurm_parallel.sh aggregate $CONFIG"
        ;;
    *)
        echo "Usage: bash run_slurm_parallel.sh {prepare|submit|aggregate|all} <config.yaml>"
        exit 1
        ;;
esac
