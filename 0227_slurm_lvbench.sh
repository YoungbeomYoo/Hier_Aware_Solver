#!/bin/bash
# ============================================================
# SLURM — LVBench Full-Set (qid-level parallel)
#
# 세 가지 모드 지원:
#   visual   — exact match tree filter + visual judge
#   semantic — Qwen3-Embedding-0.6B semantic matching + visual judge
#   exact    — semantic 구조에서 embedding 빼고 exact key matching만
#
# Usage:
#   # Semantic (기본)
#   bash 0227_slurm_lvbench.sh submit
#   bash 0227_slurm_lvbench.sh submit gigabyte_a6000 semantic
#
#   # Exact (semantic 구조 + exact key matching)
#   bash 0227_slurm_lvbench.sh submit gigabyte_a6000 exact
#
#   # 결과 합치기
#   bash 0227_slurm_lvbench.sh aggregate gigabyte_a6000 semantic
#   bash 0227_slurm_lvbench.sh aggregate gigabyte_a6000 exact
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

ACTION="${1:-all}"
PARTITION="${2:-gigabyte_a6000}"
MODE="${3:-semantic}"

# Singularity container
SIMG="/scratch2/youngbeom/simg/acl2026.simg"

# Config 선택
if [ "$MODE" = "visual" ]; then
    CONFIG="config/lvbench_full_tree_search_visual.yaml"
    JOB_NAME="lv_vis"
    DEFAULT_OUTPUT="./output/lvbench_full_visual"
elif [ "$MODE" = "exact" ]; then
    CONFIG="config/lvbench_full_tree_search_exact.yaml"
    JOB_NAME="lv_ext"
    DEFAULT_OUTPUT="./output/lvbench_full_exact"
elif [ "$MODE" = "exact_vl" ]; then
    CONFIG="config/lvbench_full_tree_search_exact_videolucy.yaml"
    JOB_NAME="lv_evl"
    DEFAULT_OUTPUT="./output/lvbench_full_exact_videolucy"
else
    CONFIG="config/lvbench_full_tree_search_semantic.yaml"
    JOB_NAME="lv_sem"
    DEFAULT_OUTPUT="./output/lvbench_full_semantic"
fi

# Config에서 output_dir 추출
OUTPUT_BASE=$(python3 -c "
import yaml
with open('$CONFIG') as f:
    c = yaml.safe_load(f)
print(c.get('paths', {}).get('output_dir', '$DEFAULT_OUTPUT'))
")

# Auto-version: 다음 vN 폴더 결정
if [ ! -d "$OUTPUT_BASE" ]; then
    OUTPUT_DIR="${OUTPUT_BASE}/v1"
else
    LAST_V=$(ls -d ${OUTPUT_BASE}/v* 2>/dev/null | sort -t'v' -k2 -n | tail -1 | grep -oP 'v\K\d+' || echo "0")
    NEXT_V=$((LAST_V + 1))
    OUTPUT_DIR="${OUTPUT_BASE}/v${NEXT_V}"
fi

mkdir -p "$OUTPUT_DIR/logs"
mkdir -p "$OUTPUT_DIR/by_qid"

QID_LIST="${OUTPUT_DIR}/question_list.tsv"

prepare() {
    echo "=== Preparing question list (mode=$MODE) ==="
    echo "  Config: $CONFIG"
    echo "  Output: $OUTPUT_DIR"

    python3 -c "
import sys, os, json, yaml
sys.path.insert(0, '$SCRIPT_DIR')
with open('$CONFIG') as f:
    config = yaml.safe_load(f)

adapter_config = config.get('paths', {})
adapter_config['output_dir'] = '$OUTPUT_DIR'

from adapters.lvbench import LVBenchAdapter
adapter = LVBenchAdapter(adapter_config)
questions_by_video = adapter.load_questions()

with open('$QID_LIST', 'w') as f:
    for vid in sorted(questions_by_video.keys()):
        for q in questions_by_video[vid]:
            f.write(f\"{q['question_id']}\t{vid}\n\")

total = sum(len(qs) for qs in questions_by_video.values())
print(f'  Found {total} questions from {len(questions_by_video)} videos')
"
    N_QUESTIONS=$(wc -l < "$QID_LIST")
    echo "  Question list saved to: $QID_LIST ($N_QUESTIONS questions)"
}

submit() {
    if [ ! -f "$QID_LIST" ]; then
        echo "Question list not found. Running prepare first..."
        prepare
    fi

    N_QUESTIONS=$(wc -l < "$QID_LIST")
    MAX_ARRAY=1000  # SLURM MaxArraySize=1001

    echo "=== Submitting SLURM array job (mode=$MODE) ==="
    echo "  Config: $CONFIG"
    echo "  Partition: $PARTITION"
    echo "  Questions: $N_QUESTIONS"
    echo "  Output: $OUTPUT_DIR"
    echo "  Container: $SIMG"
    echo ""

    # 1000개씩 나눠서 제출 (array index는 항상 1-N, offset으로 실제 줄 번호 계산)
    START=1
    BATCH=1
    while [ $START -le $N_QUESTIONS ]; do
        END=$((START + MAX_ARRAY - 1))
        if [ $END -gt $N_QUESTIONS ]; then
            END=$N_QUESTIONS
        fi

        ARRAY_SIZE=$((END - START + 1))
        OFFSET=$((START - 1))

        echo "  Batch $BATCH: questions $START-$END (array 1-$ARRAY_SIZE, offset=$OFFSET)"

        sbatch <<EOF
#!/bin/bash
############ Settings for sbatch #############
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --qos=big_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --array=1-${ARRAY_SIZE}
#SBATCH --output=${OUTPUT_DIR}/logs/slurm_%A_%a.out
#SBATCH --error=${OUTPUT_DIR}/logs/slurm_%A_%a.err
######### End of settings for sbatch #########

set -Eeuo pipefail
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=0
export NCCL_P2P_LEVEL=NVL
export NCCL_TIMEOUT=14400

ml purge

# offset + array_task_id = 실제 줄 번호
ACTUAL_IDX=\$((${OFFSET} + SLURM_ARRAY_TASK_ID))

echo "CUDA_VISIBLE_DEVICES: \$CUDA_VISIBLE_DEVICES"
echo "SLURM_NODELIST: \$SLURM_NODELIST"
echo "Running task \${ACTUAL_IDX}/${N_QUESTIONS} (mode=${MODE}, batch=${BATCH})"

# N번째 줄에서 qid와 video_id 파싱
LINE=\$(sed -n "\${ACTUAL_IDX}p" ${QID_LIST})
QID=\$(echo "\$LINE" | cut -f1)
VID=\$(echo "\$LINE" | cut -f2)

echo "=== Q=\${QID} V=\${VID} ==="
echo "Config: ${CONFIG}"
echo "Output: ${OUTPUT_DIR}"
echo ""

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
    --output_dir "${OUTPUT_DIR}" \\
    --cached 1
INNER

echo ""
echo "=== Task \${ACTUAL_IDX} done ==="
EOF

        START=$((END + 1))
        BATCH=$((BATCH + 1))
    done

    echo ""
    echo "  SLURM array job(s) submitted! (mode=$MODE)"
    echo ""
    echo "  Monitor: squeue -u \$USER -n $JOB_NAME"
    echo "  After done: bash 0227_slurm_lvbench.sh aggregate $PARTITION $MODE"
}

aggregate_results() {
    echo "=== Aggregating results (mode=$MODE) ==="
    echo "  Output: $OUTPUT_DIR"
    python aggregate.py --output_dir "$OUTPUT_DIR"
}

status() {
    echo "=== Job Status (mode=$MODE) ==="
    squeue -u $USER -n $JOB_NAME -o "%.10i %.9P %.12j %.8T %.10M %.6D %R" 2>/dev/null || echo "  No jobs found"
    echo ""
    if [ -d "$OUTPUT_DIR/by_qid" ]; then
        DONE=$(ls "$OUTPUT_DIR/by_qid/"*.json 2>/dev/null | wc -l)
        TOTAL=$(wc -l < "$QID_LIST" 2>/dev/null || echo "?")
        echo "  Progress: $DONE / $TOTAL questions done"
    fi
}

run_both_modes() {
    echo "============================================================"
    echo "  LVBench Full — Running BOTH modes"
    echo "  Partition: $PARTITION"
    echo "============================================================"
    echo ""

    # Visual mode
    echo "--- [1/2] Visual (exact match) ---"
    MODE="visual"
    CONFIG="config/lvbench_full_tree_search_visual.yaml"
    JOB_NAME="lv_vis"
    DEFAULT_OUTPUT="./output/lvbench_full_visual"
    OUTPUT_BASE=$(python3 -c "
import yaml
with open('$CONFIG') as f:
    c = yaml.safe_load(f)
print(c.get('paths', {}).get('output_dir', '$DEFAULT_OUTPUT'))
")
    if [ ! -d "$OUTPUT_BASE" ]; then
        OUTPUT_DIR="${OUTPUT_BASE}/v1"
    else
        LAST_V=$(ls -d ${OUTPUT_BASE}/v* 2>/dev/null | sort -t'v' -k2 -n | tail -1 | grep -oP 'v\K\d+' || echo "0")
        NEXT_V=$((LAST_V + 1))
        OUTPUT_DIR="${OUTPUT_BASE}/v${NEXT_V}"
    fi
    mkdir -p "$OUTPUT_DIR/logs" "$OUTPUT_DIR/by_qid"
    QID_LIST="${OUTPUT_DIR}/question_list.tsv"
    prepare
    submit
    echo ""

    # Semantic mode
    echo "--- [2/2] Semantic matching ---"
    MODE="semantic"
    CONFIG="config/lvbench_full_tree_search_semantic.yaml"
    JOB_NAME="lv_sem"
    DEFAULT_OUTPUT="./output/lvbench_full_semantic"
    OUTPUT_BASE=$(python3 -c "
import yaml
with open('$CONFIG') as f:
    c = yaml.safe_load(f)
print(c.get('paths', {}).get('output_dir', '$DEFAULT_OUTPUT'))
")
    if [ ! -d "$OUTPUT_BASE" ]; then
        OUTPUT_DIR="${OUTPUT_BASE}/v1"
    else
        LAST_V=$(ls -d ${OUTPUT_BASE}/v* 2>/dev/null | sort -t'v' -k2 -n | tail -1 | grep -oP 'v\K\d+' || echo "0")
        NEXT_V=$((LAST_V + 1))
        OUTPUT_DIR="${OUTPUT_BASE}/v${NEXT_V}"
    fi
    mkdir -p "$OUTPUT_DIR/logs" "$OUTPUT_DIR/by_qid"
    QID_LIST="${OUTPUT_DIR}/question_list.tsv"
    prepare
    submit

    echo ""
    echo "============================================================"
    echo "  Both jobs submitted!"
    echo "  Monitor: squeue -u \$USER"
    echo "============================================================"
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
    status)
        status
        ;;
    all)
        prepare
        submit
        echo ""
        echo "Jobs submitted (mode=$MODE). After all complete, run:"
        echo "  bash 0227_slurm_lvbench.sh aggregate $PARTITION $MODE"
        ;;
    both)
        run_both_modes
        ;;
    *)
        echo "Usage: bash 0227_slurm_lvbench.sh {submit|aggregate|status|all|both} [partition] [mode]"
        echo ""
        echo "Actions:"
        echo "  all       — prepare + submit (single mode)"
        echo "  both      — submit BOTH visual + semantic"
        echo "  submit    — submit SLURM array job"
        echo "  aggregate — merge by_qid results"
        echo "  status    — check job progress"
        echo ""
        echo "Modes: semantic (default), visual"
        echo "Partitions: gigabyte_a6000 (default)"
        echo ""
        echo "Examples:"
        echo "  bash 0227_slurm_lvbench.sh both                          # 둘 다"
        echo "  bash 0227_slurm_lvbench.sh all gigabyte_a6000 visual     # visual만"
        echo "  bash 0227_slurm_lvbench.sh all gigabyte_a6000 semantic   # semantic만"
        exit 1
        ;;
esac
