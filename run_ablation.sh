#!/bin/bash
# ============================================================
# Ablation Experiment Runner — 130-question subset (array job)
#
# 문제별 병렬 실행으로 GPU 130개까지 동시 사용 가능.
#
# Usage:
#   # 전체 5개 실험 한번에 제출
#   bash run_ablation.sh all
#
#   # 특정 실험만 제출
#   bash run_ablation.sh videolucy_prompt
#   bash run_ablation.sh strict_accumulate
#   bash run_ablation.sh videolucy_accumulate
#   bash run_ablation.sh text_only
#   bash run_ablation.sh twostage
#
#   # 결과 확인 (모든 실험 비교)
#   bash run_ablation.sh compare
#
#   # 결과 모으기 (aggregate)
#   bash run_ablation.sh aggregate <name>
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

SUBSET_IDS="output/video_mme_full_visual/v1/ablation_subset_ids.txt"
FULL_QID_TSV="output/video_mme_full_visual/v1/question_list.tsv"
CONFIG_DIR="config/ablation"

# Singularity container
SIMG="/scratch2/youngbeom/simg/acl2026.simg"

# ① baseline_exact 는 이미 결과 복사 완료 (output/ablation_baseline_exact/v1/)
EXPERIMENTS_R1="videolucy_prompt strict_accumulate videolucy_accumulate text_only twostage"
EXPERIMENTS_R2="text_only_compact text_only_strict no_caption"
EXPERIMENTS_R3="no_query llm_select"
EXPERIMENTS_R4="always_visual always_visual_acc"
EXPERIMENTS_R5="strict_twostage strict_always_visual"
EXPERIMENTS_R6="relaxed_text_only relaxed_visual relaxed_twostage relaxed_always_visual answerjudge_text_only answerjudge_visual answerjudge_twostage answerjudge_always_visual"
EXPERIMENTS_R7="c3_compact no_filter no_filter_compact visual_relaxed_rejudge"
EXPERIMENTS_R8="coarse_first"
EXPERIMENTS_R9="visual_enrich"
EXPERIMENTS="$EXPERIMENTS_R1 $EXPERIMENTS_R2 $EXPERIMENTS_R3 $EXPERIMENTS_R4 $EXPERIMENTS_R5 $EXPERIMENTS_R6 $EXPERIMENTS_R7 $EXPERIMENTS_R8 $EXPERIMENTS_R9"

# ============================================================
# Shared QID list: subset 130개의 qid→vid TSV 생성
# ============================================================
SHARED_QID_LIST="output/ablation_subset_qid_list.tsv"

prepare_shared() {
    echo "=== Preparing shared QID list (130-question subset) ==="

    if [ ! -f "$FULL_QID_TSV" ]; then
        echo "[ERROR] Full question list not found: $FULL_QID_TSV"
        echo "  Run the full visual experiment first to generate it."
        return 1
    fi

    python3 -c "
subset = set()
with open('$SUBSET_IDS') as f:
    for line in f:
        qid = line.strip()
        if qid:
            subset.add(qid)

found = 0
with open('$FULL_QID_TSV') as fin, open('$SHARED_QID_LIST', 'w') as fout:
    for line in fin:
        parts = line.strip().split('\t')
        if len(parts) == 2 and parts[0] in subset:
            fout.write(line)
            found += 1

print(f'  Filtered {found}/{len(subset)} questions → $SHARED_QID_LIST')
if found != len(subset):
    missing = subset - set(open('$SHARED_QID_LIST').read().split())
    print(f'  WARNING: {len(subset) - found} questions not found in full TSV')
"
    N_QUESTIONS=$(wc -l < "$SHARED_QID_LIST")
    echo "  Total: ${N_QUESTIONS} questions ready"
    echo ""
}

# ============================================================
# Submit: array job으로 문제별 병렬 실행
# ============================================================
submit_one() {
    local NAME="$1"
    local TIME_LIMIT="${2:-01:00:00}"  # default 1h, D3 needs 3h
    local CONFIG="${CONFIG_DIR}/${NAME}.yaml"
    local OUTPUT_DIR="./output/ablation_${NAME}/v1"

    if [ ! -f "$CONFIG" ]; then
        echo "[ERROR] Config not found: $CONFIG"
        return 1
    fi

    # Shared QID list 없으면 생성
    if [ ! -f "$SHARED_QID_LIST" ]; then
        prepare_shared
    fi

    local N_QUESTIONS
    N_QUESTIONS=$(wc -l < "$SHARED_QID_LIST")

    echo "=== Submitting: ${NAME} (${N_QUESTIONS} array tasks) ==="
    echo "  Config:    ${CONFIG}"
    echo "  Output:    ${OUTPUT_DIR}"
    echo "  Container: ${SIMG}"

    mkdir -p "${OUTPUT_DIR}/logs" "${OUTPUT_DIR}/by_qid"

    sbatch <<EOF
#!/bin/bash
############ Settings for sbatch #############
#SBATCH --job-name=abl_${NAME}
#SBATCH --partition=suma_a6000,gigabyte_a6000,tyan_a6000,asus_6000ada
#SBATCH --qos=big_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=${TIME_LIMIT}
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
LINE=\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" ${SCRIPT_DIR}/${SHARED_QID_LIST})
QID=\$(echo "\$LINE" | cut -f1)
VID=\$(echo "\$LINE" | cut -f2)

echo "CUDA_VISIBLE_DEVICES: \$CUDA_VISIBLE_DEVICES"
echo "SLURM_NODELIST: \$SLURM_NODELIST"
echo "=== abl_${NAME} | Q=\${QID} V=\${VID} | task \${SLURM_ARRAY_TASK_ID}/${N_QUESTIONS} ==="

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
# Utilities
# ============================================================
compare_results() {
    echo "=== Ablation Results Comparison ==="
    echo ""
    python3 compare_ablation.py
}

aggregate_one() {
    local NAME="$1"
    local OUTPUT_DIR="./output/ablation_${NAME}/v1"
    echo "=== Aggregating: ${NAME} ==="
    python aggregate.py --output_dir "$OUTPUT_DIR"
}

status_all() {
    echo "=== Ablation Job Status ==="
    squeue -u $USER -n "$(echo $EXPERIMENTS | sed 's/ /,abl_/g; s/^/abl_/')" \
        -o "%.10i %.9P %.16j %.8T %.10M %.6D %R" 2>/dev/null || true
    echo ""
    echo "=== Progress ==="
    for exp in baseline_exact $EXPERIMENTS; do
        local dir="./output/ablation_${exp}/v1/by_qid"
        if [ -d "$dir" ]; then
            local done=$(ls "$dir"/*.json 2>/dev/null | wc -l)
            printf "  %-25s %3d/130\n" "$exp" "$done"
        else
            printf "  %-25s (no results)\n" "$exp"
        fi
    done
}

# ============================================================
# Main
# ============================================================
case "${1:-all}" in
    all)
        echo "============================================"
        echo "  Submitting ALL ablation experiments"
        echo "  (baseline_exact already done)"
        echo "  Container: ${SIMG}"
        echo "  Mode: per-question array jobs"
        echo "============================================"
        echo ""
        prepare_shared
        for exp in $EXPERIMENTS; do
            submit_one "$exp"
        done
        echo "============================================"
        echo "  Monitor: bash run_ablation.sh status"
        echo "  Results: bash run_ablation.sh compare"
        echo "============================================"
        ;;
    r2)
        echo "============================================"
        echo "  Submitting Round 2 experiments"
        echo "  Container: ${SIMG}"
        echo "============================================"
        echo ""
        prepare_shared
        for exp in $EXPERIMENTS_R2; do
            submit_one "$exp"
        done
        echo "============================================"
        echo "  Round 2: 3 × 130 = 390 array tasks submitted!"
        echo "  Monitor: bash run_ablation.sh status"
        echo "  Results: bash run_ablation.sh compare"
        echo "============================================"
        ;;
    r3)
        echo "============================================"
        echo "  Submitting Round 3 experiments (A0 + B3)"
        echo "  Container: ${SIMG}"
        echo "============================================"
        echo ""
        prepare_shared
        for exp in $EXPERIMENTS_R3; do
            submit_one "$exp"
        done
        echo "============================================"
        echo "  Round 3: 2 × 130 = 260 array tasks submitted!"
        echo "  Monitor: bash run_ablation.sh status"
        echo "  Results: bash run_ablation.sh compare"
        echo "============================================"
        ;;
    r4)
        echo "============================================"
        echo "  Submitting Round 4 experiments (D3 always visual)"
        echo "  Container: ${SIMG}"
        echo "  Time limit: 3h (VLM every hop)"
        echo "============================================"
        echo ""
        prepare_shared
        for exp in $EXPERIMENTS_R4; do
            submit_one "$exp" "03:00:00"
        done
        echo "============================================"
        echo "  Round 4: 2 × 130 = 260 array tasks submitted!"
        echo "  Monitor: bash run_ablation.sh status"
        echo "  Results: bash run_ablation.sh compare"
        echo "============================================"
        ;;
    r5)
        echo "============================================"
        echo "  Submitting Round 5: C0 missing cells"
        echo "  Container: ${SIMG}"
        echo "============================================"
        echo ""
        prepare_shared
        submit_one "strict_twostage" "01:30:00"
        submit_one "strict_always_visual" "03:00:00"
        echo "============================================"
        echo "  Round 5: 2 × 130 = 260 array tasks submitted!"
        echo "============================================"
        ;;
    r6)
        echo "============================================"
        echo "  Submitting Round 6: C1(relaxed) + C3(VL+AJ) full grid"
        echo "  C1×{D0,D1,D2,D3} + C3×{D0,D1,D2,D3} = 8 experiments"
        echo "  Container: ${SIMG}"
        echo "============================================"
        echo ""
        prepare_shared
        # C1 (relaxed) × D
        submit_one "relaxed_text_only" "01:00:00"
        submit_one "relaxed_visual" "01:00:00"
        submit_one "relaxed_twostage" "01:30:00"
        submit_one "relaxed_always_visual" "03:00:00"
        # C3 (VL+AJ) × D
        submit_one "answerjudge_text_only" "01:00:00"
        submit_one "answerjudge_visual" "01:00:00"
        submit_one "answerjudge_twostage" "01:30:00"
        submit_one "answerjudge_always_visual" "03:00:00"
        echo "============================================"
        echo "  Round 6: 8 × 130 = 1040 array tasks submitted!"
        echo "  Monitor: bash run_ablation.sh status"
        echo "  Results: bash run_ablation.sh compare"
        echo "============================================"
        ;;
    r7)
        echo "============================================"
        echo "  Submitting Round 7: 미검증 축 검증"
        echo "  c3_compact, no_filter, no_filter_compact, visual_relaxed_rejudge"
        echo "  Container: ${SIMG}"
        echo "============================================"
        echo ""
        prepare_shared
        submit_one "c3_compact" "01:00:00"
        submit_one "no_filter" "01:00:00"
        submit_one "no_filter_compact" "01:00:00"
        submit_one "visual_relaxed_rejudge" "01:30:00"
        echo "============================================"
        echo "  Round 7: 4 × 130 = 520 array tasks submitted!"
        echo "  Monitor: bash run_ablation.sh status"
        echo "  Results: bash run_ablation.sh compare"
        echo "============================================"
        ;;
    r8)
        echo "============================================"
        echo "  Submitting Round 8: Phase 0 (Coarse-First)"
        echo "  G1: Level_2+Level_3 전체 overview로 1차 답변"
        echo "  Container: ${SIMG}"
        echo "============================================"
        echo ""
        prepare_shared
        submit_one "coarse_first" "01:00:00"
        echo "============================================"
        echo "  Round 8: 1 × 130 = 130 array tasks submitted!"
        echo "  Monitor: bash run_ablation.sh status"
        echo "  Results: bash run_ablation.sh compare"
        echo "============================================"
        ;;
    r9)
        echo "============================================"
        echo "  Submitting Round 9: D5 Visual Enrichment"
        echo "  C3+D5+F1+G1: hop loop에서 visual caption → C3 rejudge"
        echo "  Container: ${SIMG}"
        echo "============================================"
        echo ""
        prepare_shared
        submit_one "visual_enrich" "01:00:00"
        echo "============================================"
        echo "  Round 9: 1 × 130 = 130 array tasks submitted!"
        echo "  Monitor: bash run_ablation.sh status"
        echo "  Results: bash run_ablation.sh compare"
        echo "============================================"
        ;;
    prepare)
        prepare_shared
        ;;
    status)
        status_all
        ;;
    compare)
        compare_results
        ;;
    aggregate)
        if [ -z "$2" ]; then
            echo "Usage: bash run_ablation.sh aggregate <name>"
            exit 1
        fi
        aggregate_one "$2"
        ;;
    baseline_exact)
        echo "baseline_exact already has results (copied from full run)."
        echo "  → output/ablation_baseline_exact/v1/by_qid/ (130 files)"
        ;;
    *)
        # Shared QID list 없으면 생성
        if [ ! -f "$SHARED_QID_LIST" ]; then
            prepare_shared
        fi
        submit_one "$1"
        ;;
esac
