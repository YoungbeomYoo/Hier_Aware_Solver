#!/usr/bin/env python3
"""
세 실험의 노드 선택 품질 비교 (semantic vs visual)

선택된 target_leaves의 caption이 질문/cues와 얼마나 관련있는지 분석.
- 단어 겹침 기반 relevance score
- 선택된 시간 구간 비교
- 정답률 차이

Usage:
    python compare_node_quality.py \
        output/video_mme_subset_tree_search_semantic/v2 \
        output/video_mme_subset_tree_search_visual3/v1 \
        output/video_mme_subset_tree_search_visual2

    python compare_node_quality.py DIR_A DIR_B DIR_C --qid 604-1
"""

import os
import json
import argparse
import re
from collections import defaultdict


def load_results(result_dir):
    by_qid_dir = os.path.join(result_dir, "by_qid")
    if not os.path.isdir(by_qid_dir):
        return {}
    results = {}
    for fname in sorted(os.listdir(by_qid_dir)):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(by_qid_dir, fname)) as f:
                r = json.load(f)
            qid = r.get("question_id", fname.replace(".json", ""))
            results[qid] = r
        except Exception:
            pass
    return results


def get_question_keywords(result):
    """질문에서 핵심 키워드 추출."""
    cues = result.get("cues", [])
    if cues:
        return cues

    # plan_info에서 추출
    plan = result.get("plan_info", {})
    if isinstance(plan, dict):
        cues = plan.get("cues", [])
        if cues:
            return cues

    return []


def extract_target_leaves(result):
    """모든 hop의 target_leaves 추출."""
    leaves = []
    for hc in result.get("hop_contexts", []):
        hop = hc.get("hop", "?")
        for leaf in hc.get("target_leaves", []):
            leaves.append({
                "hop": hop,
                "start": float(leaf.get("start_time", 0)),
                "end": float(leaf.get("end_time", 0)),
                "caption": leaf.get("caption", "") or "",
            })
    return leaves


def tokenize(text):
    """간단한 토큰화 — 소문자 + 알파벳/숫자 단어."""
    return set(re.findall(r'[a-z0-9]+', text.lower()))


STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to",
    "of", "and", "or", "for", "with", "from", "by", "it", "its", "this",
    "that", "these", "those", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might",
    "shall", "can", "not", "no", "but", "if", "then", "than", "so", "as",
    "video", "scene", "screen", "shows", "appears", "seen", "displayed",
    "image", "frame", "shot", "clip", "featuring", "where", "while",
    "which", "what", "who", "how", "when", "sequence", "captures",
}


def compute_relevance(cues, caption):
    """cue 키워드와 caption의 단어 겹침 기반 relevance score."""
    if not cues or not caption:
        return 0.0

    cue_tokens = set()
    for cue in cues:
        cue_tokens |= tokenize(str(cue))
    cue_tokens -= STOP_WORDS

    cap_tokens = tokenize(caption) - STOP_WORDS

    if not cue_tokens:
        return 0.0

    overlap = cue_tokens & cap_tokens
    return len(overlap) / len(cue_tokens) if cue_tokens else 0


def format_time(sec):
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def analyze_semantic_match_quality(result):
    """semantic_match의 top_matches 품질 분석."""
    sm = result.get("semantic_match", {})
    if not sm:
        return None

    top_scores = sm.get("top_scores", [])
    if not top_scores:
        return None

    bad_matches = []
    good_matches = []

    for entry in top_scores[:5]:
        for m in entry.get("top_matches", []):
            q_el = m.get("q_element", "")
            matched = m.get("matched_to", "")
            sim = m.get("similarity", 0)

            # 간단한 품질 판단: 단어 겹침이 있으면 good
            q_tokens = tokenize(q_el) - STOP_WORDS
            m_tokens = tokenize(matched) - STOP_WORDS
            overlap = q_tokens & m_tokens

            if sim > 0.99 and not overlap:
                bad_matches.append({
                    "q": q_el[:40], "m": matched[:40], "sim": sim,
                    "l1_idx": entry.get("idx"),
                })
            elif overlap:
                good_matches.append({
                    "q": q_el[:40], "m": matched[:40], "sim": sim,
                    "l1_idx": entry.get("idx"),
                })

    return {
        "total_matches": sum(len(e.get("top_matches", [])) for e in top_scores[:5]),
        "good": good_matches,
        "bad": bad_matches,
        "selected_l1": sm.get("selected_l1", []),
    }


def print_question_detail(qid, results_map, labels):
    """한 문제에 대한 세 실험 비교."""
    print(f"\n{'='*70}")
    print(f"  Question: {qid}")

    # 질문/cues 출력 (첫 번째 가용한 결과에서)
    for label in labels:
        r = results_map.get(label, {})
        if r:
            cues = get_question_keywords(r)
            vid = r.get("video_id", "?")
            print(f"  Video: {vid}")
            if cues:
                print(f"  Cues: {cues}")
            # question from plan_info
            pi = r.get("plan_info", {})
            if isinstance(pi, dict) and pi.get("original_question"):
                q = pi["original_question"]
                print(f"  Q: {q[:120]}")
            break

    print()

    # 각 실험 결과 비교
    for label in labels:
        r = results_map.get(label)
        if not r:
            print(f"  [{label}] — no result")
            continue

        correct = r.get("correct", False)
        pred = r.get("pred", "?")
        answer = r.get("answer", "?")
        mark = "O" if correct else "X"
        conf = r.get("confidence", "?")
        hops = r.get("total_hops", "?")

        print(f"  [{label}] {mark} pred={pred} gt={answer}  hops={hops} conf={conf}")

        # target leaves
        leaves = extract_target_leaves(r)
        cues = get_question_keywords(r)

        if leaves:
            # relevance score 계산
            relevances = []
            for leaf in leaves:
                rel = compute_relevance(cues, leaf["caption"])
                relevances.append(rel)

            avg_rel = sum(relevances) / len(relevances) if relevances else 0
            any_relevant = sum(1 for rel in relevances if rel > 0)

            print(f"    Segments: {len(leaves)}, Avg relevance: {avg_rel:.2f}, "
                  f"Any relevant: {any_relevant}/{len(leaves)}")

            # 상위 3개 + 하위 3개
            indexed = list(zip(relevances, leaves))
            indexed.sort(key=lambda x: -x[0])

            for i, (rel, leaf) in enumerate(indexed[:3]):
                cap = leaf["caption"][:70]
                print(f"    + hop{leaf['hop']} {format_time(leaf['start'])}-{format_time(leaf['end'])} "
                      f"rel={rel:.2f} | {cap}")

            # semantic match 품질 (semantic만)
            sm_info = analyze_semantic_match_quality(r)
            if sm_info:
                bad = sm_info["bad"]
                good = sm_info["good"]
                print(f"    Semantic match: {len(good)} good, {len(bad)} bad (high-sim but unrelated)")
                if bad[:2]:
                    for b in bad[:2]:
                        print(f"      BAD: '{b['q']}' → '{b['m']}' sim={b['sim']:.4f} (L1[{b['l1_idx']}])")
                if good[:2]:
                    for g in good[:2]:
                        print(f"      GOOD: '{g['q']}' → '{g['m']}' sim={g['sim']:.4f} (L1[{g['l1_idx']}])")

        print()


def main():
    parser = argparse.ArgumentParser(description="노드 선택 품질 비교")
    parser.add_argument("dirs", nargs="+", help="실험 디렉토리 (2-4개)")
    parser.add_argument("--qid", type=str, default=None, help="특정 문제만")
    parser.add_argument("--detail", action="store_true", help="모든 문제 개별 출력")
    parser.add_argument("--diff_only", action="store_true", help="결과가 다른 문제만")
    args = parser.parse_args()

    # 결과 로드
    all_results = {}
    labels = []
    for d in args.dirs:
        label = "/".join(d.rstrip("/").split("/")[-2:])
        labels.append(label)
        all_results[label] = load_results(d)
        print(f"  {label}: {len(all_results[label])} results")

    # 공통 qid
    common_qids = None
    for label in labels:
        qids = set(all_results[label].keys())
        common_qids = qids if common_qids is None else common_qids & qids
    common_qids = sorted(common_qids) if common_qids else []
    print(f"  Common: {len(common_qids)} questions\n")

    if args.qid:
        results_map = {label: all_results[label].get(args.qid) for label in labels}
        print_question_detail(args.qid, results_map, labels)
        return

    # === Aggregate 비교 ===
    stats = {label: {
        "correct": 0, "total": 0,
        "avg_relevance": [], "any_relevant_pct": [],
        "total_segments": [],
        "bad_semantic_matches": 0, "good_semantic_matches": 0,
    } for label in labels}

    for qid in common_qids:
        for label in labels:
            r = all_results[label].get(qid)
            if not r:
                continue

            s = stats[label]
            s["total"] += 1
            if r.get("correct"):
                s["correct"] += 1

            leaves = extract_target_leaves(r)
            cues = get_question_keywords(r)
            s["total_segments"].append(len(leaves))

            if leaves and cues:
                rels = [compute_relevance(cues, l["caption"]) for l in leaves]
                avg_r = sum(rels) / len(rels) if rels else 0
                any_r = sum(1 for rel in rels if rel > 0) / len(rels) if rels else 0
                s["avg_relevance"].append(avg_r)
                s["any_relevant_pct"].append(any_r)

            sm_info = analyze_semantic_match_quality(r)
            if sm_info:
                s["bad_semantic_matches"] += len(sm_info["bad"])
                s["good_semantic_matches"] += len(sm_info["good"])

    print(f"{'='*70}")
    print(f"  Node Selection Quality Comparison — {len(common_qids)} questions")
    print(f"{'='*70}")

    # 정답률
    print(f"\n  Accuracy:")
    for label in labels:
        s = stats[label]
        n = s["total"]
        c = s["correct"]
        acc = c / n * 100 if n else 0
        print(f"    {label:<45} {c}/{n} ({acc:.1f}%)")

    # Relevance
    print(f"\n  Keyword Relevance (cue words ∩ caption words):")
    print(f"    {'Experiment':<45} {'AvgRel':>7} {'%HasRel':>8} {'AvgSegs':>8}")
    print(f"    {'-'*68}")
    for label in labels:
        s = stats[label]
        avg_r = sum(s["avg_relevance"]) / len(s["avg_relevance"]) if s["avg_relevance"] else 0
        any_r = sum(s["any_relevant_pct"]) / len(s["any_relevant_pct"]) * 100 if s["any_relevant_pct"] else 0
        avg_seg = sum(s["total_segments"]) / len(s["total_segments"]) if s["total_segments"] else 0
        print(f"    {label:<45} {avg_r:>6.3f} {any_r:>7.1f}% {avg_seg:>7.1f}")

    # Semantic match quality (semantic만)
    has_sm = False
    for label in labels:
        s = stats[label]
        if s["bad_semantic_matches"] + s["good_semantic_matches"] > 0:
            has_sm = True
            break

    if has_sm:
        print(f"\n  Semantic Embedding Match Quality:")
        for label in labels:
            s = stats[label]
            good = s["good_semantic_matches"]
            bad = s["bad_semantic_matches"]
            total = good + bad
            if total:
                print(f"    {label}:")
                print(f"      Good matches (word overlap): {good}")
                print(f"      Bad matches (high sim, no overlap): {bad}")
                print(f"      Bad ratio: {bad/total*100:.1f}%")

    # Per-question: 어떤 실험이 더 나은 노드를 골랐는지
    print(f"\n  Per-question winner (highest avg relevance):")
    winners = defaultdict(int)
    for qid in common_qids:
        best_label = None
        best_rel = -1
        for label in labels:
            r = all_results[label].get(qid)
            if not r:
                continue
            leaves = extract_target_leaves(r)
            cues = get_question_keywords(r)
            if leaves and cues:
                rels = [compute_relevance(cues, l["caption"]) for l in leaves]
                avg_r = sum(rels) / len(rels) if rels else 0
                if avg_r > best_rel:
                    best_rel = avg_r
                    best_label = label
        if best_label:
            winners[best_label] += 1

    for label in labels:
        print(f"    {label:<45} wins {winners.get(label, 0)}/{len(common_qids)}")

    # Correctness + relevance 상관관계
    print(f"\n  Relevance when Correct vs Wrong:")
    for label in labels:
        correct_rels = []
        wrong_rels = []
        for qid in common_qids:
            r = all_results[label].get(qid)
            if not r:
                continue
            leaves = extract_target_leaves(r)
            cues = get_question_keywords(r)
            if leaves and cues:
                rels = [compute_relevance(cues, l["caption"]) for l in leaves]
                avg_r = sum(rels) / len(rels) if rels else 0
                if r.get("correct"):
                    correct_rels.append(avg_r)
                else:
                    wrong_rels.append(avg_r)
        avg_c = sum(correct_rels) / len(correct_rels) if correct_rels else 0
        avg_w = sum(wrong_rels) / len(wrong_rels) if wrong_rels else 0
        print(f"    {label}:")
        print(f"      Correct: avg_rel={avg_c:.3f} ({len(correct_rels)} qs)")
        print(f"      Wrong:   avg_rel={avg_w:.3f} ({len(wrong_rels)} qs)")

    # Detail output
    if args.detail or args.diff_only:
        for qid in common_qids:
            if args.diff_only:
                preds = set()
                for label in labels:
                    r = all_results[label].get(qid)
                    if r:
                        preds.add(r.get("correct", False))
                if len(preds) <= 1:
                    continue

            results_map = {label: all_results[label].get(qid) for label in labels}
            print_question_detail(qid, results_map, labels)


if __name__ == "__main__":
    main()
