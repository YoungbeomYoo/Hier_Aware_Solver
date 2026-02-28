# Experiment Status

## Full Run Results (900Q, sorted by accuracy)

| # | Config | Axes | Accuracy | vs Best | vs VL | Status |
|---|--------|------|----------|---------|-------|--------|
| 1 | **R10a Flat Baseline** | Flat (no tree) | **54.8%** | **+1.6%p** | +2.9%p | Done |
| 2 | C0 (Strict Judge) | A1 B1 C0 D0 F1 G1 | 53.9% | +0.7%p | +2.0%p | Done |
| 3 | R10b No KE | -KE | 53.4% | +0.2%p | +1.5%p | Done |
| 4 | R10c No Caption | -Cap | 53.4% | +0.2%p | +1.6%p | Done |
| 5 | **Best** | A1 B1 C3 D0 F1 G1 | **53.2%** | — | +1.3%p | Done |
| 6 | C2 (-AJ) | A1 B1 C2 D0 F1 G1 | 53.2% | +0.0%p | +1.3%p | Done |
| 7 | -B1 (No Filter) | A1 B0 C3 D0 F1 G1 | 53.1% | -0.1%p | +1.2%p | Done |
| 8 | D5 (Visual) | A1 B1 C3 D5 F1 G1 | 52.9% | -0.3%p | +1.0%p | Done |
| 9 | -F1 (No History) | A1 B1 C3 D0 F0 G1 | 52.9% | -0.3%p | +1.0%p | Done |
| 10 | -A1 (Rule Query) | A0 B1 C3 D0 F1 G1 | 52.8% | -0.4%p | +0.9%p | Done |
| 11 | VideoLucy | (baseline) | 51.9% | -1.3%p | — | Baseline |
| 12 | -G1 (F0, No Phase0) | A1 B1 C3 D0 F0 G0 | 47.7% | -5.5%p | -4.2%p | Done |
| 13 | -G1 (F1, No Phase0) | A1 B1 C3 D0 F1 G0 | 47.4% | -5.8%p | -4.4%p | Done |

## Key Findings

1. **Phase 0(G1)이 유일한 대형 기여자**: +5.8%p, 나머지 ±0.4%p
2. **Flat > Best**: 54.8% > 53.2%, 트리 탐색이 오히려 성능 하락
3. **Hop loop이 병목**: Phase 0 = 54.8%, Hop = 39.4%, 가중평균 = 53.2%
4. **C0 > C3 in hop loop**: C0 hop 45.7% vs C3 hop 39.4%
5. **Caption/KE 불필요**: R10b(-KE)=53.4%, R10c(-Cap)=53.4% — 제거해도 동일/개선
6. **성능 원천 = Phase 0 coarse overview**: 전체 요약 → 1회 판단이 핵심
7. **Tree 구축은 필수, Tree 탐색은 문제**: Stage 2 summary가 근본, hop loop이 병목

## 카테고리별 Flat vs Best (hierarchical의 장단점)

Best > Flat (hop loop이 유리):
- **Counting**: Best 45.8% vs Flat 39.6% (+6.2%p) — 특정 시간대 반복 관찰
- **Temporal Reasoning**: Best 50.5% vs Flat 48.4% (+2.1%p) — 시간 순서 탐색
- **Action Recognition**: Best 49.2% vs Flat 47.6% (+1.6%p)

Flat > Best (전체 맥락이 중요):
- **Info Synopsis**: Flat 76.7% vs Best 69.9% (+6.7%p) — 전체 흐름 파악
- **Object Reasoning**: Flat 50.8% vs Best 49.2% (+1.7%p)
- **Action Reasoning**: Flat 49.4% vs Best 48.3% (+1.1%p)

## R10 Final Results (ALL DONE)
- R10a Flat Baseline: **54.8%** (+1.6%p) ✅
- R10b No KE: **53.4%** (+0.2%p) ✅
- R10c No Caption: **53.4%** (+0.2%p) ✅

## Hop Loop 실패 원인 (Tree Reconstruction 분석)
- 8개 실패 사례 분석: 10개 leaf (300s) = 30~55분 영상의 9~16%만 커버
- Keyword 기반 tree filter가 의미적으로 중요한 구간을 정확히 못 찾음
- Counting: 산발적 선택 (max_gap 660~1560s), 연속 관찰 불가
- 핵심: "좁고 깊게(hop)" < "넓고 얕게(flat)" — navigation 정확도 부족
- 원본: agent_comparison.md Section 6.9
