"""
Information Aggregator — Vgent-style Structured Reasoning

Aggregates sub-question verification results across segments
into a concise summary for final answer generation.
"""

import json


class InfoAggregator:
    """Aggregate sub-question answers across segments.

    Args:
        llm_fn: callable(prompt, max_tokens) -> dict
        prompt_template: prompt with {question}, {candidates}, {information} placeholders
    """

    def __init__(self, llm_fn, prompt_template: str):
        self.llm_fn = llm_fn
        self.prompt_template = prompt_template

    def aggregate(self, sub_questions: dict, check_results: dict,
                  leaves: list[dict], selected_indices: list[int],
                  question: str, options: list[str]) -> str:
        """Aggregate verification results into summary.

        Args:
            sub_questions: {"Q1": "Is there...?", ...}
            check_results: {leaf_idx: {"Q1": "yes", ...}, ...}
            leaves: list of all leaf dicts
            selected_indices: indices of positively-verified leaves
            question: original question
            options: list of option strings

        Returns:
            summary string (concise, ~20 words)
        """
        # Build information string: "segment [0] (0s-30s): Q1=yes, Q2=no; ..."
        info_parts = []
        for leaf_idx in selected_indices:
            answers = check_results.get(leaf_idx, {})
            if not answers:
                continue

            leaf = leaves[leaf_idx]
            st = float(leaf.get("start_time", 0))
            et = float(leaf.get("end_time", 0))

            ans_str = ", ".join(
                f"{k}={v}" for k, v in sorted(answers.items())
            )
            info_parts.append(f"segment [{leaf_idx}] ({st:.0f}s-{et:.0f}s): {ans_str}")

        if not info_parts:
            return "No relevant information found in verified segments."

        information = "\n".join(info_parts)

        # Add sub-question definitions for context
        sq_str = "\n".join(f"{k}: {v}" for k, v in sorted(sub_questions.items()))
        information = f"Sub-questions:\n{sq_str}\n\nResults:\n{information}"

        candidates_str = "\n".join(
            f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)
        )

        prompt = self.prompt_template.format(
            question=question,
            candidates=candidates_str,
            information=information,
        )

        try:
            result = self.llm_fn(prompt, max_tokens=256)
        except Exception as e:
            print(f"    [Aggregate] Error: {e}")
            return "Aggregation failed."

        if isinstance(result, dict):
            summary = result.get("summary", "")
            if summary:
                print(f"    [Aggregate] Summary: {summary}")
                return summary

        print(f"    [Aggregate] Unexpected result: {result}")
        return str(result) if result else "No aggregation result."
