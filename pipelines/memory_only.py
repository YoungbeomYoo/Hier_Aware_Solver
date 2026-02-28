"""
Memory-Only Pipeline — Text-only inference from memory

Flow: Memory → MemoryContextFormatter → (optional: QueryDecomposer) → SimpleVLM → AnswerParser

가장 기본적인 baseline. 프레임 로딩 없이 memory text만으로 답변.
"""

from pipelines.base import BasePipeline
from components.answer_parser import extract_choice_letter, parse_response_index


class MemoryOnlyPipeline(BasePipeline):
    """Memory-only solver pipeline.

    Required components:
        - formatter: MemoryContextFormatter
        - simple_vlm: SimpleVLM (text-only)

    Optional components:
        - decomposer: QueryDecomposer (cue extraction for context filtering)
    """

    def solve(self, question_data: dict, memory: dict, video_id: str) -> dict:
        formatter = self.components["formatter"]
        simple_vlm = self.components["simple_vlm"]

        question = question_data["question"]
        options = question_data["options"]
        answer = question_data.get("answer")

        # Format memory as flat text
        memory_text = formatter.format_flat(memory, max_chars=12000)

        # Format subtitles if available
        subtitles = question_data.get("subtitles", "")

        # Build prompt
        opt_text = "\n".join(options)
        prompt = f"""This video's subtitles are listed below:
{subtitles if subtitles else "(No subtitles provided.)"}

This video's memory (summaries of key segments with timestamps) is listed below:
{memory_text if memory_text else "(No memory provided.)"}

Select the best answer to the following multiple-choice question based on the video information above.
Respond with only the letter (A, B, C, or D) of the correct option.

Question: {question}
Options:
{opt_text}

The best answer is:
"""

        # Inference
        response = simple_vlm.infer(prompt, max_new_tokens=2)
        pred = extract_choice_letter(response)

        # Check correctness
        correct = self.adapter.check_correct(pred, answer)

        return {
            "pred": pred,
            "answer": answer,
            "correct": correct,
            "method": "memory_only",
            "raw_response": response,
            "memory_chars": len(memory_text),
        }
