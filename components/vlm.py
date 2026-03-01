"""
VLM Inference — Text-only LLM + Vision VLM 모듈

- TextOnlyLLM: 프레임 없이 Qwen3-VL을 텍스트 LLM으로 사용
- VisionVLM: 프레임 + 메모리 컨텍스트로 VLM inference
- SimpleVLM: 간단한 text-only VLM (Video-MME baseline)
- GeminiLLM: Google Gemini API 기반 text-only LLM
- APIBasedLLM: 범용 API 기반 LLM (DeepSeek 등)
"""

import gc
import os
import torch
from PIL import Image
from components.json_extractor import extract_json


class TextOnlyLLM:
    """Qwen3-VL을 텍스트 전용 LLM으로 사용.

    JSON 포맷 출력을 강제하는 system prompt 포함.
    모든 LLM 기반 컴포넌트의 llm_fn으로 주입됨.
    """

    def __init__(self, model, processor, system_prompt: str | None = None):
        self.model = model
        self.processor = processor
        self.system_prompt = system_prompt or "You are a logical AI assistant that outputs strictly in JSON format."

    def reason(self, prompt_text: str, max_tokens: int = 256) -> dict:
        """텍스트 전용 inference → JSON dict 반환."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": [{"type": "text", "text": prompt_text}]},
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = self.processor(
            text=text, images=None, videos=None, return_tensors="pt"
        ).to(self.model.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=False
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, output_ids)
            ]
            response = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True
            )[0].strip()

        del inputs, output_ids, generated_ids_trimmed
        torch.cuda.empty_cache()

        parsed = extract_json(response)
        return parsed if parsed else {}

    def __call__(self, prompt_text: str, max_tokens: int = 256) -> dict:
        """llm_fn interface — 다른 컴포넌트에서 callable로 사용."""
        return self.reason(prompt_text, max_tokens)


class VisionVLM:
    """Qwen3-VL Vision inference.

    프레임 + 메모리 컨텍스트 + 질문으로 답변 생성.
    prompt_template으로 시스템 프롬프트와 유저 프롬프트 커스터마이징 가능.
    """

    DEFAULT_SYS_PROMPT = (
        "You are an expert video analyzer. "
        "Analyze the video frames and memory context to answer the multiple choice question. "
        "Output ONLY valid JSON with keys: answer, confidence, observation."
    )

    DEFAULT_USER_TEMPLATE = """{history_text}{frame_intro}

[Video Memory Context]
{memory_context}

Question: {question}
Options:
{options_text}

Select the best answer. Output ONLY valid JSON:
{{"answer": "<A, B, C, or D>", "confidence": "high", "observation": "Key visual details that support the answer"}}
- "high": Clear visual/textual evidence supports your answer.
- "low": Uncertain, describe what additional information would help."""

    def __init__(self, model, processor,
                 sys_prompt: str | None = None,
                 user_template: str | None = None,
                 image_token_size: int = 256,
                 memory_budget: int = 20000):
        self.model = model
        self.processor = processor
        self.sys_prompt = sys_prompt or self.DEFAULT_SYS_PROMPT
        self.user_template = user_template or self.DEFAULT_USER_TEMPLATE
        self.image_token_size = image_token_size
        self.memory_budget = memory_budget

    def infer(self, frames_np, memory_context: str,
              question: str, options: list[str],
              hop_history: list[dict] | None = None,
              max_tokens: int = 200) -> dict:
        """Vision VLM inference.

        Args:
            frames_np: numpy array (N, H, W, 3)
            memory_context: formatted memory text
            question: question text
            options: list of option strings
            hop_history: previous hop observations (optional)
            max_tokens: max generation tokens

        Returns:
            {"answer": str, "confidence": str, "observation": str, "raw_response": str}
        """
        from qwen_vl_utils import process_vision_info

        opt_text = "\n".join(options)

        # Build history text
        history_text = ""
        if hop_history:
            history_text = "[Previous Observations]\n"
            for h in hop_history:
                if h.get("type") == "frame_inference" and h.get("observation"):
                    history_text += (
                        f"Hop {h['hop']}: {h.get('type', 'unknown')}\n"
                        f"  Observation: {h.get('observation', 'N/A')}\n"
                        f"  Previous answer: {h.get('answer', 'N/A')} "
                        f"(confidence: {h.get('confidence', 'N/A')})\n\n"
                    )
            history_text += "Based on ALL observations so far, answer the question.\n\n"

        # Truncate memory
        if len(memory_context) > self.memory_budget:
            memory_context = memory_context[:self.memory_budget] + "\n... [truncated for memory budget]"

        # Build frame list
        frame_list = [Image.fromarray(f) for f in frames_np]
        pixel_count = self.image_token_size * 32 * 32

        # Build user content
        user_content = []
        if history_text:
            user_content.append({"type": "text", "text": history_text})

        frame_intro = f"The following {len(frame_list)} frames are from the relevant video segments:"
        user_content.append({"type": "text", "text": frame_intro})
        for img in frame_list:
            user_content.append({
                "type": "image", "image": img,
                "min_pixels": pixel_count, "max_pixels": pixel_count,
            })

        user_content.append({"type": "text", "text": f"\n[Video Memory Context]\n{memory_context}"})
        user_content.append({"type": "text", "text": (
            f"\nQuestion: {question}\nOptions:\n{opt_text}\n\n"
            "Select the best answer. Output ONLY valid JSON:\n"
            '{"answer": "B", "confidence": "high", "observation": "Key visual details"}\n'
            '- "high": Clear visual/textual evidence supports your answer.\n'
            '- "low": Uncertain, describe what additional information would help.'
        )})

        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.sys_prompt}]},
            {"role": "user", "content": user_content},
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos_vis = process_vision_info(messages, image_patch_size=16)
        inputs = self.processor(
            text=text, images=images, videos=videos_vis,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=False,
                use_cache=True, top_p=None, num_beams=1,
                top_k=None, temperature=None,
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, output_ids)
            ]
            response = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True
            )[0].strip()

        del inputs, output_ids, generated_ids_trimmed, user_content, frame_list, images, videos_vis
        torch.cuda.empty_cache()
        gc.collect()

        parsed = extract_json(response)

        import re
        answer_str = parsed.get("answer", "")
        confidence = parsed.get("confidence", "low")
        observation = parsed.get("observation", "")

        if not answer_str:
            answer_str = response.strip()
            confidence = "low"
        if confidence not in ("high", "low"):
            confidence = "low"

        return {
            "answer": answer_str,
            "confidence": confidence,
            "observation": observation,
            "raw_response": response,
        }


class SimpleVLM:
    """Simple text-only VLM (Video-MME baseline style).

    프레임 없이 memory text + question으로 답변 생성.
    max_new_tokens=2로 A/B/C/D만 출력.
    """

    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    def infer(self, prompt: str, max_new_tokens: int = 2) -> str:
        """Simple text-only inference.

        Returns:
            Raw response string.
        """
        messages = [
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], return_tensors="pt")

        for k in inputs:
            if torch.is_tensor(inputs[k]):
                inputs[k] = inputs[k].to(self.model.device)

        with torch.no_grad():
            gen_ids = self.model.generate(
                **inputs,
                do_sample=False, top_p=None, num_beams=1,
                use_cache=True, max_new_tokens=max_new_tokens,
                top_k=None, temperature=None,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, gen_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0].strip()

        del inputs, gen_ids, generated_ids_trimmed
        torch.cuda.empty_cache()

        return response


class GeminiLLM:
    """Google Gemini API 기반 LLM.

    TextOnlyLLM과 동일한 인터페이스 — llm_fn으로 주입 가능.
    API key는 환경변수 GEMINI_API_KEY 또는 API_KEY에서 로딩.
    """

    def __init__(self, model_name: str = "gemini-2.0-flash"):
        import google.generativeai as genai
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("API_KEY", "")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def reason(self, prompt_text: str, max_tokens: int = 256) -> dict:
        """Gemini API로 text-only inference → JSON dict."""
        response = self.model.generate_content(
            prompt_text,
            generation_config={"max_output_tokens": max_tokens, "temperature": 0},
        )
        text = response.text.strip()
        parsed = extract_json(text)
        return parsed if parsed else {}

    def __call__(self, prompt_text: str, max_tokens: int = 256) -> dict:
        return self.reason(prompt_text, max_tokens)


class APIBasedLLM:
    """범용 OpenAI-compatible API 기반 LLM.

    DeepSeek, vLLM, Ollama 등 OpenAI API 호환 서버에 사용.
    TextOnlyLLM과 동일한 인터페이스.
    """

    def __init__(self, base_url: str, api_key: str = "",
                 model: str = "default", system_prompt: str | None = None):
        try:
            from openai import OpenAI
            self.client = OpenAI(base_url=base_url, api_key=api_key or "dummy")
        except ImportError:
            raise ImportError("openai package required. pip install openai")
        self.model = model
        self.system_prompt = system_prompt or "You are a logical AI assistant that outputs strictly in JSON format."

    def reason(self, prompt_text: str, max_tokens: int = 256) -> dict:
        """API 호출 → JSON dict."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt_text},
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0,
        )
        text = response.choices[0].message.content.strip()
        parsed = extract_json(text)
        return parsed if parsed else {}

    def __call__(self, prompt_text: str, max_tokens: int = 256) -> dict:
        return self.reason(prompt_text, max_tokens)
