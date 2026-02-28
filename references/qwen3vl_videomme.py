from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

import os
import json
import re
from glob import glob
from tqdm import tqdm
import torch

MEMORY_PATH = '/lustre/youngbeom/DyHiStreamMem/poc/results/Video-MME/videomme-long-stage1_30sec'
QUESTION_PATH = '/lustre/youngbeom/DyHiStreamMem/vqa/video-mme/split_per_duration/long'
OUTPUT_PATH = '/lustre/youngbeom/DyHiStreamMem/vqa/video-mme/output/30sec_flat_memory_only'
MODEL_PATH = '/scratch2/youngbeom/ckpt/Qwen3-VL-8B-Instruct'

# print("Loading model...")
# model = Qwen3VLForConditionalGeneration.from_pretrained(
#     MODEL_PATH,
#     dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )
# processor = AutoProcessor.from_pretrained(MODEL_PATH)
# print("Model loaded.")

# -------------------------
# Utils
# -------------------------
def safe_filename(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(s))

def read_json_any(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def iter_json_files(root):
    return sorted(glob(os.path.join(root, "**", "*.json"), recursive=True))

def build_memory_index(memory_root):
    """
    stage1_30sec memory json 예시:
    [{"start": 0.0, "end": 30.0, "caption": "...", "summary": "..."}, ...]
    파일명 stem을 video_id로 사용.
    """
    idx = {}
    for fp in tqdm(glob(os.path.join(memory_root, "*.json")), desc="Indexing memory"):
        try:
            vid = os.path.splitext(os.path.basename(fp))[0]
            segments = read_json_any(fp)
            idx[vid] = segments
        except Exception:
            continue
    return idx

def flatten_memory(memory_obj, max_chars=12000):
    """
    memory_obj: {"video_id":..., "memory": {"goal":..., "steps":[...]}}
    -> 텍스트로 flatten
    """
    if not memory_obj:
        return ""

    mem = memory_obj.get("memory", {})
    goal = mem.get("goal", "")
    steps = mem.get("steps", [])

    lines = []
    if goal:
        lines.append(f"[GOAL] {goal}")

    # steps 구조: [{"S01": {"step": "...", "substeps": [...] }}, ...]
    for step_block in steps:
        if not isinstance(step_block, dict):
            continue
        for sid, sval in step_block.items():
            step_name = ""
            substeps = []
            if isinstance(sval, dict):
                step_name = sval.get("step", "")
                substeps = sval.get("substeps", []) or []

            if step_name:
                lines.append(f"\n[{sid}] {step_name}")

            for ss in substeps:
                if not isinstance(ss, dict):
                    continue
                st = ss.get("start", None)
                ed = ss.get("end", None)
                txt = ss.get("substep", "") or ""
                # 너무 의미 없는 문장 제거(원하면 주석 처리)
                if "No substep sentences provided" in txt:
                    continue
                if st is not None and ed is not None:
                    lines.append(f"  - ({st:.2f}-{ed:.2f}s) {txt}")
                else:
                    lines.append(f"  - {txt}")

    out = "\n".join(lines).strip()
    if len(out) > max_chars:
        out = out[:max_chars] + "\n...[TRUNCATED]"
    return out

def get_subtitles_from_question(qobj):
    """
    Video-MME 쪽 question json에 subtitles가 들어있는 경우를 대비.
    (데이터마다 키가 다를 수 있어서 흔한 키들을 대응)
    """
    for k in ["subtitles", "subtitle", "subs", "captions", "caption"]:
        if k in qobj and qobj[k]:
            v = qobj[k]
            if isinstance(v, list):
                return "\n".join(map(str, v))
            return str(v)
    return ""  # 없으면 빈 문자열

def build_prompt(subtitles_text, memory_text, question, options):
    # options: ["A. ...", "B. ...", ...] 형태라고 가정
    opt_text = "\n".join(options)

    # “오직 A/B/C/D만” 강하게
    prompt = f"""This video's subtitles are listed below:
{subtitles_text if subtitles_text else "(No subtitles provided.)"}

This video's memory (summaries of key segments with timestamps) is listed below:
{memory_text if memory_text else "(No memory provided.)"}

Select the best answer to the following multiple-choice question based on the video information above.
Respond with only the letter (A, B, C, or D) of the correct option.

Question: {question}
Options:
{opt_text}

The best answer is:
"""
    return prompt

def extract_choice(text):
    """
    모델 출력에서 A/B/C/D만 추출
    """
    if not text:
        return None
    m = re.search(r"\b([ABCD])\b", text.strip().upper())
    return m.group(1) if m else None

# -------------------------
# Main
# -------------------------
def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    # out_file = os.path.join(OUTPUT_PATH, "predictions.jsonl")
    by_qid_dir = os.path.join(OUTPUT_PATH, "by_qid")
    os.makedirs(by_qid_dir, exist_ok=True)

    # (선택) 전체 jsonl도 동시에 남기고 싶으면 keep_jsonl=True로
    keep_jsonl = False
    jsonl_fp = os.path.join(OUTPUT_PATH, "predictions.jsonl")
    wf_jsonl = open(jsonl_fp, "w", encoding="utf-8") if keep_jsonl else None

    print("Loading model...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    print("Model loaded.")

    mem_index = build_memory_index(MEMORY_PATH)

    q_files = iter_json_files(QUESTION_PATH)
    print(f"Found {len(q_files)} question json files.")

    for qfp in tqdm(q_files, desc="Solving"):
        try:
            qdata = read_json_any(qfp)
        except Exception:
            continue

        # 파일 하나에 dict 1개 or list 여러개일 수 있게 처리
        qlist = qdata if isinstance(qdata, list) else [qdata]

        for qobj in qlist:
            video_id = str(qobj.get("videoID") or qobj.get("video_id") or "")
            question_id = qobj.get("question_id", os.path.basename(qfp))
            question = qobj.get("question", "")
            options = qobj.get("options", [])
            gt = qobj.get("answer", None)

            segments = mem_index.get(video_id)
            if segments:
                cleaned = [{"start": s["start"], "end": s["end"], "caption": s["caption"]} for s in segments if isinstance(s, dict)]
                memory_text = json.dumps(cleaned, ensure_ascii=False)
            else:
                memory_text = ""
            subtitles_text = get_subtitles_from_question(qobj)

            prompt = build_prompt(subtitles_text, memory_text, question, options)

            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                }
            ]

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[text],
                return_tensors="pt",
            )

            # device_map="auto"라서 model.device가 애매할 수 있음
            for k in inputs:
                if torch.is_tensor(inputs[k]):
                    inputs[k] = inputs[k].to(model.device)

            with torch.no_grad():
                gen_ids = model.generate(
                    **inputs,
                    do_sample=False,
                    top_p=None,
                    num_beams=1,
                    use_cache=True,
                    max_new_tokens=2,
                    top_k=None,
                    temperature=None,
                )
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, gen_ids)
            ]
            response = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()
            pred = extract_choice(response)

            rec = {
                "question_id": question_id,
                "videoID": video_id,
                "pred": pred,
                "raw_output": response,
                "answer": gt,
                "source_question_file": qfp,
            }

            # ✅ qid 별 파일 저장
            qid_safe = safe_filename(question_id)
            qid_file = os.path.join(by_qid_dir, f"{qid_safe}.json")
            with open(qid_file, "w", encoding="utf-8") as f:
                json.dump(rec, f, ensure_ascii=False, indent=2)

            # (옵션) jsonl 저장도 같이
            if wf_jsonl is not None:
                wf_jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if wf_jsonl is not None:
        wf_jsonl.close()

    print(f"Done. Saved per-qid jsons in: {by_qid_dir}")
    if keep_jsonl:
        print(f"Also saved jsonl: {jsonl_fp}")

if __name__ == "__main__":
    # def connect_debugpy():
    #     import debugpy
    #     if not debugpy.is_client_connected():
    #         debugpy.listen(("0.0.0.0", 1234))
    #         print("Waiting for debugger to attach...")
    #         debugpy.wait_for_client()
    #     debugpy.configure(subProcess=True)

    # connect_debugpy()
    main()