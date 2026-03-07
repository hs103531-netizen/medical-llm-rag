"""
01_kormed_loader.py
-------------------
KorMedMCQA 데이터셋을 로드하고 파인튜닝용 JSONL로 변환하는 스크립트

데이터셋: sean0042/KorMedMCQA
- 한국 의사/간호사/약사 국가고시 문제 (2012~2024)
- train: 1,890개 / dev: 164개 / test: 435개

변환 방식:
  question + 보기(A~E) → input
  answer + cot(풀이)   → output

사용법:
    python 01_kormed_loader.py
"""

import json
from pathlib import Path
from datasets import load_dataset

# ── 설정 ──────────────────────────────────────────────────────────────────────

OUTPUT_DIR = Path("./data/jsonl")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = """당신은 한국 의료 국가고시 전문가 AI입니다.
주어진 의료 문제를 분석하고 정확한 답변과 근거를 제시하세요.
반드시 전문 의료진 상담을 권장하는 내용을 포함하세요."""


# ── 데이터 변환 함수 ──────────────────────────────────────────────────────────

def format_question(row: dict) -> str:
    """문제 + 보기를 input 텍스트로 변환"""
    question = row["question"]
    choices = f"A. {row['A']}\nB. {row['B']}\nC. {row['C']}\nD. {row['D']}\nE. {row['E']}"
    return f"{question}\n\n{choices}"


def format_answer(row: dict) -> str:
    """정답 + 풀이를 output 텍스트로 변환"""
    num_to_letter = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E"}
    raw_answer = row["answer"]

    if isinstance(raw_answer, int):
        answer_key = num_to_letter.get(raw_answer, str(raw_answer))
    else:
        answer_key = str(raw_answer).upper()

    answer_text = row.get(answer_key, "")
    cot = row.get("cot", "") or ""

    output = f"정답: {answer_key}. {answer_text}"
    if cot:
        output += f"\n\n풀이: {cot}"
    return output


def to_llama3_format(input_text: str, output_text: str) -> dict:
    """LLaMA3 파인튜닝 포맷으로 변환"""
    text = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{input_text}"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{output_text}"
        "<|eot_id|>"
        "<|end_of_text|>"
    )
    return {"text": text}


# ── 메인 실행 ─────────────────────────────────────────────────────────────────

def main():
    print("[데이터 로드] sean0042/KorMedMCQA (doctor)")
    ds = load_dataset("sean0042/KorMedMCQA", name="doctor")

    print(f"  train: {len(ds['train']):,}개")
    print(f"  dev  : {len(ds['dev']):,}개")

    # 샘플 미리보기
    sample = ds["train"][0]
    print(f"\n--- 원본 샘플 ---")
    print(f"문제: {sample['question'][:80]}...")
    print(f"정답: {sample['answer']}")
    print(f"풀이: {sample['cot'][:80]}..." if sample.get('cot') else "풀이: 없음")

    # 변환
    print("\n[JSONL 변환 중...]")
    train_data = []
    val_data = []

    for row in ds["train"]:
        input_text  = format_question(row)
        output_text = format_answer(row)
        train_data.append(to_llama3_format(input_text, output_text))

    for row in ds["dev"]:
        input_text  = format_question(row)
        output_text = format_answer(row)
        val_data.append(to_llama3_format(input_text, output_text))

    # 저장
    train_path = OUTPUT_DIR / "train.jsonl"
    val_path   = OUTPUT_DIR / "val.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for record in train_data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for record in val_data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"  train.jsonl 저장: {len(train_data):,}개 → {train_path}")
    print(f"  val.jsonl   저장: {len(val_data):,}개 → {val_path}")

    # 변환 결과 미리보기
    print(f"\n--- 변환된 샘플 (앞 400자) ---")
    print(train_data[0]["text"][:400])
    print("\n✅ 완료! 이제 train.jsonl, val.jsonl을 Google Drive에 업로드하세요.")


if __name__ == "__main__":
    main()