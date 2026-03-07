"""
03_jsonl_formatter.py
---------------------
파인튜닝용 JSONL 데이터 포맷 변환기

LLM 파인튜닝(SFT)에는 Instruction-Input-Output 구조의 JSONL이 필요합니다.
이 스크립트는 전처리된 의료 상담 데이터를 모델별 프롬프트 템플릿에 맞게 변환합니다.

지원 포맷:
1. Alpaca 포맷 (범용)
2. ChatML 포맷 (LLaMA 3 / Mistral)
3. Llama-3 한국어 포맷 (beomi/Llama-3-Open-Ko)

사용법:
    python 03_jsonl_formatter.py --input ./data/processed/consultation_clean.json
                                 --format llama3
                                 --train_ratio 0.9
"""

import json
import random
import argparse
from pathlib import Path
from datetime import datetime


# ── 설정 ──────────────────────────────────────────────────────────────────────

JSONL_DIR = Path("./data/jsonl")

# 시스템 프롬프트: 모델에게 역할을 부여
SYSTEM_PROMPT = """당신은 환자의 증상과 질문을 듣고 의학적 정보를 제공하는 의료 상담 AI입니다.
정확하고 신뢰할 수 있는 의료 정보를 바탕으로 답변하되, 반드시 전문 의료진 상담을 권장하세요.
진단이나 처방을 직접 내리지 말고, 가능한 원인과 권장 진료과를 안내하는 방식으로 답변하세요."""


# ── 프롬프트 템플릿 ───────────────────────────────────────────────────────────

def format_alpaca(record: dict) -> dict:
    """
    Alpaca 포맷 (가장 범용적, 많은 오픈소스 파인튜닝 코드에서 지원)

    {
        "instruction": "시스템 역할 + 태스크 설명",
        "input": "환자 질문",
        "output": "의사 답변"
    }
    """
    return {
        "instruction": SYSTEM_PROMPT,
        "input": record["input"],
        "output": record["output"],
    }


def format_chatml(record: dict) -> dict:
    """
    ChatML 포맷 (LLaMA 3, Mistral에서 권장)

    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]
    }
    """
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": record["input"]},
            {"role": "assistant", "content": record["output"]},
        ]
    }


def format_llama3(record: dict) -> dict:
    """
    LLaMA 3 공식 포맷 (beomi/Llama-3-Open-Ko-8B 파인튜닝 권장)

    특수 토큰:
    <|begin_of_text|>
    <|start_header_id|>system<|end_header_id|>
    ...
    <|eot_id|>
    """
    text = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{record['input']}"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{record['output']}"
        "<|eot_id|>"
        "<|end_of_text|>"
    )
    return {"text": text}


# 포맷 함수 매핑
FORMAT_FUNCTIONS = {
    "alpaca": format_alpaca,
    "chatml": format_chatml,
    "llama3": format_llama3,
}


# ── 메인 변환 클래스 ──────────────────────────────────────────────────────────

class JSONLFormatter:

    def __init__(self, format_type: str = "llama3"):
        if format_type not in FORMAT_FUNCTIONS:
            raise ValueError(f"지원 포맷: {list(FORMAT_FUNCTIONS.keys())}")
        self.format_type = format_type
        self.format_fn = FORMAT_FUNCTIONS[format_type]

    def convert(self, records: list[dict]) -> list[dict]:
        """전처리된 records를 파인튜닝 포맷으로 변환"""
        formatted = []
        for record in records:
            try:
                formatted.append(self.format_fn(record))
            except KeyError as e:
                print(f"[경고] 필드 누락으로 건너뜀: {e}")
        return formatted

    def split_train_val(
        self,
        records: list[dict],
        train_ratio: float = 0.9,
        seed: int = 42,
    ) -> tuple[list, list]:
        """학습/검증 데이터 분리"""
        random.seed(seed)
        shuffled = records.copy()
        random.shuffle(shuffled)

        split_idx = int(len(shuffled) * train_ratio)
        train = shuffled[:split_idx]
        val = shuffled[split_idx:]

        print(f"\n데이터 분리: 학습 {len(train):,}개 / 검증 {len(val):,}개")
        return train, val

    def save_jsonl(self, records: list[dict], output_path: str):
        """JSONL 형태로 저장 (한 줄 = 하나의 샘플)"""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"[저장] {path} ({len(records):,}개)")

    def verify_output(self, jsonl_path: str, n_samples: int = 2):
        """저장된 JSONL 검증 및 미리보기"""
        print(f"\n{'='*60}")
        print(f"JSONL 검증: {jsonl_path}")
        print(f"{'='*60}")

        path = Path(jsonl_path)
        if not path.exists():
            print("[오류] 파일 없음")
            return

        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        print(f"총 줄 수: {len(lines):,}")

        for i, line in enumerate(lines[:n_samples]):
            record = json.loads(line)
            print(f"\n--- 샘플 {i+1} ---")

            if self.format_type == "llama3":
                # LLaMA3 포맷은 text 필드만 있음
                preview = record["text"][:300].replace("\n", "↵")
                print(f"text (앞 300자): {preview}...")

            elif self.format_type == "chatml":
                for msg in record["messages"]:
                    role = msg["role"]
                    content = msg["content"][:100].replace("\n", " ")
                    print(f"[{role}]: {content}...")

            elif self.format_type == "alpaca":
                print(f"instruction: {record['instruction'][:80]}...")
                print(f"input: {record['input'][:100]}...")
                print(f"output: {record['output'][:100]}...")


# ── 데이터셋 카드 생성 (HuggingFace Hub 업로드용) ─────────────────────────────

def create_dataset_card(
    output_dir: Path,
    train_count: int,
    val_count: int,
    format_type: str,
):
    """HuggingFace Hub에 업로드할 때 함께 올리는 README"""
    card = f"""---
language:
- ko
task_categories:
- text-generation
- question-answering
domain:
- medical
---

# 한국어 의료 상담 파인튜닝 데이터셋

## 개요
AI Hub 한국어 의료 상담 데이터를 LLM 파인튜닝(SFT)용으로 변환한 데이터셋입니다.

## 데이터 통계
| 분할 | 샘플 수 |
|------|--------|
| train | {train_count:,} |
| validation | {val_count:,} |
| **합계** | **{train_count + val_count:,}** |

## 포맷
`{format_type}` 포맷 사용

## 원본 데이터 출처
- AI Hub 한국어 의료 상담 데이터 (https://aihub.or.kr)
- 라이선스: AI Hub 데이터 활용 정책 준수

## 생성일
{datetime.now().strftime("%Y-%m-%d")}
"""
    card_path = output_dir / "README.md"
    with open(card_path, "w", encoding="utf-8") as f:
        f.write(card)
    print(f"[데이터셋 카드 생성] {card_path}")


# ── 실행 ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="파인튜닝용 JSONL 변환")
    parser.add_argument("--input", type=str, default="./data/processed/consultation_clean.json")
    parser.add_argument("--output_dir", type=str, default="./data/jsonl")
    parser.add_argument(
        "--format",
        choices=["alpaca", "chatml", "llama3"],
        default="llama3",
        help="파인튜닝 모델에 맞는 포맷 선택",
    )
    parser.add_argument("--train_ratio", type=float, default=0.9)
    args = parser.parse_args()

    # 데이터 로드
    with open(args.input, "r", encoding="utf-8") as f:
        records = json.load(f)
    if isinstance(records, dict):
        records = records.get("data", [])

    print(f"[입력 데이터] {len(records):,}개")

    # 변환 실행
    formatter = JSONLFormatter(format_type=args.format)
    formatted = formatter.convert(records)

    # 학습/검증 분리
    train_data, val_data = formatter.split_train_val(formatted, args.train_ratio)

    # 저장
    output_dir = Path(args.output_dir)
    formatter.save_jsonl(train_data, output_dir / "train.jsonl")
    formatter.save_jsonl(val_data, output_dir / "val.jsonl")

    # 검증
    formatter.verify_output(output_dir / "train.jsonl")

    # 데이터셋 카드 생성
    create_dataset_card(output_dir, len(train_data), len(val_data), args.format)

    print(f"\n✅ JSONL 변환 완료! 다음 단계: Colab에서 04_finetune.ipynb 실행")
