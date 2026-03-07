"""
02_preprocessor.py
------------------
한국어 의료 텍스트 전처리 파이프라인

주요 처리 단계:
1. 텍스트 정제 (노이즈 제거, 인코딩 정규화)
2. 한국어 의료 특화 정규화
3. 토큰 길이 필터링 (Colab VRAM 최적화)
4. 품질 필터링 (너무 짧거나 의미없는 샘플 제거)

사용법:
    python 02_preprocessor.py --input ./data/raw/sample_consultation.json
"""

import re
import json
import unicodedata
import argparse
from pathlib import Path
from typing import Optional
from transformers import AutoTokenizer


# ── 설정 ──────────────────────────────────────────────────────────────────────

PROCESSED_DIR = Path("./data/processed")

# 토큰 길이 제한 (Colab T4 16GB VRAM 기준 최적값)
MAX_INPUT_TOKENS = 256
MAX_OUTPUT_TOKENS = 512
MIN_INPUT_CHARS = 10   # 너무 짧은 질문 제거
MIN_OUTPUT_CHARS = 20  # 너무 짧은 답변 제거


# ── 텍스트 정제 함수 ──────────────────────────────────────────────────────────

def normalize_unicode(text: str) -> str:
    """
    유니코드 정규화 (한국어 텍스트 필수 처리)
    NFC: 완성형 한글로 통일 (NFD 분해형 → NFC 합성형)
    예: 한글 → 한글
    """
    return unicodedata.normalize("NFC", text)


def remove_html_tags(text: str) -> str:
    """HTML 태그 제거 (웹 크롤링 잔재)"""
    return re.sub(r"<[^>]+>", "", text)


def normalize_whitespace(text: str) -> str:
    """
    공백 정규화
    - 연속 공백 → 단일 공백
    - 연속 줄바꿈 → 최대 2개로 제한
    - 앞뒤 공백 제거
    """
    text = re.sub(r"[ \t]+", " ", text)        # 연속 공백
    text = re.sub(r"\n{3,}", "\n\n", text)      # 연속 줄바꿈
    return text.strip()


def remove_special_noise(text: str) -> str:
    """
    의료 텍스트 노이즈 제거
    - 이메일, URL 제거 (개인정보)
    - 전화번호 마스킹
    - 특수문자 과다 반복 제거
    """
    text = re.sub(r"https?://\S+", "", text)                     # URL
    text = re.sub(r"\S+@\S+\.\S+", "", text)                     # 이메일
    text = re.sub(r"\d{2,3}-\d{3,4}-\d{4}", "[전화번호]", text)  # 전화번호
    text = re.sub(r"([!?.])\1{2,}", r"\1", text)                 # 반복 문장부호
    text = re.sub(r"[ㅋㅎㅠㅜ]{3,}", "", text)                   # 자음/모음 반복 (구어체)
    return text


def normalize_medical_terms(text: str) -> str:
    """
    한국어 의료 텍스트 표준화
    동일 의미의 다양한 표현을 통일하여 모델 학습 효율 향상
    """
    # 단위 표기 통일
    text = re.sub(r"㎎|밀리그램", "mg", text)
    text = re.sub(r"㎖|밀리리터", "ml", text)
    text = re.sub(r"㎝|센티미터", "cm", text)

    # 자주 쓰이는 의료 약어 풀어쓰기 (선택적)
    medical_abbr = {
        "고혈압": "고혈압(혈압이 정상 범위보다 높은 상태)",
        # 필요시 추가
    }
    # 주의: 파인튜닝용 데이터는 풀어쓰기 하지 않는 것이 일반적
    # RAG 청크 데이터에만 적용 권장

    return text


def clean_text(text: str) -> str:
    """전체 정제 파이프라인 순서대로 적용"""
    text = normalize_unicode(text)
    text = remove_html_tags(text)
    text = remove_special_noise(text)
    text = normalize_whitespace(text)
    return text


# ── 품질 필터링 ───────────────────────────────────────────────────────────────

def is_valid_sample(
    input_text: str,
    output_text: str,
    min_input: int = MIN_INPUT_CHARS,
    min_output: int = MIN_OUTPUT_CHARS,
) -> tuple[bool, str]:
    """
    샘플 유효성 검사

    Returns:
        (통과 여부, 실패 이유)
    """
    if len(input_text) < min_input:
        return False, f"입력 너무 짧음 ({len(input_text)}자)"

    if len(output_text) < min_output:
        return False, f"출력 너무 짧음 ({len(output_text)}자)"

    # 한글 포함 비율 체크 (의료 데이터는 최소 30% 이상 한글)
    korean_chars = len(re.findall(r"[가-힣]", input_text))
    korean_ratio = korean_chars / len(input_text) if input_text else 0
    if korean_ratio < 0.1:
        return False, f"한글 비율 낮음 ({korean_ratio:.1%})"

    # 중복 문장 비율 체크
    sentences = re.split(r"[.!?]\s", output_text)
    if len(sentences) > 3 and len(set(sentences)) / len(sentences) < 0.5:
        return False, "중복 문장 과다"

    return True, "통과"


# ── 토큰 길이 필터링 ──────────────────────────────────────────────────────────

class TokenLengthFilter:
    """
    토크나이저 기반 길이 필터
    Colab T4(16GB) 환경에서 OOM 방지를 위해 필수
    """

    def __init__(self, model_name: str = "beomi/Llama-3-Open-Ko-8B"):
        """
        model_name: 파인튜닝할 모델의 토크나이저
        한국어 모델 추천:
        - beomi/Llama-3-Open-Ko-8B (LLaMA 3 한국어)
        - yanolja/EEVE-Korean-Instruct-10.8B-v1.0
        - MLP-KTLim/llama-3-Korean-Bllossom-8B
        """
        print(f"[토크나이저 로드] {model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.available = True
        except Exception as e:
            print(f"  [경고] 토크나이저 로드 실패: {e}")
            print("  → 문자 수 기반 근사 필터로 대체합니다.")
            self.available = False

    def get_token_count(self, text: str) -> int:
        if self.available:
            return len(self.tokenizer.encode(text))
        else:
            # 한국어 평균 토큰 길이 약 1.5자/토큰으로 근사
            return len(text) // 2

    def is_within_limit(
        self,
        input_text: str,
        output_text: str,
        max_input: int = MAX_INPUT_TOKENS,
        max_output: int = MAX_OUTPUT_TOKENS,
    ) -> tuple[bool, dict]:
        input_tokens = self.get_token_count(input_text)
        output_tokens = self.get_token_count(output_text)

        info = {"input_tokens": input_tokens, "output_tokens": output_tokens}
        if input_tokens > max_input:
            return False, {**info, "reason": f"입력 토큰 초과 ({input_tokens} > {max_input})"}
        if output_tokens > max_output:
            return False, {**info, "reason": f"출력 토큰 초과 ({output_tokens} > {max_output})"}

        return True, info


# ── 메인 전처리 클래스 ────────────────────────────────────────────────────────

class MedicalPreprocessor:

    def __init__(
        self,
        model_name: str = "beomi/Llama-3-Open-Ko-8B",
        use_token_filter: bool = True,
    ):
        self.use_token_filter = use_token_filter
        if use_token_filter:
            self.token_filter = TokenLengthFilter(model_name)

        self.stats = {
            "total": 0,
            "passed": 0,
            "filtered_quality": 0,
            "filtered_token": 0,
        }

    def detect_fields(self, records: list[dict]) -> tuple[str, str]:
        """
        데이터의 필드명 자동 감지
        input/output 또는 질문/답변 등 다양한 필드명 지원
        """
        if not records:
            return "input", "output"

        sample = records[0]
        keys = list(sample.keys())

        # 우선순위 순서로 탐색
        input_candidates  = ["input", "질문", "question", "instruction", "text"]
        output_candidates = ["output", "답변", "answer", "response", "label"]

        input_field  = next((k for k in input_candidates  if k in keys), keys[0])
        output_field = next((k for k in output_candidates if k in keys), keys[1] if len(keys) > 1 else keys[0])

        print(f"  → 필드 자동 감지: 입력='{input_field}', 출력='{output_field}'")
        return input_field, output_field

    def process(self, records: list[dict], input_field: str = None, output_field: str = None) -> list[dict]:
        """
        전체 전처리 파이프라인 실행

        Args:
            records: 원본 데이터 리스트
            input_field: 입력 필드명 (None이면 자동 감지)
            output_field: 출력 필드명 (None이면 자동 감지)

        Returns:
            전처리 완료된 records (필드명은 input/output으로 통일)
        """
        # 필드명 자동 감지
        if input_field is None or output_field is None:
            input_field, output_field = self.detect_fields(records)

        self.stats["total"] = len(records)
        processed = []

        for record in records:
            input_text  = clean_text(record[input_field])
            output_text = clean_text(record[output_field])

            # 1. 품질 필터링
            valid, reason = is_valid_sample(input_text, output_text)
            if not valid:
                self.stats["filtered_quality"] += 1
                continue

            # 2. 토큰 길이 필터링
            if self.use_token_filter:
                within_limit, token_info = self.token_filter.is_within_limit(
                    input_text, output_text
                )
                if not within_limit:
                    self.stats["filtered_token"] += 1
                    continue

            # 필드명을 input/output으로 통일 (다음 단계 호환성)
            extra = {k: v for k, v in record.items() if k not in [input_field, output_field]}
            new_record = {**extra, "input": input_text, "output": output_text}
            processed.append(new_record)

        self.stats["passed"] = len(processed)
        self._print_stats()
        return processed

    def _print_stats(self):
        s = self.stats
        print("\n" + "="*50)
        print("전처리 결과")
        print("="*50)
        print(f"전체 샘플     : {s['total']:,}개")
        print(f"통과          : {s['passed']:,}개 ({s['passed']/s['total']*100:.1f}%)")
        print(f"품질 필터 제거 : {s['filtered_quality']:,}개")
        print(f"토큰 초과 제거 : {s['filtered_token']:,}개")

    def save(self, records: list[dict], output_path: str):
        """전처리 완료 데이터 저장"""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

        print(f"\n[저장 완료] {path} ({len(records):,}개)")


# ── 실행 ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="의료 텍스트 전처리")
    parser.add_argument("--input",        type=str, default="./data/raw/sample_consultation.json")
    parser.add_argument("--output",       type=str, default="./data/processed/consultation_clean.json")
    parser.add_argument("--input_field",  type=str, default=None, help="입력 필드명 (기본: 자동 감지)")
    parser.add_argument("--output_field", type=str, default=None, help="출력 필드명 (기본: 자동 감지)")
    parser.add_argument("--no_token_filter", action="store_true", help="토큰 필터 비활성화")
    args = parser.parse_args()

    # 데이터 로드
    with open(args.input, "r", encoding="utf-8") as f:
        raw = json.load(f)
    records = raw.get("data", raw) if isinstance(raw, dict) else raw

    # 전처리 실행
    preprocessor = MedicalPreprocessor(use_token_filter=not args.no_token_filter)
    cleaned = preprocessor.process(records, input_field=args.input_field, output_field=args.output_field)

    # 저장
    preprocessor.save(cleaned, args.output)

    # 미리보기
    print(f"\n[전처리 후 샘플]")
    print(f"Q: {cleaned[0]['input']}")
    print(f"A: {cleaned[0]['output']}")