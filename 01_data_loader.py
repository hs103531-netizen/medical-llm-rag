"""
01_data_loader.py
-----------------
AI Hub 한국어 의료 데이터셋 로더 및 초기 탐색 스크립트

지원 데이터셋:
- 한국어 의료 상담 데이터 (AI Hub #71)
- 진료기록 요약 데이터 (AI Hub #464)
- 의료 특화 언어모델 학습 데이터 (AI Hub #582)

사용법:
    python 01_data_loader.py --data_path ./data/raw --dataset_type consultation
"""

import os
import json
import argparse
import pandas as pd
from pathlib import Path
from typing import Union


# ── 설정 ──────────────────────────────────────────────────────────────────────

RAW_DIR = Path("./data/raw")
PROCESSED_DIR = Path("./data/processed")

# AI Hub 데이터셋 타입별 파일 구조 정의
DATASET_CONFIGS = {
    "consultation": {
        # AI Hub '한국어 의료 상담 데이터' (#71)
        # 다운로드 경로: https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=71
        "description": "한국어 의료 상담 QA 데이터 (환자 질문 + 의사 답변)",
        "expected_format": "json",
        "key_fields": {
            "input": "질문",       # 환자 질문 필드명 (실제 다운로드 후 확인 필요)
            "output": "답변",      # 의사 답변 필드명
            "category": "진료과",  # 진료과 카테고리
        },
        "sample_count": 180000,
    },
    "summary": {
        # AI Hub '진료기록 요약 데이터' (#464)
        # 다운로드 경로: https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=464
        "description": "진료기록 원문 → 요약 쌍 데이터",
        "expected_format": "json",
        "key_fields": {
            "input": "원문",
            "output": "요약",
        },
        "sample_count": 50000,
    },
}


# ── 유틸리티 함수 ─────────────────────────────────────────────────────────────

def load_json_file(file_path: Union[str, Path]) -> dict | list:
    """JSON 파일 로드 (UTF-8 인코딩 명시)"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl_file(file_path: Union[str, Path]) -> list[dict]:
    """JSONL 파일 로드 (줄 단위 JSON)"""
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [경고] {line_num}번째 줄 파싱 실패: {e}")
    return records


def explore_json_structure(data: dict | list, depth: int = 0, max_depth: int = 3):
    """
    JSON 구조 탐색 출력 (AI Hub 데이터는 다운로드 후 구조 확인 필수)
    
    AI Hub 데이터는 버전/라이선스에 따라 필드명이 다를 수 있어서
    실제 데이터를 받은 뒤 이 함수로 구조를 먼저 파악하는 것이 중요합니다.
    """
    indent = "  " * depth
    if depth > max_depth:
        print(f"{indent}...")
        return

    if isinstance(data, dict):
        for key, value in list(data.items())[:5]:  # 상위 5개 키만 출력
            print(f"{indent}[key] '{key}': {type(value).__name__}", end="")
            if isinstance(value, str):
                preview = value[:50].replace("\n", " ")
                print(f" → '{preview}...'")
            elif isinstance(value, (int, float, bool)):
                print(f" → {value}")
            else:
                print()
                explore_json_structure(value, depth + 1, max_depth)

    elif isinstance(data, list):
        print(f"{indent}[list] 길이: {len(data)}개")
        if len(data) > 0:
            print(f"{indent}[첫 번째 항목]")
            explore_json_structure(data[0], depth + 1, max_depth)


# ── 메인 로더 클래스 ──────────────────────────────────────────────────────────

class AIHubMedicalLoader:
    """
    AI Hub 한국어 의료 데이터셋 로더
    
    AI Hub 데이터 다운로드 후 아래 단계로 진행:
    1. AI Hub 회원가입 및 데이터 신청 (보통 1~3일 승인)
    2. 데이터 압축 해제 → ./data/raw/ 폴더에 저장
    3. explore_structure() 로 필드명 확인
    4. load() 로 데이터 로드
    """

    def __init__(self, raw_dir: Path = RAW_DIR):
        self.raw_dir = raw_dir
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.data = []

    def explore_structure(self, file_path: str):
        """
        데이터 구조 탐색 (다운로드 후 가장 먼저 실행)
        
        예시:
            loader = AIHubMedicalLoader()
            loader.explore_structure("./data/raw/medical_consultation.json")
        """
        print(f"\n{'='*50}")
        print(f"파일 구조 탐색: {file_path}")
        print(f"{'='*50}")

        path = Path(file_path)
        if not path.exists():
            print(f"[오류] 파일이 없습니다: {file_path}")
            print("AI Hub에서 데이터를 먼저 다운로드하세요.")
            print("→ https://aihub.or.kr")
            return

        # 확장자에 따라 로드 방식 분기
        if path.suffix == ".json":
            data = load_json_file(path)
        elif path.suffix == ".jsonl":
            data = load_jsonl_file(path)
        else:
            print(f"[미지원 형식] {path.suffix}")
            return

        explore_json_structure(data)

    def load_consultation_data(
        self,
        file_path: str,
        input_field: str = "질문",
        output_field: str = "답변",
        category_field: str = None,
    ) -> list[dict]:
        """
        의료 상담 데이터 로드

        Args:
            file_path: 데이터 파일 경로
            input_field: 입력(질문) 필드명 → explore_structure()로 확인
            output_field: 출력(답변) 필드명
            category_field: 카테고리 필드명 (없으면 None)

        Returns:
            [{"input": "...", "output": "...", "category": "..."}, ...]
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(
                f"파일 없음: {file_path}\n"
                "AI Hub에서 데이터를 다운로드하세요: https://aihub.or.kr"
            )

        print(f"\n[로드 중] {path.name}")
        raw_data = load_json_file(path)

        # AI Hub JSON은 보통 {"data": [...]} 또는 직접 리스트 형태
        if isinstance(raw_data, dict):
            # 가장 큰 리스트를 가진 키를 자동 탐색
            list_keys = [k for k, v in raw_data.items() if isinstance(v, list)]
            if list_keys:
                raw_data = raw_data[list_keys[0]]
                print(f"  → 데이터 키: '{list_keys[0]}'")

        records = []
        skipped = 0

        for item in raw_data:
            try:
                record = {
                    "input": item[input_field].strip(),
                    "output": item[output_field].strip(),
                }
                if category_field and category_field in item:
                    record["category"] = item[category_field]

                # 빈 텍스트 필터링
                if not record["input"] or not record["output"]:
                    skipped += 1
                    continue

                records.append(record)

            except KeyError as e:
                skipped += 1
                if skipped <= 3:  # 처음 3개만 경고 출력
                    print(f"  [경고] 필드 없음: {e} → explore_structure()로 필드명 확인 필요")

        print(f"  → 로드 완료: {len(records):,}개 (건너뜀: {skipped:,}개)")
        self.data = records
        return records

    def get_statistics(self) -> pd.DataFrame:
        """로드된 데이터 기본 통계"""
        if not self.data:
            print("[오류] 데이터를 먼저 로드하세요.")
            return

        df = pd.DataFrame(self.data)
        df["input_len"] = df["input"].str.len()
        df["output_len"] = df["output"].str.len()

        print("\n" + "="*50)
        print("데이터 기본 통계")
        print("="*50)
        print(f"총 샘플 수   : {len(df):,}개")
        print(f"입력 평균 길이: {df['input_len'].mean():.0f}자")
        print(f"출력 평균 길이: {df['output_len'].mean():.0f}자")
        print(f"입력 최대 길이: {df['input_len'].max():,}자")
        print(f"출력 최대 길이: {df['output_len'].max():,}자")

        if "category" in df.columns:
            print(f"\n진료과별 분포 (상위 10개):")
            print(df["category"].value_counts().head(10).to_string())

        return df


# ── 데이터 미리보기 (AI Hub 없이 테스트용 샘플 데이터) ─────────────────────────

def create_sample_data():
    """
    AI Hub 승인 대기 중일 때 파이프라인 테스트용 샘플 데이터 생성
    실제 데이터와 동일한 구조로 만들어서 파이프라인 미리 테스트 가능
    """
    sample_records = [
        {
            "질문": "요즘 두통이 자주 생기고 눈이 침침한데 어떤 과를 가야 하나요?",
            "답변": "두통과 시력 저하가 동반될 경우 신경과 또는 안과 진료를 권장합니다. "
                   "특히 두통이 갑자기 심해지거나 구역질을 동반하면 신경과를 먼저 방문하시는 것이 좋습니다.",
            "진료과": "신경과",
        },
        {
            "질문": "당뇨 환자인데 발이 저리고 감각이 없어요. 위험한 건가요?",
            "답변": "당뇨병성 말초신경병증 증상일 수 있습니다. 방치하면 당뇨병성 족부 궤양으로 진행될 수 있어 "
                   "내분비내과 또는 신경과에서 정밀 검사를 받으시길 강력히 권장합니다.",
            "진료과": "내분비내과",
        },
        {
            "질문": "혈압이 160/100 정도 나오는데 약을 먹어야 하나요?",
            "답변": "수축기 160mmHg는 2기 고혈압에 해당합니다. 생활습관 개선만으로는 조절이 어려운 수준이므로 "
                   "순환기내과 전문의와 상담 후 적절한 약물 치료를 시작하는 것이 권장됩니다.",
            "진료과": "순환기내과",
        },
        {
            "질문": "갑자기 가슴이 두근거리고 숨이 차요. 심장 문제일까요?",
            "답변": "심계항진과 호흡곤란이 동반될 경우 부정맥, 심부전 등 심장 문제일 수 있습니다. "
                   "증상이 지속되면 즉시 순환기내과 진료를 받으시고, 갑작스러운 흉통이 동반되면 응급실을 방문하세요.",
            "진료과": "순환기내과",
        },
        {
            "질문": "무릎이 계속 아프고 계단을 오르내릴 때 특히 심해요.",
            "답변": "슬관절 통증의 경우 퇴행성 관절염, 반월상연골판 손상 등이 원인일 수 있습니다. "
                   "정형외과에서 X-ray 및 MRI 검사를 통해 정확한 원인을 파악하는 것이 중요합니다.",
            "진료과": "정형외과",
        },
    ]

    # 샘플 데이터를 실제 AI Hub 포맷으로 저장
    sample_path = RAW_DIR / "sample_consultation.json"
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    with open(sample_path, "w", encoding="utf-8") as f:
        json.dump({"data": sample_records}, f, ensure_ascii=False, indent=2)

    print(f"[샘플 데이터 생성 완료] {sample_path}")
    print("→ 실제 AI Hub 데이터 수신 전까지 이 파일로 파이프라인을 테스트하세요.")
    return str(sample_path)


# ── 실행 ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Hub 의료 데이터 로더")
    parser.add_argument("--mode", choices=["sample", "load", "explore"], default="sample",
                        help="sample: 테스트 데이터 생성 / load: 실제 데이터 로드 / explore: 구조 탐색")
    parser.add_argument("--file", type=str, default=None, help="로드할 파일 경로")
    parser.add_argument("--input_field", type=str, default="질문", help="입력 필드명")
    parser.add_argument("--output_field", type=str, default="답변", help="출력 필드명")
    args = parser.parse_args()

    loader = AIHubMedicalLoader()

    if args.mode == "sample":
        # AI Hub 승인 전 테스트
        sample_path = create_sample_data()
        records = loader.load_consultation_data(
            sample_path, input_field="질문", output_field="답변", category_field="진료과"
        )
        loader.get_statistics()
        print(f"\n[샘플 미리보기]")
        print(f"Q: {records[0]['input']}")
        print(f"A: {records[0]['output']}")

    elif args.mode == "explore":
        # 실제 데이터 구조 탐색
        if not args.file:
            print("[오류] --file 옵션으로 파일 경로를 지정하세요.")
        else:
            loader.explore_structure(args.file)

    elif args.mode == "load":
        # 실제 데이터 로드
        if not args.file:
            print("[오류] --file 옵션으로 파일 경로를 지정하세요.")
        else:
            records = loader.load_consultation_data(
                args.file,
                input_field=args.input_field,
                output_field=args.output_field,
            )
            loader.get_statistics()
