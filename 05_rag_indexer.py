"""
05_rag_indexer.py
-----------------
의료 문서를 청크로 분할하고 벡터DB(ChromaDB)에 저장하는 스크립트

사용법:
    pip install chromadb sentence-transformers langchain langchain-community
    python 05_rag_indexer.py

저장 위치: ./data/vectordb/
"""

import json
from pathlib import Path
from typing import List, Dict

# ── 설정 ──────────────────────────────────────────────────────────────────────

VECTORDB_DIR  = "./data/vectordb"
CHUNK_SIZE    = 300       # 청크 최대 글자 수
CHUNK_OVERLAP = 50        # 청크 겹치는 글자 수
EMBED_MODEL   = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"  # 한국어 임베딩 모델

# ── 의료 문서 데이터 ───────────────────────────────────────────────────────────
# 실제 서비스라면 병원 FAQ, 의료 가이드라인 등 문서를 여기에 추가
# 지금은 주요 질환별 의료 상담 가이드를 직접 작성

MEDICAL_DOCUMENTS = [
    {
        "id": "headache_001",
        "category": "신경과",
        "title": "두통 증상 가이드",
        "content": """
두통은 가장 흔한 신경과 증상 중 하나입니다.

【종류】
- 긴장성 두통: 스트레스, 피로로 인한 가장 흔한 두통. 머리 전체가 조이는 느낌.
- 편두통: 한쪽 머리가 심하게 박동성으로 아프며 구역질, 빛·소리 과민 동반.
- 군발성 두통: 한쪽 눈 주변 극심한 통증, 눈물·코막힘 동반.

【즉시 병원 방문 필요 증상 (위험 신호)】
- 갑작스럽고 극심한 두통 ("생애 최악의 두통")
- 발열, 목 경직 동반 두통 → 뇌수막염 의심
- 의식 변화, 마비, 언어장애 동반 → 뇌졸중 의심
- 3일 이상 지속되는 두통
- 구역질·구토 동반 두통

【권장 진료과】
신경과, 신경외과 (위험 신호 있을 경우 응급실)

【일반적 처치】
충분한 수면, 수분 섭취, 스트레스 관리. 시판 진통제는 주 2회 이하로 제한.
        """.strip(),
    },
    {
        "id": "diabetes_001",
        "category": "내분비내과",
        "title": "당뇨병 관리 가이드",
        "content": """
당뇨병은 인슐린 분비 또는 작용 이상으로 혈당이 높아지는 만성질환입니다.

【진단 기준】
- 공복혈당 126 mg/dL 이상
- 식후 2시간 혈당 200 mg/dL 이상
- 당화혈색소(HbA1c) 6.5% 이상

【식단 관리】
- 탄수화물: 정제 탄수화물(흰쌀, 흰빵) 줄이고 통곡물로 대체
- 채소 충분히 섭취 (식이섬유 혈당 상승 억제)
- 과일은 소량씩 (과당 주의)
- 규칙적인 식사 시간 유지 (혈당 변동 최소화)
- 음주 절제 (저혈당 위험)

【운동】
- 유산소 운동: 주 150분 이상 (빠른 걷기, 수영)
- 근력 운동: 주 2~3회
- 식후 30분 가벼운 산책 효과적

【합병증 주의】
- 당뇨병성 신경병증: 발 저림, 감각 저하
- 당뇨병성 망막병증: 시력 저하
- 당뇨병성 신증: 단백뇨, 신기능 저하
- 발 궤양: 작은 상처도 주의 필요

【권장 진료과】
내분비내과 (정기 검진 필수: 3~6개월마다 HbA1c 측정)
        """.strip(),
    },
    {
        "id": "hypertension_001",
        "category": "순환기내과",
        "title": "고혈압 관리 가이드",
        "content": """
고혈압은 수축기 혈압 140mmHg 이상 또는 이완기 혈압 90mmHg 이상인 상태입니다.

【혈압 분류】
- 정상: 120/80 mmHg 미만
- 주의혈압: 120~129 / 80 mmHg 미만
- 고혈압 1기: 130~139 / 80~89 mmHg
- 고혈압 2기: 140/90 mmHg 이상
- 고혈압 위기: 180/120 mmHg 이상 → 즉시 응급실

【생활습관 개선】
- 나트륨 섭취 하루 2,300mg(소금 6g) 이하
- 체중 감량 (5kg 감량 시 수축기 혈압 약 5mmHg 감소)
- 규칙적 유산소 운동 (주 5회 30분 이상)
- 금연, 절주
- 스트레스 관리

【약물 치료】
혈압 140/90 이상이면 생활습관 개선과 함께 약물 치료 시작 권장.
의사 처방 없이 임의로 복용 중단 금지.

【즉시 병원 방문】
- 혈압 180/120 이상
- 두통, 시야 흐림, 흉통, 호흡곤란 동반

【권장 진료과】
순환기내과, 내과
        """.strip(),
    },
    {
        "id": "chest_pain_001",
        "category": "순환기내과",
        "title": "가슴 통증 및 두근거림 가이드",
        "content": """
가슴 통증과 심계항진(두근거림)은 다양한 원인으로 발생합니다.

【심장 관련 원인 (위험)】
- 협심증: 운동 시 가슴 압박감, 쥐어짜는 듯한 통증
- 심근경색: 갑작스러운 극심한 흉통, 팔·턱으로 방사통, 식은땀
- 부정맥: 가슴 두근거림, 맥박 불규칙

【비심장성 원인】
- 역류성 식도염: 식후 가슴 쓰림, 누울 때 심해짐
- 과호흡 증후군: 불안·스트레스 시 발생
- 근골격계: 특정 자세나 압박 시 통증

【즉시 응급실 방문 증상】
- 극심한 흉통 (20분 이상 지속)
- 왼쪽 팔, 턱, 등으로 방사되는 통증
- 식은땀, 구역질, 호흡곤란 동반
- 의식 저하

【부정맥 증상】
- 가슴 두근거림 (빠르거나 불규칙한 맥박)
- 어지러움, 실신
- 호흡곤란

【권장 진료과】
순환기내과. 위험 증상 시 즉시 응급실.
        """.strip(),
    },
    {
        "id": "knee_pain_001",
        "category": "정형외과",
        "title": "무릎 통증 가이드",
        "content": """
무릎 통증은 연령, 활동 수준에 따라 다양한 원인으로 발생합니다.

【주요 원인】
- 퇴행성 관절염: 50세 이상, 계단·오래 걷기 시 통증, 아침 강직
- 반월상연골판 손상: 운동 중 무릎 비틀림 후 발생, 잠김 현상
- 십자인대 손상: 스포츠 활동 중 "뚝" 소리와 함께 발생
- 슬개골 연골연화증: 젊은 여성에 많음, 계단 내려갈 때 심해짐
- 거위발 건염: 무릎 안쪽 통증, 당뇨·비만과 연관

【즉시 병원 방문】
- 무릎이 심하게 붓고 열감이 있는 경우
- 걷기 불가능한 통증
- 관절이 잠기거나 빠지는 느낌

【생활 관리】
- 적정 체중 유지 (체중 1kg 감소 = 무릎 부담 4kg 감소)
- 대퇴사두근 강화 운동 (무릎 보호)
- 무릎에 무리가 되는 쪼그려 앉기 자제
- 냉찜질 (급성기), 온찜질 (만성기)

【권장 진료과】
정형외과 (X-ray, 필요시 MRI 검사)
        """.strip(),
    },
    {
        "id": "digestive_001",
        "category": "소화기내과",
        "title": "소화기 증상 가이드",
        "content": """
복통, 소화불량, 설사, 변비 등 소화기 증상은 매우 흔합니다.

【역류성 식도염】
- 증상: 식후 가슴 쓰림, 신물 올라옴, 목 이물감
- 관리: 식후 바로 눕지 않기, 취침 3시간 전 식사 금지, 맵고 짠 음식 자제

【과민성 장증후군】
- 증상: 복통과 함께 설사 또는 변비 반복, 스트레스 시 악화
- 관리: 규칙적 식사, 스트레스 관리, 유산균 섭취

【위염/위궤양】
- 증상: 공복 시 또는 식후 명치 통증, 구역질
- 주의: H. pylori 감염 검사 필요

【즉시 병원 방문】
- 혈변, 흑색변
- 심한 복통 (응급)
- 체중 급격히 감소
- 삼키기 어려움

【권장 진료과】
소화기내과, 내과
        """.strip(),
    },
    {
        "id": "respiratory_001",
        "category": "호흡기내과",
        "title": "호흡기 증상 가이드",
        "content": """
기침, 호흡곤란, 가래 등 호흡기 증상 가이드입니다.

【감기 vs 독감 구별】
- 감기: 서서히 시작, 콧물·기침 주증상, 발열 미미
- 독감: 갑작스러운 고열(38.5도 이상), 전신 근육통, 극심한 피로

【기침이 오래 지속될 때 (3주 이상)】
- 후비루 증후군 (코 분비물이 목으로 넘어감)
- 기관지염, 천식
- 역류성 식도염
- 결핵 (기침 + 체중감소 + 야간 발한 → 반드시 검사)

【호흡곤란】
- 갑작스러운 호흡곤란 → 즉시 응급실
- 운동 시만 호흡곤란 → 천식, 심장 문제 가능성
- 누울 때 심해짐 → 심부전 가능성

【천식 관리】
- 유발 요인 회피 (먼지, 꽃가루, 찬 공기)
- 흡입기 항상 휴대
- 야간·새벽 기침·호흡곤란 주의

【권장 진료과】
호흡기내과, 이비인후과 (코 관련), 내과
        """.strip(),
    },
]


# ── 청크 분할 함수 ─────────────────────────────────────────────────────────────

def split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """텍스트를 일정 크기로 청크 분할"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap
        if start >= len(text):
            break
    return chunks


def prepare_documents(docs: List[Dict]) -> tuple:
    """문서를 청크로 분할하고 메타데이터 준비"""
    all_texts = []
    all_metadatas = []
    all_ids = []

    for doc in docs:
        chunks = split_into_chunks(doc["content"])
        for i, chunk in enumerate(chunks):
            all_texts.append(chunk)
            all_metadatas.append({
                "doc_id": doc["id"],
                "category": doc["category"],
                "title": doc["title"],
                "chunk_index": i,
            })
            all_ids.append(f"{doc['id']}_chunk{i}")

    return all_texts, all_metadatas, all_ids


# ── 메인 실행 ─────────────────────────────────────────────────────────────────

def main():
    try:
        import chromadb
        from chromadb.utils import embedding_functions
    except ImportError:
        print("❌ chromadb가 없어요. 아래 명령어로 설치하세요:")
        print("pip install chromadb sentence-transformers")
        return

    print("[문서 준비 중...]")
    texts, metadatas, ids = prepare_documents(MEDICAL_DOCUMENTS)
    print(f"  총 청크 수: {len(texts)}개 ({len(MEDICAL_DOCUMENTS)}개 문서)")

    print(f"\n[임베딩 모델 로드] {EMBED_MODEL}")
    print("  (첫 실행 시 모델 다운로드 약 1~2분 소요)")
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )

    print(f"\n[벡터DB 구축] {VECTORDB_DIR}")
    client = chromadb.PersistentClient(path=VECTORDB_DIR)

    # 기존 컬렉션 삭제 후 재생성
    try:
        client.delete_collection("medical_docs")
        print("  기존 컬렉션 삭제")
    except:
        pass

    collection = client.create_collection(
        name="medical_docs",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    # 배치로 추가 (한 번에 너무 많으면 느림)
    batch_size = 50
    for i in range(0, len(texts), batch_size):
        batch_texts     = texts[i:i+batch_size]
        batch_metadatas = metadatas[i:i+batch_size]
        batch_ids       = ids[i:i+batch_size]
        collection.add(
            documents=batch_texts,
            metadatas=batch_metadatas,
            ids=batch_ids,
        )
        print(f"  추가 중: {min(i+batch_size, len(texts))}/{len(texts)}")

    print(f"\n✅ 벡터DB 구축 완료!")
    print(f"   저장 위치: {VECTORDB_DIR}")
    print(f"   총 청크: {collection.count()}개")

    # 검색 테스트
    print(f"\n[검색 테스트]")
    test_queries = [
        "두통이 3일째 계속되고 구역질이 나요",
        "혈압이 150/95 나왔는데 위험한가요",
        "당뇨 진단 후 식단 관리 방법",
    ]
    for query in test_queries:
        results = collection.query(query_texts=[query], n_results=1)
        top = results["metadatas"][0][0]
        print(f"\n  질문: {query}")
        print(f"  → 매칭 문서: [{top['category']}] {top['title']}")

    print("\n✅ 05_rag_indexer.py 완료! 다음: python 06_rag_chain.py")


if __name__ == "__main__":
    main()
