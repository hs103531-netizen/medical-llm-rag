"""
app.py
------
한국어 의료 상담 RAG 시스템 - Gradio UI (HuggingFace Spaces 배포용)

실행:
    pip install gradio
    python app.py
"""

import os
import threading

import gradio as gr
import torch
import chromadb
from chromadb.utils import embedding_functions
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── 설정 (06_rag_chain.py와 동일) ────────────────────────────────────────────

VECTORDB_DIR    = "./data/vectordb"
EMBED_MODEL     = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
FINETUNED_MODEL = "jdi1009/checkpoints"
TOP_K           = 2
MAX_NEW_TOKENS  = 300

# ── 의료 문서 (05_rag_indexer.py와 동일 — 벡터DB가 없을 때 자동 구축용) ─────

MEDICAL_DOCUMENTS = [
    {
        "id": "headache_001",
        "category": "신경과",
        "title": "두통 증상 가이드",
        "content": """두통은 가장 흔한 신경과 증상 중 하나입니다.

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
충분한 수면, 수분 섭취, 스트레스 관리. 시판 진통제는 주 2회 이하로 제한.""",
    },
    {
        "id": "diabetes_001",
        "category": "내분비내과",
        "title": "당뇨병 관리 가이드",
        "content": """당뇨병은 인슐린 분비 또는 작용 이상으로 혈당이 높아지는 만성질환입니다.

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
내분비내과 (정기 검진 필수: 3~6개월마다 HbA1c 측정)""",
    },
    {
        "id": "hypertension_001",
        "category": "순환기내과",
        "title": "고혈압 관리 가이드",
        "content": """고혈압은 수축기 혈압 140mmHg 이상 또는 이완기 혈압 90mmHg 이상인 상태입니다.

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
순환기내과, 내과""",
    },
    {
        "id": "chest_pain_001",
        "category": "순환기내과",
        "title": "가슴 통증 및 두근거림 가이드",
        "content": """가슴 통증과 심계항진(두근거림)은 다양한 원인으로 발생합니다.

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
순환기내과. 위험 증상 시 즉시 응급실.""",
    },
    {
        "id": "knee_pain_001",
        "category": "정형외과",
        "title": "무릎 통증 가이드",
        "content": """무릎 통증은 연령, 활동 수준에 따라 다양한 원인으로 발생합니다.

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
정형외과 (X-ray, 필요시 MRI 검사)""",
    },
    {
        "id": "digestive_001",
        "category": "소화기내과",
        "title": "소화기 증상 가이드",
        "content": """복통, 소화불량, 설사, 변비 등 소화기 증상은 매우 흔합니다.

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
소화기내과, 내과""",
    },
    {
        "id": "respiratory_001",
        "category": "호흡기내과",
        "title": "호흡기 증상 가이드",
        "content": """기침, 호흡곤란, 가래 등 호흡기 증상 가이드입니다.

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
호흡기내과, 이비인후과 (코 관련), 내과""",
    },
]


# ── 유틸: 청크 분할 (05_rag_indexer.py와 동일) ───────────────────────────────

def _split_chunks(text: str, size: int = 300, overlap: int = 50):
    chunks, start = [], 0
    while start < len(text):
        chunk = text[start : start + size].strip()
        if chunk:
            chunks.append(chunk)
        start += size - overlap
        if start >= len(text):
            break
    return chunks


def _build_vectordb():
    """벡터DB가 없을 때 MEDICAL_DOCUMENTS로 자동 구축"""
    print("[벡터DB 자동 구축 중...]")
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    os.makedirs(VECTORDB_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=VECTORDB_DIR)

    try:
        client.delete_collection("medical_docs")
    except Exception:
        pass

    collection = client.create_collection(
        name="medical_docs",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    texts, metadatas, ids = [], [], []
    for doc in MEDICAL_DOCUMENTS:
        for i, chunk in enumerate(_split_chunks(doc["content"])):
            texts.append(chunk)
            metadatas.append(
                {"doc_id": doc["id"], "category": doc["category"],
                 "title": doc["title"], "chunk_index": i}
            )
            ids.append(f"{doc['id']}_chunk{i}")

    for i in range(0, len(texts), 50):
        collection.add(
            documents=texts[i : i + 50],
            metadatas=metadatas[i : i + 50],
            ids=ids[i : i + 50],
        )

    print(f"[벡터DB 구축 완료] {collection.count()}개 청크")
    return collection


# ── RAG 클래스 (06_rag_chain.py와 동일 로직) ─────────────────────────────────

def _make_rag_prompt(question: str, context: str) -> str:
    system = (
        "당신은 환자의 증상과 질문을 듣고 의학적 정보를 제공하는 의료 상담 AI입니다.\n"
        "아래 [참고 의료 정보]를 바탕으로 환자의 질문에 친절하고 명확하게 답변하세요.\n"
        "반드시 전문 의료진 상담을 권장하는 내용을 포함하세요.\n"
        "진단이나 처방을 직접 내리지 말고, 가능한 원인과 권장 진료과를 안내하세요."
    )
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{system}\n\n"
        f"[참고 의료 정보]\n{context}"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{question}"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


class MedicalRAG:
    def __init__(self):
        self.collection = None
        self.model      = None
        self.tokenizer  = None
        self.device     = "cuda" if torch.cuda.is_available() else "cpu"
        self._ready     = False
        self._status    = "초기화 대기 중..."

    # ── 로딩 ──────────────────────────────────────────────────────────────────

    def load_vectordb(self):
        self._status = "벡터DB 로드 중..."
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL
        )
        client = chromadb.PersistentClient(path=VECTORDB_DIR)
        try:
            self.collection = client.get_collection(
                name="medical_docs", embedding_function=ef
            )
            print(f"[벡터DB 로드] {self.collection.count()}개 청크")
        except Exception:
            # 컬렉션이 없으면 자동 구축
            self.collection = _build_vectordb()

    def load_model(self):
        self._status = f"LLM 로드 중... ({FINETUNED_MODEL})"
        print(f"[모델 로드] {FINETUNED_MODEL}  device={self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            FINETUNED_MODEL, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            FINETUNED_MODEL,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )
        if self.device == "cpu":
            self.model = self.model.to(self.device)

        self.model.eval()
        print("[모델 로드 완료]")

    def initialize(self):
        try:
            self.load_vectordb()
            self.load_model()
            self._ready  = True
            self._status = "준비 완료"
        except Exception as e:
            self._status = f"로드 실패: {e}"
            raise

    # ── 추론 (06_rag_chain.py generate() 그대로) ──────────────────────────────

    def retrieve(self, question: str, top_k: int = TOP_K) -> str:
        results = self.collection.query(
            query_texts=[question], n_results=top_k
        )
        parts = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            parts.append(f"[{meta['category']}] {meta['title']}\n{doc}")
        return "\n\n".join(parts)

    def generate(self, question: str) -> dict:
        context = self.retrieve(question)
        prompt  = _make_rag_prompt(question, context)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.2,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        answer = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        return {"question": question, "context": context, "answer": answer}


# ── 싱글턴 RAG 인스턴스 (앱 시작 시 백그라운드 로드) ─────────────────────────

rag = MedicalRAG()

def _background_init():
    try:
        rag.initialize()
    except Exception as e:
        print(f"[초기화 오류] {e}")

threading.Thread(target=_background_init, daemon=True).start()


# ── Gradio 예시 질문 ──────────────────────────────────────────────────────────

EXAMPLES = [
    "두통이 3일째 계속되고 구역질도 나는데 어떻게 해야 하나요?",
    "당뇨 진단을 받았는데 식단 관리는 어떻게 해야 하나요?",
    "혈압이 150/95 정도 나오는데 위험한가요?",
    "가슴이 두근거리고 숨이 차요. 심장 문제일까요?",
    "무릎이 계단을 내려갈 때마다 아픈데 무슨 문제일까요?",
    "기침이 3주 넘게 계속되는데 어디서 진료 받아야 하나요?",
]


# ── Gradio 콜백 ───────────────────────────────────────────────────────────────

def answer(question: str):
    """질문 → (답변, 참고문서) 반환"""
    question = question.strip()
    if not question:
        return "질문을 입력해 주세요.", ""

    if not rag._ready:
        return f"모델 준비 중입니다. 잠시 후 다시 시도해 주세요.\n현재 상태: {rag._status}", ""

    try:
        result  = rag.generate(question)
        answer_text  = result["answer"]
        context_text = result["context"]
        return answer_text, context_text
    except Exception as e:
        return f"오류가 발생했습니다: {e}", ""


def status_check():
    return f"시스템 상태: {rag._status}"


# ── Gradio UI (gr.Interface) ──────────────────────────────────────────────────

DESCRIPTION = (
    "파인튜닝된 LLM + RAG(벡터DB 검색) 기반 의료 상담 시스템입니다.\n"
    "증상을 입력하면 관련 의료 정보를 검색하여 답변을 생성합니다.\n\n"
    "⚠️ **주의**: 실제 진단·처방을 대체할 수 없습니다. "
    "응급 상황이라면 즉시 병원을 방문하거나 119에 연락하세요."
)

demo = gr.Interface(
    fn=answer,
    inputs=gr.Textbox(
        label="증상 또는 질문을 입력하세요",
        placeholder="예: 두통이 3일째 계속되고 구역질도 납니다...",
        lines=3,
    ),
    outputs=[
        gr.Textbox(label="AI 답변", lines=10),
        gr.Textbox(label="참고한 의료 정보 (RAG 검색 결과)", lines=10),
    ],
    title="한국어 의료 상담 AI",
    description=DESCRIPTION,
    examples=[[q] for q in EXAMPLES],
    cache_examples=False,
)


# ── 진입점 ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo.launch()
