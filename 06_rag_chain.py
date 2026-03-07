"""
06_rag_chain.py
---------------
RAG 파이프라인: 벡터DB 검색 + 파인튜닝 모델 답변 생성

흐름:
    사용자 질문
        → 벡터DB에서 관련 의료 문서 검색
        → 검색된 문서 + 질문을 프롬프트에 주입
        → HuggingFace 파인튜닝 모델로 답변 생성

사용법:
    python 06_rag_chain.py
"""

import torch
import chromadb
from chromadb.utils import embedding_functions
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── 설정 ──────────────────────────────────────────────────────────────────────

VECTORDB_DIR  = "./data/vectordb"
EMBED_MODEL   = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
FINETUNED_MODEL = "jdi1009/checkpoints"   # 파인튜닝된 모델
TOP_K         = 2     # 검색할 문서 수
MAX_NEW_TOKENS = 300


# ── 시스템 프롬프트 ────────────────────────────────────────────────────────────

def make_rag_prompt(question: str, context: str) -> str:
    """RAG 프롬프트: 검색된 문서를 컨텍스트로 주입"""
    system = """당신은 환자의 증상과 질문을 듣고 의학적 정보를 제공하는 의료 상담 AI입니다.
아래 [참고 의료 정보]를 바탕으로 환자의 질문에 친절하고 명확하게 답변하세요.
반드시 전문 의료진 상담을 권장하는 내용을 포함하세요.
진단이나 처방을 직접 내리지 말고, 가능한 원인과 권장 진료과를 안내하세요."""

    prompt = (
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
    return prompt


# ── RAG 클래스 ────────────────────────────────────────────────────────────────

class MedicalRAG:
    def __init__(self):
        self.collection = None
        self.model      = None
        self.tokenizer  = None
        self.device     = "cuda" if torch.cuda.is_available() else "cpu"

    def load_vectordb(self):
        """벡터DB 로드"""
        print(f"[벡터DB 로드] {VECTORDB_DIR}")
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL
        )
        client = chromadb.PersistentClient(path=VECTORDB_DIR)
        self.collection = client.get_collection(
            name="medical_docs",
            embedding_function=ef,
        )
        print(f"  총 {self.collection.count()}개 청크 로드 완료")

    def load_model(self):
        """파인튜닝 모델 로드"""
        print(f"\n[모델 로드] {FINETUNED_MODEL}")
        print(f"  디바이스: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            FINETUNED_MODEL,
            trust_remote_code=True,
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
        print("  모델 로드 완료")

    def retrieve(self, question: str, top_k: int = TOP_K) -> str:
        """벡터DB에서 관련 문서 검색"""
        results = self.collection.query(
            query_texts=[question],
            n_results=top_k,
        )
        # 검색된 문서들을 하나의 컨텍스트로 합치기
        docs = results["documents"][0]
        metas = results["metadatas"][0]

        context_parts = []
        for doc, meta in zip(docs, metas):
            context_parts.append(f"[{meta['category']}] {meta['title']}\n{doc}")

        return "\n\n".join(context_parts)

    def generate(self, question: str) -> dict:
        """RAG 파이프라인: 검색 → 프롬프트 생성 → 답변 생성"""
        # 1. 관련 문서 검색
        context = self.retrieve(question)

        # 2. RAG 프롬프트 생성
        prompt = make_rag_prompt(question, context)

        # 3. 토크나이징
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self.device)

        # 4. 답변 생성
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

        # 5. 디코딩 (입력 프롬프트 제거)
        answer = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        return {
            "question": question,
            "context": context,
            "answer": answer,
        }


# ── 메인 실행 ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("의료 상담 RAG 시스템")
    print("=" * 60)

    # RAG 초기화
    rag = MedicalRAG()
    rag.load_vectordb()
    rag.load_model()

    # 테스트 질문
    test_questions = [
        "두통이 3일째 계속되고 구역질도 나는데 어떻게 해야 하나요?",
        "당뇨 진단을 받았는데 식단 관리는 어떻게 해야 하나요?",
        "혈압이 150/95 정도 나오는데 위험한가요?",
        "가슴이 두근거리고 숨이 차요. 심장 문제일까요?",
    ]

    print("\n" + "=" * 60)
    print("RAG 추론 테스트")
    print("=" * 60)

    for question in test_questions:
        print(f"\n질문: {question}")
        result = rag.generate(question)
        print(f"답변: {result['answer']}")
        print("-" * 40)

    # 대화형 모드
    print("\n" + "=" * 60)
    print("대화형 모드 (종료: 'q' 입력)")
    print("=" * 60)
    while True:
        user_input = input("\n질문: ").strip()
        if user_input.lower() in ["q", "quit", "exit", "종료"]:
            print("종료합니다.")
            break
        if not user_input:
            continue
        result = rag.generate(user_input)
        print(f"답변: {result['answer']}")


if __name__ == "__main__":
    main()
