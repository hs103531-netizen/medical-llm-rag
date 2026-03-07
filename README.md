# 🏥 한국어 의료 도메인 LLM 파인튜닝 + RAG 시스템

> 한국어 의료 국가고시 데이터로 LLM을 파인튜닝하고,  
> RAG(검색 증강 생성) 파이프라인을 구축한 End-to-End 프로젝트입니다.

---

## 📌 프로젝트 개요

| 항목 | 내용 |
|------|------|
| 목적 | 의료 도메인 특화 LLM 파인튜닝 및 RAG 시스템 구축 |
| 데이터 | KorMedMCQA (한국 의사 국가고시 2012~2024, 1,890개) |
| 파인튜닝 방법 | QLoRA (LoRA rank=4, fp16, T4 GPU) |
| RAG | ChromaDB + 한국어 SBERT 임베딩 |
| 배포 모델 | [jdi1009/checkpoints](https://huggingface.co/jdi1009/checkpoints) |

---

## 🏗️ 시스템 아키텍처

```
[데이터 수집]         [파인튜닝]               [RAG 시스템]
KorMedMCQA      →   beomi/gemma-ko-2b    →   ChromaDB
(1,890개)            QLoRA (T4 GPU)           + KR-SBERT
                          ↓                        ↓
                  jdi1009/checkpoints  →    의료 상담 답변
```

---

## 🛠️ 개발 환경

### 로컬 환경 (데이터 파이프라인)
| 항목 | 사양 |
|------|------|
| OS | Windows 11 |
| Python | 3.10 |
| IDE | VSCode |
| 가상환경 | venv (medical_env) |

### 클라우드 환경 (파인튜닝 / RAG 추론)
| 항목 | 사양 |
|------|------|
| 플랫폼 | Google Colab |
| GPU | Tesla T4 (VRAM 15.6GB) |
| Python | 3.12 |
| 스토리지 | Google Drive 연동 |

---

## 📦 사용 언어 및 주요 라이브러리

### 언어
- **Python 3.10 / 3.12**

### 딥러닝 프레임워크
| 라이브러리 | 버전 | 용도 |
|-----------|------|------|
| PyTorch | 2.x | 모델 학습 및 추론 |
| transformers | 4.44.0 | HuggingFace 모델 로드/학습 |
| peft | 0.12.0 | QLoRA (LoRA 어댑터) 적용 |
| trl | 0.9.6 | SFTTrainer (지도 파인튜닝) |
| accelerate | 0.34.0 | device_map 및 학습 가속 |
| datasets | 최신 | HuggingFace 데이터셋 로드 |

### RAG / 벡터DB
| 라이브러리 | 용도 |
|-----------|------|
| chromadb | 벡터 데이터베이스 |
| sentence-transformers | 문서 임베딩 생성 |

### 평가
| 라이브러리 | 용도 |
|-----------|------|
| rouge-score | ROUGE-1/2/L 계산 |
| nltk | BLEU 계산 |
| numpy | 수치 통계 처리 |

---

## 🤖 사용 모델 상세

### 1. beomi/gemma-ko-2b (베이스 LLM)
- **출처**: Google Gemma-2B 기반 → beomi(이준범)가 한국어 데이터로 추가 사전학습
- **파라미터 수**: 2B (20억)
- **특징**: 한국어 이해 및 생성 능력 강화, T4 GPU에서 fp16으로 안정적 실행 가능
- **라이선스**: Gemma Terms of Use (비상업적 연구 허용)
- **링크**: [huggingface.co/beomi/gemma-ko-2b](https://huggingface.co/beomi/gemma-ko-2b)

### 2. jdi1009/checkpoints (파인튜닝 모델 · 본 프로젝트 산출물)
- **베이스**: beomi/gemma-ko-2b
- **파인튜닝 데이터**: KorMedMCQA doctor split (train 1,890개)
- **파인튜닝 방법**: QLoRA
  - LoRA rank: 4 / LoRA alpha: 8
  - target_modules: q_proj, k_proj, v_proj, o_proj
  - 정밀도: fp16 (bitsandbytes 미사용)
  - 옵티마이저: adamw_torch
- **학습 환경**: Google Colab Tesla T4
- **eval loss**: 0.7898
- **링크**: [huggingface.co/jdi1009/checkpoints](https://huggingface.co/jdi1009/checkpoints)

### 3. snunlp/KR-SBERT-V40K-klueNLI-augSTS (RAG 임베딩 모델)
- **출처**: 서울대학교 NLP 연구실 (snunlp)
- **기반 모델**: Sentence-BERT (SBERT)
- **학습 데이터**: KorNLI, KorSTS, KLUE-NLI (어휘 40K)
- **용도**: 의료 문서를 벡터로 변환 → ChromaDB 저장 → 질문과 유사 문서 검색
- **특징**: 한국어 문장 의미 유사도 측정에 최적화된 모델
- **링크**: [huggingface.co/snunlp/KR-SBERT-V40K-klueNLI-augSTS](https://huggingface.co/snunlp/KR-SBERT-V40K-klueNLI-augSTS)

---

## 📊 학습 데이터

### KorMedMCQA
| 항목 | 내용 |
|------|------|
| 출처 | [sean0042/KorMedMCQA](https://huggingface.co/datasets/sean0042/KorMedMCQA) |
| 내용 | 한국 의사 국가고시 기출문제 (2012~2024년) |
| 형식 | 5지선다 객관식 (A~E 보기) |
| train | 1,890개 |
| validation | 164개 |
| test | 435개 |
| 전처리 | 질문+보기 → LLaMA3 chat 포맷 JSONL 변환 |

---

## 📈 평가 결과

| 지표 | 점수 |
|------|------|
| ROUGE-1 | 0.2267 |
| ROUGE-2 | 0.0000 |
| ROUGE-L | 0.2267 |
| BLEU | **0.4151** |

- **평가 데이터**: KorMedMCQA doctor test 50개
- **BLEU 0.41**: 정답 형식(`정답: X.`) 준수율 양호
- **ROUGE-2 = 0**: 평균 응답 길이 19자로 짧아 2-gram 측정 어려움 (정상)

---

## 📁 파일 구조

```
medical_llm/
│
├── 01_kormed_loader.py         # KorMedMCQA 로드 → JSONL 변환
├── 02_preprocessor.py          # 텍스트 전처리 (필드 자동 감지)
├── 03_jsonl_formatter.py       # LLaMA3 포맷 변환, train/val 분리
├── 04_finetune_T4_colab.ipynb  # Colab T4 파인튜닝 노트북
├── 05_rag_indexer.py           # 의료 문서 → ChromaDB 벡터DB 구축
├── 06_rag_chain.py             # RAG 파이프라인 (검색 + 생성)
├── 07_evaluation.py            # ROUGE/BLEU 평가 코드
└── README.md
```

---

## 🚀 실행 방법

### 1. 환경 설정
```bash
git clone https://github.com/jdi1009/medical-llm-rag
cd medical-llm-rag
python -m venv medical_env
medical_env\Scripts\activate      # Windows
pip install -r requirements.txt
```

### 2. 데이터 준비
```bash
python 01_kormed_loader.py
# → data/jsonl/train.jsonl (1,890개)
# → data/jsonl/val.jsonl (164개)
```

### 3. 파인튜닝 (Google Colab T4)
```
Google Drive > MyDrive/medical_llm/data/jsonl/ 에 업로드 후
04_finetune_T4_colab.ipynb 순서대로 실행
```

### 4. RAG 벡터DB 구축
```bash
python 05_rag_indexer.py
# → data/vectordb/ 생성
```

### 5. 평가 (Colab)
```
07_evaluation.py 내 셀 코드를 Colab에 붙여넣어 실행
```

---

## ⚙️ 파인튜닝 상세 설정

| 항목 | 값 |
|------|----|
| 베이스 모델 | beomi/gemma-ko-2b |
| 정밀도 | fp16 |
| LoRA rank | 4 |
| LoRA alpha | 8 |
| LoRA dropout | 0.1 |
| target_modules | q_proj, k_proj, v_proj, o_proj |
| max_seq_length | 256 |
| batch_size | 1 |
| gradient_accumulation | 4 (유효 배치 4) |
| 옵티마이저 | adamw_torch |
| 학습률 | 2e-4 |
| epochs | 3 |
| GPU | Tesla T4 (VRAM 15.6GB) |

---

## 🔧 주요 트러블슈팅

### bitsandbytes triton.ops 충돌
- **문제**: Colab T4에서 bitsandbytes 로드 시 `triton.ops` AttributeError 발생
- **해결**: bitsandbytes 완전 제거 → fp16 직접 로드 + `adamw_torch` 옵티마이저 사용

### HuggingFace 모델 인증 오류 (401 Unauthorized)
- **문제**: 업로드된 모델이 private 상태라 로컬에서 접근 불가
- **해결**: HuggingFace 모델 페이지 Settings → public으로 변경

### 로컬 CPU 메모리 부족 (OOM)
- **문제**: 5GB 모델을 로컬 CPU에서 로드 시 메모리 초과
- **해결**: Google Colab T4 GPU 환경에서 추론 실행

---

## 📝 한계 및 향후 개선 방향

- 파인튜닝 데이터(국가고시 객관식)와 실제 목표(대화형 상담)의 데이터 형식 괴리 존재
- `HealthSearchQA-ko` 등 대화형 의료 QA 데이터로 재학습 시 상담 품질 향상 가능
- 의료 면책 조항 강화 필요 (진단/처방 불가 명시)
- 더 큰 모델(7B 이상) 사용 시 답변 품질 향상 기대

---

## 👤 개발자

- **HuggingFace**: [jdi1009](https://huggingface.co/jdi1009)
- **GitHub**: [hs103531-netizen](https://github.com/hs103531-netizen)
