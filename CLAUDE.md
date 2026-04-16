# 프로젝트 컨텍스트 — 한국어 의료 LLM + RAG 시스템

## 프로젝트 개요
한국 의사 국가고시 데이터(KorMedMCQA)로 오픈소스 LLM을 파인튜닝하고,
RAG 파이프라인을 구축한 End-to-End 의료 AI 시스템.
현재 HuggingFace Spaces에 Gradio UI로 배포 진행 중.

## 기술 스택
- 베이스 모델: beomi/gemma-ko-2b (2B 파라미터)
- 파인튜닝: QLoRA (LoRA rank=4, fp16, T4 GPU)
- 파인튜닝 결과물: jdi1009/checkpoints (HuggingFace 공개)
- RAG: ChromaDB + snunlp/KR-SBERT
- UI: Gradio (app.py)
- 평가: ROUGE/BLEU

## 파일 구조
- 01_kormed_loader.py   : KorMedMCQA 데이터 로드
- 02_preprocessor.py   : 텍스트 전처리
- 03_jsonl_formatter.py: LLaMA3 포맷 변환
- 04_finetune_T4_colab.ipynb : Colab 파인튜닝 노트북
- 05_rag_indexer.py    : ChromaDB 벡터DB 구축
- 06_rag_chain.py      : RAG 파이프라인 (핵심 로직)
- app.py               : Gradio UI (HuggingFace Spaces 진입점)

## HuggingFace Spaces 배포 정보
- Spaces URL: huggingface.co/spaces/jdi1009/medical-rag
- GitHub 레포: github.com/hs103531-netizen/medical-llm-rag
- HF 계정: jdi1009

## Spaces push 명령어
```bash
git push https://jdi1009:HF_TOKEN@huggingface.co/spaces/jdi1009/medical-rag main
```
HF_TOKEN은 huggingface.co/settings/tokens 에서 발급 (Write 권한 필요)

## requirements.txt 주의사항
- Spaces가 Python 3.13 사용 → torch 2.1.0 미지원, 2.5.1 이상 사용
- Spaces가 gradio==4.44.0 강제 설치 → requirements.txt에 gradio 버전 명시 금지
- 파일 인코딩 반드시 UTF-8 (UTF-16 저장 시 빌드 에러 발생)
- 현재 안정 버전:
  torch==2.5.1
  transformers==4.44.0
  peft==0.12.0
  chromadb==0.5.3
  sentence-transformers==3.0.0
  accelerate==0.34.0

## 알려진 에러 & 해결법
1. ModuleNotFoundError: pyaudioloop
   → gradio 버전 충돌. requirements.txt에서 gradio 줄 삭제
2. torch==2.1.0 not found
   → Python 3.13 환경. torch==2.5.1 이상으로 변경
3. rejected (fetch first)
   → git push --force 사용
4. Authentication failed
   → HF 토큰 만료. Invalidate and refresh 후 새 토큰 사용
5. requirements.txt 인코딩 에러
   → VSCode 우하단 인코딩 클릭 → Save with Encoding → UTF-8

## 모델 한계 (면접 대비)
- 파인튜닝 데이터(객관식)와 목표(대화형 상담)의 형식 괴리
- LoRA rank=4, 데이터 1890개로 인한 정확도 제한
- 개선 방향: 더 큰 모델(7B), 대화형 데이터(HealthSearchQA-ko), rank 확대
