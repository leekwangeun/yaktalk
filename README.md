# 약톡 (YakTalk) — 약물 상호작용 확인 챗봇

여러 약을 함께 복용해도 되는지 자연어로 물어보면, 식약처 공공데이터(DUR·의약품 허가정보)를 근거로
**병용금기·성분중복·효능군중복·노인주의·임부금기** 등을 검사해 알려주는 챗봇입니다.

> 제2회 파이썬 SW 활용 경진대회 · **심화활용(Innovation, 기술기반)** 분야 — 공공데이터 활용

## 라이브 데모

- **백엔드 API (챗봇 구동 중)**: https://kwangeun2-yaktalk-api.hf.space
- 프론트엔드: Vercel 배포 예정

예) *"타이레놀이랑 판콜 같이 먹어도 돼?"* → 두 약 모두 아세트아미노펜 함유 → **성분중복 주의** 안내

## 특징

- **외부 AI API 미사용** — 모든 모델을 직접 파인튜닝한 로컬 가중치로 동작
- **공공데이터 기반 판정** — 식약처 DUR 규칙 + 의약품 허가정보(e약은요)
- **안전 우선 설계** — 생성 응답이 근거(약 이름·성분·수치)를 왜곡하면 폐기하고 검증된 템플릿으로 폴백

## 아키텍처

```
사용자 질문
   │
   ▼
[NLU]  KoELECTRA 파인튜닝 — 의도 분류 + 약물명 NER
   │   + 사전/오토마타 매칭(오타 보정, 자모 유사도)
   ▼
[판정] 규칙 엔진 — DUR 병용금기·성분중복·효능군중복·노인/임부/연령 주의
   │
   ▼
[응답] KoGPT2 파인튜닝 생성 + 4중 검증 게이트
       (검증 실패 시 템플릿 폴백)
```

- **NLU**: `src/nlu.py`, `src/matcher.py`, `src/dictionary.py`
- **판정 엔진**: `src/risk_engine.py`
- **응답 생성**: `src/generator.py`, `src/responder.py`
- **API**: `src/api.py` (FastAPI)

## 성능

100문장 실제 질문 테스트셋(`tests/testset_100.jsonl`) 평가 결과:

| 항목 | 결과 |
|---|---|
| 약물명 추출 | 98% (F1 0.993, 정밀도 1.000) |
| 의도 분류 | 95% |
| 위험 판정 | 22/22 (100%) |

평가 재현: `python scripts/evaluate.py` · 상세 리포트: [`data/eval_report.md`](data/eval_report.md)

## 기술 스택

Python · FastAPI · PyTorch · Transformers(KoELECTRA, KoGPT2) · SQLite · HTML/JS(프론트)

## 실행 방법

### 데모는 위 라이브 주소로 바로 사용 가능합니다.

### 로컬에서 백엔드를 직접 돌리려면 (백엔드 수정·테스트 시)

모델 가중치와 경량 DB는 용량 문제로 저장소에 포함되지 않습니다.
Hugging Face Space([kwangeun2/yaktalk-api](https://huggingface.co/spaces/kwangeun2/yaktalk-api))에서
`models/`(ner-drug·intent-drug·kogpt2-drug)와 `data/drug_light.db`를 받아 배치한 뒤 실행하세요.

```bash
pip install -r requirements.txt
uvicorn api:app --app-dir src --port 8000
# http://localhost:8000 에서 채팅 UI까지 서빙
```

## 데이터·모델 재현

- 데이터 수집: `scripts/collect_data.py` (식약처 공공데이터 API)
- 경량 DB 전처리: `scripts/build_light_db.py`
- 학습 데이터 합성: `scripts/synthesize_data.py`, `scripts/synthesize_gpt_data.py` (합성 결과는 `data/synth/`)
- 모델 학습: `scripts/train_koelectra.py`, `scripts/train_kogpt2.py`

## 프로젝트 구조

```
src/        핵심 로직 (NLU·판정 엔진·응답 생성·API)
scripts/    데이터 수집·전처리·합성·학습·평가 스크립트
data/synth/ 합성 학습 데이터
tests/      테스트 + 100문장 평가 테스트셋
web/        프론트엔드 채팅 UI
Dockerfile  백엔드 배포용 (Hugging Face Spaces)
```
