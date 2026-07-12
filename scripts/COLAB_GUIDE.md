# Colab 학습 가이드 (KoELECTRA NER + 의도 분류)

로컬 CPU로는 전체 학습이 수 시간 걸리므로 Colab 무료 T4 GPU로 학습한다 (예상: NER 10~15분, 의도 분류 5~10분).

> ⚠️ **의도 분류 5클래스 재학습 (2026-07 업데이트).** 성분 질문을 처리하기 위해 의도 클래스가
> 4개(기타·병용·복용법·부작용) → **5개(+성분)** 로 확장되었다. `data/synth/`의 jsonl은 이미 성분
> 데이터를 포함해 재생성되어 있으므로, 아래 순서대로 **의도 분류만** 다시 학습해 `models/intent-drug`를
> 교체하면 된다. (NER·KoGPT2는 성분 클래스와 무관하므로 재학습 불필요. 현재 프로덕션은 규칙 기반으로
> 성분 질문을 이미 처리하며, 5클래스 모델은 "성분"이라는 단어 없는 패러프레이즈까지 잡기 위한 보강이다.)

## 준비물
- Google 계정 (colab.research.google.com)
- 업로드할 파일: `scripts/train_koelectra.py`, `data/synth/` 폴더의 jsonl 6개

## 순서

### 1. 새 노트북 생성 후 GPU 켜기
런타임 → 런타임 유형 변경 → **T4 GPU** 선택

### 2. 파일 업로드 (셀 실행)
```python
from google.colab import files
files.upload()  # train_koelectra.py 선택
```
```python
import os
os.makedirs("data/synth", exist_ok=True)
os.makedirs("models", exist_ok=True)
from google.colab import files
up = files.upload()  # jsonl 6개 선택
for name in up:
    os.rename(name, f"data/synth/{name}")
```

### 3. 베이스 모델 다운로드 (Colab에서는 HF에서 직접)
```python
!pip -q install transformers
from huggingface_hub import snapshot_download
snapshot_download("monologg/koelectra-small-v3-discriminator",
                  local_dir="models/koelectra-small-v3-discriminator")
```

### 4. 학습 실행
```python
!python train_koelectra.py --task ner --out models/ner-drug --epochs 3 --batch 64 --data-dir data/synth --base models/koelectra-small-v3-discriminator
```
```python
!python train_koelectra.py --task intent --out models/intent-drug --epochs 3 --batch 64 --data-dir data/synth --base models/koelectra-small-v3-discriminator
```
(스크립트를 재업로드한 경우 `--base` 생략 가능 — 경로가 없으면 HF에서 자동 다운로드하도록 폴백 처리됨)
각 실행 마지막의 `test: {...}` 지표(F1/accuracy)를 **캡처해둘 것** — 결과보고서 "성능 평가" 재료.

### 5. 학습된 모델 다운로드
```python
!zip -r ner-drug.zip models/ner-drug
!zip -r intent-drug.zip models/intent-drug
from google.colab import files
files.download("ner-drug.zip")
files.download("intent-drug.zip")
```

### 6. 로컬 배치
zip을 풀어 프로젝트의 `models/ner-drug`, `models/intent-drug`에 놓는다.
로컬 확인: `python scripts/train_koelectra.py`는 필요 없고, 추론은 챗봇 실행 시 자동 로드.

## KoGPT2 응답 생성 학습 (2차 — NER과 같은 방식)

업로드: `train_kogpt2.py` + `data/synth/gpt_{train,val,test}.jsonl` 3개

```python
from huggingface_hub import snapshot_download
snapshot_download("skt/kogpt2-base-v2", local_dir="models/kogpt2-base-v2")
```
```python
!python train_kogpt2.py --out models/kogpt2-drug --epochs 3 --batch 16 --data-dir data/synth --base models/kogpt2-base-v2
```
- 예상 15~20분 (T4). 마지막에 출력되는 **생성 샘플 3개(GEN vs REF)와 test_loss를 캡처** — 보고서 재료.
- GEN이 REF와 비슷한 자연스러운 문장이면 성공. 다운로드:
```python
!zip -r kogpt2-drug.zip models/kogpt2-drug
from google.colab import files
files.download("kogpt2-drug.zip")
```
- 로컬 배치: `models/kogpt2-drug`에 풀면 챗봇이 자동으로 생성 모드로 전환 (검증 실패 시 템플릿 폴백)

## 기대 성능 (합성 테스트셋 기준)
- NER: 스팬 F1 0.95+ (합성 데이터와 같은 분포이므로 높게 나옴 — 보고서에는 직접 작성한 실제 질문 100문장 평가를 함께 실을 것)
- 의도 분류: accuracy 0.98+

## 실제 질문 테스트셋 (팀 작업, 보고서용)
팀원들이 실제로 물어볼 법한 문장 100개를 만들어 `data/synth/ner_real_test.jsonl` 형식으로 저장 →
`--data-dir`로 평가하면 일반화 성능 수치를 얻는다. 합성 vs 실제 성능 차이 분석이 보고서 4장의 핵심 재료.
