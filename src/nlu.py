# 직접 파인튜닝한 KoELECTRA 모델 래퍼 (NER 스팬 추출 + 의도 분류)
# 모델 폴더가 없으면 available=False — 파이프라인은 사전 매칭만으로도 동작한다.
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
NER_DIR = ROOT / "models" / "ner-drug"
INTENT_DIR = ROOT / "models" / "intent-drug"
MAX_LEN = 64


class DrugNER:
    def __init__(self, model_dir: Path = NER_DIR):
        self.available = model_dir.exists()
        if not self.available:
            return
        from transformers import AutoModelForTokenClassification, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
        self.model.eval()

    @torch.no_grad()
    def spans(self, text: str) -> list[tuple[int, int, str]]:
        """(start, end, surface) 목록 — B/I 태그를 문자 오프셋 스팬으로 복원."""
        if not self.available:
            return []
        enc = self.tok(text, return_offsets_mapping=True, return_tensors="pt",
                       truncation=True, max_length=MAX_LEN)
        offsets = enc.pop("offset_mapping")[0].tolist()
        pred = self.model(**enc).logits.argmax(-1)[0].tolist()
        spans, cur = [], None
        for (s, e), p in zip(offsets, pred):
            if s == e:
                continue
            if p == 1:  # B-DRUG
                if cur:
                    spans.append(cur)
                cur = [s, e]
            elif p == 2 and cur:  # I-DRUG
                cur[1] = e
            else:
                if cur:
                    spans.append(cur)
                    cur = None
        if cur:
            spans.append(cur)
        return [(s, e, text[s:e]) for s, e in spans]


class IntentClassifier:
    LABELS = ["기타", "병용", "복용법", "부작용", "성분"]

    def __init__(self, model_dir: Path = INTENT_DIR):
        self.available = model_dir.exists()
        if not self.available:
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()

    @torch.no_grad()
    def predict(self, text: str) -> str:
        if not self.available:
            return self._rule_fallback(text)
        enc = self.tok(text, return_tensors="pt", truncation=True, max_length=MAX_LEN)
        idx = self.model(**enc).logits.argmax(-1).item()
        return self.model.config.id2label[idx]

    @staticmethod
    def _rule_fallback(text: str) -> str:
        if any(k in text for k in ("부작용", "이상반응")):
            return "부작용"
        if any(k in text for k in ("몇 번", "몇 알", "복용법", "식전", "식후", "언제 먹")):
            return "복용법"
        if any(k in text for k in ("같이", "함께", "동시", "병용", "먹어도")):
            return "병용"
        return "기타"
