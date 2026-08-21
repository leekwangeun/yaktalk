# KoGPT2 응답 생성기 + 사실 검증 게이트
# 모델(models/kogpt2-drug)이 없으면 available=False → responder가 템플릿으로 폴백
import re
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GEN_DIR = ROOT / "models" / "kogpt2-drug"
DB_PATH = ROOT / "data" / "drug_light.db"

PROMPT_END = "\n출력:"


def serialize_facts(drug_a: str, drug_b: str | None, level: str, findings: list) -> str:
    """검사 결과 → 모델 입력 문자열. 학습 데이터 합성과 런타임이 반드시 같은 형식을 쓴다.
    findings: [(check, detail)] — risk_engine.Finding의 check/detail."""
    parts = [f"A={drug_a}", f"B={drug_b or '없음'}", f"판정={level}"]
    for check, detail in findings[:4]:
        parts.append(f"근거={check}:{detail[:80]}")
    return "입력: " + " ; ".join(parts) + PROMPT_END


class ResponseGenerator:
    def __init__(self, model_dir: Path = GEN_DIR, db_path: Path = DB_PATH):
        self.available = model_dir.exists()
        if not self.available:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir)
        self.model.eval()
        # 검증 게이트용 성분명 사전 (출력에 입력에 없는 성분이 나오면 폐기)
        conn = sqlite3.connect(db_path)
        self.ingredient_vocab = {r[0] for r in conn.execute(
            "SELECT DISTINCT mtral_nm FROM mcode_dur_map WHERE LENGTH(mtral_nm) >= 3")}
        conn.close()

    def generate(self, drug_a, drug_b, level, findings) -> str | None:
        """생성 성공 + 검증 통과 시 문장, 실패 시 None(→템플릿 폴백)."""
        if not self.available:
            return None
        prompt = serialize_facts(drug_a, drug_b, level, findings)
        try:
            enc = self.tok(prompt, return_tensors="pt", truncation=True, max_length=256)
            with self.torch.no_grad():
                # 사실 복사가 핵심인 태스크라 샘플링 대신 greedy — 이름·수치 복사 충실도가 관건.
                # no_repeat_ngram_size는 쓰지 않는다: 이 태스크는 약 이름·성분명을 그대로,
                # 때로는 한 문장에 두 번 반복해야 하는데 n-gram 반복을 금지하면 모델이
                # 글자를 비틀어 이름을 망가뜨린다 (예: 아세트아미노펜 → '아세세트아미노에펜').
                out = self.model.generate(
                    **enc, max_new_tokens=120, do_sample=False,
                    pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
                    eos_token_id=self.tok.eos_token_id)
            text = self.tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        except Exception:
            return None
        return text if self._validate(text, prompt, drug_a, drug_b, level) else None

    # ---------- 사실 검증 게이트 ----------
    def _validate(self, text: str, prompt: str, drug_a, drug_b, level) -> bool:
        if not (20 <= len(text) <= 400):
            return False
        # 1) 숫자 검증: 출력의 모든 수치는 입력에 있던 것이어야 함 (용량 지어내기 차단)
        in_nums = set(re.findall(r"\d[\d,\.]*", prompt))
        for n in re.findall(r"\d[\d,\.]*", text):
            if n not in in_nums:
                return False
        # 2) 약 이름 검증: 이름을 줄여 쓰는 건 허용하되 글자가 망가지면 폐기
        for name in filter(None, [drug_a, drug_b]):
            if not self._name_intact(name, text):
                return False
        # 3) 성분 검증: 입력에 없는 성분명이 출력에 등장하면 폐기
        for ing in self.ingredient_vocab:
            if ing in text and ing not in prompt:
                return False
        # 4) 인용 검증: 따옴표/괄호 안의 이름은 입력에 글자 그대로 있어야 함
        #    (모델이 약·성분 이름을 미묘하게 변형하는 것을 차단, 예: '이트라코이나졸')
        for q in re.findall(r"'([^']{2,30})'", text):
            if q not in prompt:
                return False
        for q in re.findall(r"\(([가-힣A-Za-z0-9 ,]{2,30})\)", text):
            if q not in prompt:
                return False
        # 5) 판정 일관성: 등급과 문장 논조가 맞아야 함
        if level == "금기" and not any(k in text for k in ("금기", "안 돼", "안 되", "드시지 마")):
            return False
        if level == "정보없음" and any(k in text for k in ("금기 조합", "드시지 마세요")):
            return False
        # 6) 괄호 짝 검증: 짝이 안 맞으면 이름·효능군 표기가 깨진 것
        #    (예: '같은 효능군해열진단용제)' — 여는 괄호가 없고 용어도 변형됨)
        if text.count("(") != text.count(")") or text.count("[") != text.count("]"):
            return False
        return True

    @staticmethod
    def _name_intact(name: str, text: str) -> bool:
        """약 이름이 온전한 형태로 인용됐는지 확인.

        모델이 긴 이름을 앞부분만 따서 쓰는 것은 자연스러우므로 허용한다.
        (타이레놀정500밀리그람(아세트아미노펜) → "타이레놀정과 ...")
        다만 글자 사이에 공백을 끼워 넣어 이름을 망가뜨리는 것은 막는다.
        (영진아스피린장용정 → "영진아 스피린장용은 ..." )
        """
        ns_name = re.sub(r"\s+", "", name)
        ns_text = re.sub(r"\s+", "", text)
        # 공백을 무시했을 때 출력에 실제로 등장한 가장 긴 접두사
        hit = ""
        for i in range(len(ns_name), 0, -1):
            if ns_name[:i] in ns_text:
                hit = ns_name[:i]
                break
        if len(hit) < min(len(ns_name), 4):
            return False
        # 그 접두사는 출력에도 공백 없이 그대로 있어야 한다
        return hit in text
