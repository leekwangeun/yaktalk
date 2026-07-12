# 학습 데이터 합성: 질문 템플릿 × 제품명 사전 → NER(BIO)·의도 분류 데이터
# 실행: python scripts/synthesize_data.py
# 출력: data/synth/ner_{train,val,test}.jsonl, intent_{train,val,test}.jsonl
import json
import random
import re
import sqlite3
import sys
from pathlib import Path

import hgtk

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "drug_light.db"
OUT_DIR = ROOT / "data" / "synth"

random.seed(20260703)

CATEGORY_WORDS = ["감기약", "두통약", "진통제", "해열제", "소염제", "알레르기약", "소화제", "수면제", "변비약"]

# (템플릿, 의도) — {A}/{B} 약 이름 슬롯, {A_이랑}식 조사 마커
PAIR_TEMPLATES = [
    ("{A}{A_이랑} {B} 같이 먹어도 돼?", "병용"),
    ("{A}{A_이랑} {B} 같이 복용해도 되나요?", "병용"),
    ("{A} 먹고 있는데 {B} 먹어도 될까?", "병용"),
    ("{A} 복용 중인데 {B}{B_을} 추가로 먹어도 괜찮나요?", "병용"),
    ("{A}하고 {B} 동시에 복용 가능해?", "병용"),
    ("{A}{A_과} {B}{B_을} 함께 먹으면 안 되나?", "병용"),
    ("{A} 먹은 지 얼마 안 됐는데 {B} 먹어도 됨?", "병용"),
    ("{A}, {B} 병용해도 문제 없을까요?", "병용"),
    ("아까 {A} 먹었는데 지금 {B} 먹어도 괜찮을까", "병용"),
    ("{A}{A_이랑} {B} 궁합 어때?", "병용"),
    ("{A}{A_과} {B} 상호작용 있어?", "병용"),
    ("엄마가 {A} 드시는데 {B}도 드려도 되나요?", "병용"),
    ("할아버지가 {A} 복용중인데 {B} 같이 드려도 될까요?", "병용"),
    ("{A} 처방받았는데 집에 있는 {B} 먹어도 되나", "병용"),
    ("{B} 먹기 전에 {A} 먹었는데 괜찮겠지?", "병용"),
]
SINGLE_TEMPLATES = [
    ("{A} 부작용 뭐가 있어?", "부작용"),
    ("{A} 먹으면 졸린가요?", "부작용"),
    ("{A} 부작용 알려줘", "부작용"),
    ("{A} 먹고 속이 쓰린데 부작용인가요?", "부작용"),
    ("{A}{A_은} 어떤 부작용이 있나요?", "부작용"),
    ("{A} 하루에 몇 번 먹어야 돼?", "복용법"),
    ("{A} 식전에 먹어 식후에 먹어?", "복용법"),
    ("{A} 몇 알씩 먹는 거야?", "복용법"),
    ("{A} 복용법 알려주세요", "복용법"),
    ("임신 중인데 {A} 먹어도 되나요?", "병용"),
    ("임산부가 {A} 복용해도 괜찮아요?", "병용"),
    ("10살 아이한테 {A} 먹여도 되나요?", "병용"),
    ("일곱살 애기 {A} 먹여도 돼?", "병용"),
    ("할머니가 {A} 드셔도 괜찮을까요?", "병용"),
    ("{A} 효능이 뭐야?", "기타"),
    ("{A}{A_은} 무슨 약이야?", "기타"),
    ("{A} 성분이 뭐야?", "성분"),
    ("{A}의 성분 알려줘", "성분"),
    ("{A}에 뭐가 들어있어?", "성분"),
    ("{A}{A_은} 무슨 성분으로 만들어졌어?", "성분"),
    ("{A} 주성분 알려주세요", "성분"),
    ("{A}에 어떤 성분이 함유되어 있나요?", "성분"),
    ("{A} 뭐로 만든 약이야?", "성분"),
    ("{A} 성분 좀 보여줘", "성분"),
]
NO_DRUG_SENTENCES = [
    ("안녕하세요", "기타"), ("고마워", "기타"), ("뭘 할 수 있어?", "기타"),
    ("오늘 날씨 어때?", "기타"), ("머리가 아파요", "기타"), ("감기 걸린 것 같아", "기타"),
    ("열이 나요", "기타"), ("속이 안 좋아", "기타"), ("약국 어디에 있어?", "기타"),
    ("병원 가야 할까?", "기타"), ("배가 아픈데 어떡하지", "기타"), ("잠이 안 와요", "기타"),
]

_JOSA_TABLE = {"이랑": ("이랑", "랑"), "은": ("은", "는"), "이": ("이", "가"),
               "을": ("을", "를"), "과": ("과", "와")}


def pick_josa(word: str, spec: str) -> str:
    with_b, without_b = _JOSA_TABLE[spec]
    last = word[-1]
    if not hgtk.checker.is_hangul(last):
        return with_b  # 숫자/영문으로 끝나면 받침형으로 통일
    return with_b if hgtk.checker.has_batchim(last) else without_b


_SLOT_RE = re.compile(r"\{([AB])(?:_([가-힣]+))?\}")


def render(tpl: str, slots: dict) -> tuple[str, list]:
    out, ents, i = "", [], 0
    for m in _SLOT_RE.finditer(tpl):
        out += tpl[i:m.start()]
        key, josa_spec = m.group(1), m.group(2)
        if josa_spec:
            out += pick_josa(slots[key], josa_spec)
        else:
            start = len(out)
            out += slots[key]
            ents.append([start, len(out), "DRUG"])
        i = m.end()
    out += tpl[i:]
    return out, ents


_VOWEL_CONFUSE = {"ㅐ": "ㅔ", "ㅔ": "ㅐ", "ㅗ": "ㅜ", "ㅜ": "ㅗ", "ㅚ": "ㅞ", "ㅒ": "ㅖ"}


def inject_typo(name: str) -> str:
    """자모 단위 오타 1개 주입 (모음 혼동 / 받침 탈락 / 음절 탈락)."""
    chars = list(name)
    idxs = [i for i, ch in enumerate(chars) if hgtk.checker.is_hangul(ch)]
    if not idxs:
        return name
    i = random.choice(idxs)
    cho, jung, jong = hgtk.letter.decompose(chars[i])
    op = random.random()
    if op < 0.5 and jung in _VOWEL_CONFUSE:
        chars[i] = hgtk.letter.compose(cho, _VOWEL_CONFUSE[jung], jong)
    elif op < 0.8 and jong:
        chars[i] = hgtk.letter.compose(cho, jung, "")
    elif len(chars) > 3:
        del chars[i]
    return "".join(chars)


def load_names() -> list[str]:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT DISTINCT norm_name FROM products
        WHERE (cancel_name IS NULL OR cancel_name='정상')
          AND LENGTH(norm_name) BETWEEN 2 AND 8
    """).fetchall()
    conn.close()
    names = [r[0] for r in rows if re.fullmatch(r"[가-힣a-zA-Z0-9]+", r[0] or "")]
    return names


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    names = load_names()
    print(f"제품명 풀: {len(names):,}개")
    samples = []

    def slot_name() -> str:
        if random.random() < 0.12:
            return random.choice(CATEGORY_WORDS)
        n = random.choice(names)
        return inject_typo(n) if random.random() < 0.20 else n

    for _ in range(25000):
        tpl, intent = random.choice(PAIR_TEMPLATES)
        a, b = slot_name(), slot_name()
        text, ents = render(tpl, {"A": a, "B": b})
        samples.append({"text": text, "entities": ents, "intent": intent})

    for _ in range(12000):
        tpl, intent = random.choice(SINGLE_TEMPLATES)
        text, ents = render(tpl, {"A": slot_name()})
        samples.append({"text": text, "entities": ents, "intent": intent})

    for _ in range(3000):
        text, intent = random.choice(NO_DRUG_SENTENCES)
        samples.append({"text": text, "entities": [], "intent": intent})

    random.shuffle(samples)
    n = len(samples)
    splits = {"train": samples[: int(n * 0.9)],
              "val": samples[int(n * 0.9): int(n * 0.95)],
              "test": samples[int(n * 0.95):]}

    for split, rows in splits.items():
        with open(OUT_DIR / f"ner_{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({"text": r["text"], "entities": r["entities"]},
                                   ensure_ascii=False) + "\n")
        with open(OUT_DIR / f"intent_{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({"text": r["text"], "intent": r["intent"]},
                                   ensure_ascii=False) + "\n")
        print(f"{split}: {len(rows):,}문장")

    from collections import Counter
    print("의도 분포:", Counter(s["intent"] for s in samples))
    print("샘플 3개:")
    for s in samples[:3]:
        print(" ", s["text"], "| ents:", [s["text"][e[0]:e[1]] for e in s["entities"]], "| intent:", s["intent"])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
