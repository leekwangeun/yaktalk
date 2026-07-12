# KoGPT2 학습용 (검사결과 → 설명 문장) 쌍 데이터 합성
# 실행: python scripts/synthesize_gpt_data.py → data/synth/gpt_{train,val,test}.jsonl
# 핵심: 입력 직렬화·근거 detail 문구를 risk_engine/generator와 완전히 동일한 형식으로 생성
import json
import random
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from generator import serialize_facts  # noqa: E402
from synthesize_data import pick_josa  # noqa: E402

DB = ROOT / "data" / "drug_light.db"
OUT = ROOT / "data" / "synth"
random.seed(20260704)

conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row

PRODUCTS = [r["item_name"] for r in conn.execute(
    "SELECT item_name FROM products WHERE cancel_name IS NULL OR cancel_name='정상'").fetchall()]
INGS = [r["mtral_nm"] for r in conn.execute(
    "SELECT DISTINCT mtral_nm FROM mcode_dur_map WHERE LENGTH(mtral_nm) BETWEEN 3 AND 15").fetchall()]
TABOO = conn.execute("SELECT * FROM dur_mix_taboo WHERE prohibit_content != ''").fetchall()
EFCY = conn.execute("SELECT * FROM dur_efcy_dup").fetchall()
ELDER = conn.execute("SELECT * FROM dur_elderly WHERE prohibit_content != ''").fetchall()
PREG = conn.execute("SELECT * FROM dur_pregnancy WHERE prohibit_content != ''").fetchall()
AGE = conn.execute("SELECT * FROM dur_age_taboo").fetchall()
CAP = conn.execute("SELECT * FROM dur_capacity WHERE mix_type='단일'").fetchall()


def j(w, spec):
    return pick_josa(w, spec)


def t_none(a, b):
    return random.choice([
        f"{a}{j(a,'과')} {b} 조합에서 등록된 금기나 주의 정보는 찾지 못했어요. 다만 모든 상호작용이 데이터에 있는 건 아니니 새로운 약을 함께 드실 땐 확인해 보시는 게 좋아요.",
        f"식약처 데이터 기준으로 {a}{j(a,'과')} {b} 사이에 알려진 병용 문제는 없었어요. 그래도 몸 상태에 따라 다를 수 있으니 참고용으로 봐주세요.",
        f"{a}{j(a,'과')} {b}{j(b,'을')} 함께 복용하면 안 된다는 정보는 확인되지 않았어요. 이상 증상이 느껴지면 복용을 멈추고 상담을 받아보세요.",
        f"두 약({a}, {b}) 사이에 등록된 금기 정보는 없어요. 다만 정보가 없다는 것이 완전히 안전하다는 뜻은 아니에요.",
    ])


def t_dup(a, b, ing, max_qty=None):
    base = random.choice([
        f"{a}{j(a,'과')} {b} 모두 '{ing}' 성분이 들어 있어요. 같은 성분을 두 약으로 겹쳐 드시면 과량 복용이 될 수 있어요.",
        f"두 약({a}, {b})에 '{ing}'{j(ing,'이')} 중복으로 들어 있어요. 모르고 함께 드시면 한 성분을 이중으로 먹게 되는 셈이라 주의가 필요해요.",
        f"{a}에도 {b}에도 '{ing}' 성분이 포함돼 있어요. 함께 드시면 의도치 않게 과량이 될 수 있으니 조심하세요.",
    ])
    if max_qty:
        base += random.choice([
            f" 특히 {ing}{j(ing,'은')} 1일 최대용량({max_qty}) 기준이 있어서 넘지 않게 주의해야 해요.",
            f" {ing}의 1일 최대용량은 {max_qty}로 정해져 있으니 초과하지 않도록 확인하세요.",
        ])
    return base


def t_efcy(a, b, ia, ib, group):
    return random.choice([
        f"{a}의 '{ia}'{j(ia,'과')} {b}의 '{ib}'{j(ib,'은')} 같은 효능군({group})에 속해요. 효과가 겹치는 약을 함께 드시면 과잉 작용 위험이 있어요.",
        f"두 약의 성분({ia}, {ib})이 모두 {group} 계열이에요. 같은 작용을 하는 약이 겹치므로 함께 드시기 전에 확인이 필요해요.",
        f"{a}{j(a,'과')} {b}{j(b,'은')} 성분은 달라도 같은 효능군({group}) 약이에요. 중복 복용이 되지 않도록 주의하세요.",
    ])


def t_taboo(a, b, ia, ib, reason):
    return random.choice([
        f"{a}{j(a,'과')} {b}{j(b,'은')} 함께 복용하면 안 되는 병용금기 조합이에요. '{ia}'{j(ia,'과')} '{ib}' 사이에 {reason} 위험이 보고되어 있어요.",
        f"이 조합({a} + {b})은 식약처 병용금기 목록에 있어요. 사유는 {reason}이에요. 함께 드시지 마세요.",
        f"{a}의 '{ia}'{j(ia,'과')} {b}의 '{ib}'{j(ib,'은')} 병용금기예요. {reason} 우려가 있으니 절대 같이 복용하면 안 돼요.",
    ])


def t_preg(a, ing, grade, reason):
    if grade == "1등급":
        return random.choice([
            f"{a}에 든 '{ing}'{j(ing,'은')} 임부금기 1등급 성분이라 임신 중에는 복용하면 안 돼요.",
            f"'{ing}'{j(ing,'은')} 임부금기 1등급이에요. 임신 중이라면 {a} 복용을 피하고 의사와 상의하세요.",
        ])
    return random.choice([
        f"{a}의 '{ing}'{j(ing,'은')} 임부금기 {grade} 성분이에요. 꼭 필요한 경우에만 의사 판단 하에 복용해야 해요.",
        f"'{ing}'{j(ing,'은')} 임부금기 {grade}로 분류돼 있어요. 임신 중 {a} 복용은 의사와 먼저 상의하는 게 안전해요.",
    ])


def t_elder(a, ing, reason):
    return random.choice([
        f"{a}에 포함된 '{ing}'{j(ing,'은')} 노인주의 성분이에요. 고령자는 이상반응이 더 크게 나타날 수 있어 주의가 필요해요.",
        f"'{ing}'{j(ing,'은')} 고령자 주의 성분으로 등록돼 있어요. 어르신이 {a}{j(a,'을')} 드실 땐 상태를 잘 살펴보세요.",
    ])


def t_age(a, ing, age_base):
    return random.choice([
        f"'{ing}'{j(ing,'은')} {age_base} 연령에는 사용 금기예요. 해당 나이에는 {a}{j(a,'을')} 먹이면 안 돼요.",
        f"{a}의 '{ing}' 성분은 {age_base} 금기로 고시돼 있어요. 이 연령대에는 복용하면 안 돼요.",
    ])


def make_samples(n_total=9000):
    samples = []
    cap_by_name = {}
    for r in CAP:
        cap_by_name.setdefault(r["ingr_name"], r["max_qty"])

    def add(drug_a, drug_b, level, findings, target):
        samples.append({"input": serialize_facts(drug_a, drug_b, level, findings), "output": target})

    for _ in range(n_total // 5):
        a, b = random.sample(PRODUCTS, 2)
        add(a, b, "정보없음", [], t_none(a, b))

    for _ in range(n_total // 5):
        a, b = random.sample(PRODUCTS, 2)
        ing = random.choice(INGS)
        max_qty = cap_by_name.get(ing) if random.random() < 0.5 else None
        detail = f"두 약 모두 '{ing}' 성분을 포함 — 같은 성분을 이중 복용하면 과량 위험"
        if max_qty:
            detail += f" (1일 최대용량 주의: {max_qty})"
        add(a, b, "주의", [("성분중복", detail)], t_dup(a, b, ing, max_qty))

    for _ in range(n_total // 5):
        a, b = random.sample(PRODUCTS, 2)
        r = random.choice(EFCY)
        ia = ib = r["ingr_name"]
        if random.random() < 0.5:
            r2 = random.choice([x for x in EFCY if x["effect_code"] == r["effect_code"]])
            ib = r2["ingr_name"]
        detail = f"'{ia}'({r['sers_name']})와 '{ib}'({r['sers_name']}) — 같은 효능군({r['effect_code']}) 중복"
        add(a, b, "주의", [("효능군중복", detail)], t_efcy(a, b, ia, ib, r["effect_code"]))

    for _ in range(n_total // 5):
        a, b = random.sample(PRODUCTS, 2)
        r = random.choice(TABOO)
        ia, ib, reason = r["ingr_name"], r["mix_ingr_name"], r["prohibit_content"].strip().split("\n")[0][:40]
        detail = f"{ia} + {ib}: {reason}"
        add(a, b, "금기", [("병용금기", detail)], t_taboo(a, b, ia, ib, reason))

    per = n_total // 15
    for _ in range(per):
        a = random.choice(PRODUCTS)
        r = random.choice(PREG)
        reason = r["prohibit_content"].strip().split("\n")[0][:60]
        level = "금기" if r["grade"] == "1등급" else "주의"
        detail = f"{r['ingr_name']} (임부금기 {r['grade']}): {reason}"
        add(a, None, level, [("임부금기", detail)], t_preg(a, r["ingr_name"], r["grade"], reason))
    for _ in range(per):
        a = random.choice(PRODUCTS)
        r = random.choice(ELDER)
        reason = r["prohibit_content"].strip().split("\n")[0][:60]
        detail = f"{r['ingr_name']}: {reason}"
        add(a, None, "주의", [("노인주의", detail)], t_elder(a, r["ingr_name"], reason))
    for _ in range(per):
        a = random.choice(PRODUCTS)
        r = random.choice(AGE)
        detail = f"{r['ingr_name']} ({r['age_base']} 금기): {(r['prohibit_content'] or '').strip()[:60]}"
        add(a, None, "금기", [("연령금기", detail)], t_age(a, r["ingr_name"], r["age_base"]))

    random.shuffle(samples)
    return samples


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    samples = make_samples()
    n = len(samples)
    splits = {"train": samples[: int(n * 0.92)],
              "val": samples[int(n * 0.92): int(n * 0.96)],
              "test": samples[int(n * 0.96):]}
    for split, rows in splits.items():
        with open(OUT / f"gpt_{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"gpt_{split}: {len(rows):,}쌍")
    print("\n샘플:")
    s = samples[0]
    print(" INPUT :", s["input"][:150])
    print(" OUTPUT:", s["output"][:150])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
