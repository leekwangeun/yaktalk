# 100문장 테스트셋 성능 평가 → 콘솔 요약 + data/eval_report.md
# 실행: python scripts/evaluate.py
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from matcher import DrugMatcher  # noqa: E402
from nlu import DrugNER, IntentClassifier  # noqa: E402
from responder import Responder  # noqa: E402
from risk_engine import RiskEngine  # noqa: E402

TESTSET = ROOT / "tests" / "testset_100.jsonl"
REPORT = ROOT / "data" / "eval_report.md"


def rel(a: str, b: str) -> bool:
    return bool(a) and bool(b) and (a in b or b in a)


def eval_extraction(expected: list, matches) -> tuple[bool, int, int, int]:
    """(질문 성공 여부, hit 수, 기대 수, 오탐 수)"""
    hits = 0
    used = set()
    for e in expected:
        if e.startswith("CATEGORY:"):
            name = e.split(":", 1)[1]
            ok = any(m.status == "category" and m.name == name for m in matches)
        elif e.startswith("UNKNOWN:"):
            name = e.split(":", 1)[1]
            related = [m for m in matches if rel(name, m.surface)]
            ok = all(m.status == "unknown" for m in related)  # 없거나 전부 unknown이면 통과
        else:
            ok = False
            for i, m in enumerate(matches):
                if m.status in ("confirmed", "suggest", "category") and i not in used:
                    if rel(e, m.name or "") or any(rel(e, c) for c in m.candidates):
                        ok = True
                        used.add(i)
                        break
        if ok:
            hits += 1

    plain = [e for e in expected if not e.startswith(("CATEGORY:", "UNKNOWN:"))]
    fp = 0
    for m in matches:
        if m.status != "confirmed":
            continue
        if not any(rel(e, m.name or "") for e in plain) and \
           not any(rel(e.split(":", 1)[1], m.surface) for e in expected if ":" in e):
            fp += 1
    return hits == len(expected) and fp == 0, hits, len(expected), fp


def main():
    rows = [json.loads(l) for l in open(TESTSET, encoding="utf-8")]
    print(f"테스트셋: {len(rows)}문장 로드")

    matcher = DrugMatcher(ner=DrugNER())
    responder = Responder(matcher, RiskEngine(), IntentClassifier(), generator=None)
    intent_clf = responder.intent_clf

    stats = defaultdict(lambda: [0, 0])  # type → [성공, 전체]
    ext_ok = 0
    tp = fn = fp_total = 0
    intent_ok = intent_n = 0
    level_ok = level_n = 0
    failures = []

    for r in rows:
        matches = matcher.extract(r["q"])
        ok, hits, total, fp = eval_extraction(r["drugs"], matches)
        stats[r["type"]][1] += 1
        if ok:
            ext_ok += 1
            stats[r["type"]][0] += 1
        else:
            got = [(m.status, m.surface, m.name) for m in matches]
            failures.append(f"#{r['id']} [{r['type']}] {r['q']}\n    기대={r['drugs']} 실제={got}")
        tp += hits
        fn += total - hits
        fp_total += fp

        if "intent" in r:
            intent_n += 1
            if intent_clf.predict(r["q"]) == r["intent"]:
                intent_ok += 1

        if "level" in r:
            level_n += 1
            reply = responder.handle(r["q"], **r.get("modes", {}))
            if reply.level == r["level"]:
                level_ok += 1
            else:
                failures.append(f"#{r['id']} [판정] {r['q']} 기대={r['level']} 실제={reply.level}")

    prec = tp / (tp + fp_total) if tp + fp_total else 0
    rec = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0

    lines = []
    lines.append("# 성능 평가 리포트 (실사용 스타일 질문 100문장)\n")
    lines.append(f"- 약 이름 추출 — 질문 단위 성공률: **{ext_ok}/{len(rows)} ({ext_ok/len(rows)*100:.0f}%)**")
    lines.append(f"- 약 이름 추출 — 엔티티 정밀도 {prec:.3f} / 재현율 {rec:.3f} / **F1 {f1:.3f}**")
    lines.append(f"- 의도 분류 정확도: **{intent_ok}/{intent_n} ({intent_ok/intent_n*100:.0f}%)**")
    lines.append(f"- 위험도 판정 정확도(라벨 보유 {level_n}건): **{level_ok}/{level_n} ({level_ok/level_n*100:.0f}%)**\n")
    lines.append("## 유형별 추출 성공률")
    for t, (s, n) in stats.items():
        lines.append(f"- {t}: {s}/{n} ({s/n*100:.0f}%)")
    lines.append("\n## 실패 사례")
    lines.extend(f"- {f}" for f in failures) if failures else lines.append("- 없음")

    report = "\n".join(lines)
    print()
    print(report)
    REPORT.write_text(report, encoding="utf-8")
    print(f"\n저장: {REPORT}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
