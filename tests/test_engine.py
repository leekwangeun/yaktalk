# 매칭 엔진 시나리오 테스트: python tests/test_engine.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from matcher import DrugMatcher
from nlu import DrugNER, IntentClassifier
from responder import Responder
from risk_engine import RiskEngine

ner = DrugNER()
matcher = DrugMatcher(ner=ner)
intent_clf = IntentClassifier()
engine = RiskEngine()
responder = Responder(matcher, engine, intent_clf, generator=None)
passed, failed = 0, 0


def check(label, cond, info=""):
    global passed, failed
    mark = "PASS" if cond else "FAIL"
    if cond:
        passed += 1
    else:
        failed += 1
    print(f"[{mark}] {label} {info}")


# 1. 정상 표기 추출
ms = matcher.extract("타이레놀이랑 판콜에스 같이 먹어도 돼?")
names = {m.name for m in ms if m.status == "confirmed"}
check("정상 표기 추출", {"타이레놀", "판콜에스"} <= names, str(names))

# 2. 오타 보정
ms = matcher.extract("타이래놀이랑 부루팬 괜찮아?")
names = {m.name for m in ms if m.status in ("confirmed", "suggest")}
check("오타 보정 (타이래놀/부루팬)", {"타이레놀", "부루펜"} <= names, str(names))

# 3. 카테고리어 → 되묻기
ms = matcher.extract("감기약이랑 두통약 같이 먹어도 돼?")
cats = [m for m in ms if m.status == "category"]
check("카테고리어 인식", len(cats) == 2, str([(m.name, m.candidates[:2]) for m in cats]))

# 4. 킬러 데모: 타이레놀 + 판콜에스 → 성분중복 + 효능군중복 주의
seq_a = engine.resolve_products("타이레놀")[0]["item_seq"]
seq_b = engine.resolve_products("판콜에스")[0]["item_seq"]
level, findings = engine.check_pair(seq_a, seq_b)
kinds = {f.check for f in findings}
check("타이레놀+판콜 판정=주의", level == "주의", f"level={level}")
check("성분중복 검출", "성분중복" in kinds, str(kinds))
check("효능군중복 검출", "효능군중복" in kinds)
for f in findings:
    print("   -", f.check, "|", f.detail[:70])

# 5. 임산부 모드: 부루펜 → 임부금기 2등급 주의
seq = engine.resolve_products("부루펜")[0]["item_seq"]
level, findings = engine.check_single(seq, pregnant=True)
preg = [f for f in findings if f.check == "임부금기"]
check("임산부 모드 부루펜=주의", level == "주의" and preg, f"level={level}")
if preg:
    print("   -", preg[0].detail[:80])

# 6. 병용금기 실데이터 쌍: 이트라코나졸 + 심바스타틴 제품 검색
rows_a = engine.conn.execute(
    "SELECT item_seq FROM products WHERE norm_name LIKE '%이트라코나졸%' OR item_name LIKE '%이트라코나졸%' LIMIT 1").fetchone()
rows_b = engine.conn.execute(
    "SELECT item_seq FROM products WHERE item_name LIKE '%심바스타틴%' LIMIT 1").fetchone()
if rows_a and rows_b:
    level, findings = engine.check_pair(rows_a[0], rows_b[0])
    check("병용금기 쌍 판정=금기", level == "금기", f"level={level}")
    for f in findings[:2]:
        print("   -", f.check, "|", f.detail[:70])
else:
    check("병용금기 쌍 제품 검색", False, "제품 못 찾음")

# 7. 안전 조합: 알려진 금기 없음
seq_c = engine.resolve_products("베아제")
if seq_c:
    level, findings = engine.check_pair(seq_a, seq_c[0]["item_seq"])
    check("타이레놀+베아제=정보없음", level == "정보없음", f"level={level}, findings={len(findings)}")

# 8. 연령금기: 10살 아이에게 타이레놀 (아세트아미노펜 12세 미만 규칙)
level, findings = engine.check_single(seq_a, age=10)
age_hits = [f for f in findings if f.check == "연령금기"]
check("연령금기 10살+타이레놀=금기", level == "금기" and age_hits, f"level={level}")
if age_hits:
    print("   -", age_hits[0].detail[:75])
# 성인(30살)은 걸리지 않아야 함
level, findings = engine.check_single(seq_a, age=30)
check("연령금기 30살+타이레놀=정보없음", level == "정보없음", f"level={level}")

# 9. 성분중복 근거에 1일 최대용량 보강
level, findings = engine.check_pair(seq_a, seq_b)
dup = next(f for f in findings if f.check == "성분중복")
check("성분중복에 최대용량 표시", "최대용량" in dup.detail, dup.detail[-40:])

# 10. NER 앙상블: 사전에 없는 표현 보완
if ner.available:
    ms = matcher.extract("리세드론산 처방받았는데 칼슘제랑 같이 먹어도 되나")
    surfaces = {m.surface for m in ms}
    check("NER 보완 (칼슘제 인식)", any("칼슘제" in s for s in surfaces), str(surfaces))
else:
    print("[skip] NER 모델 없음")

# 11. 의도 분류 모델
if intent_clf.available:
    cases = [("타이레놀이랑 판콜 같이 먹어도 돼?", "병용"), ("게보린 부작용 뭐야?", "부작용"),
             ("타이레놀 하루 몇 번 먹어?", "복용법"), ("안녕하세요", "기타")]
    ok = all(intent_clf.predict(t) == lab for t, lab in cases)
    check("의도 분류 4케이스", ok)
else:
    print("[skip] 의도 모델 없음")

# 12. 접두 일치: "판콜" → 판콜 계열 되묻기 (파인콜 오매칭 방지)
ms = matcher.extract("타이레놀이랑 판콜 같이 먹어도 돼?")
pancol = next((m for m in ms if m.surface.startswith("판콜")), None)
ok = pancol and all(c.startswith("판콜") for c in (pancol.candidates or [pancol.name]))
check("접두 일치 (판콜→판콜 계열)", bool(ok),
      f"{pancol.status}/{pancol.name}/{pancol.candidates[:3] if pancol.candidates else []}" if pancol else "미검출")

# 13. 미등재 건기식: 오매칭 없이 unknown 처리
ms = matcher.extract("고려은단이랑 타이레놀 같이 먹어도 돼?")
unk = [m for m in ms if m.status == "unknown"]
conf = {m.name for m in ms if m.status == "confirmed"}
check("건기식 미등재=unknown", len(unk) >= 1 and conf == {"타이레놀"}, f"unknown={len(unk)}, conf={conf}")

# --- 사용자 제출 실패 TC 회귀 (TC.docx) ---
# TC-03: 3개 이상 약물 전체 pairwise 비교
ms = matcher.extract("타이레놀, 판피린나이트, 이부프로펜 같이 먹어도 돼?")
conf = [m for m in ms if m.status == "confirmed"]
check("TC-03 3개 약물 모두 인식", len(conf) == 3, str([m.name for m in conf]))
out = responder.handle("타이레놀, 판피린나이트, 이부프로펜 같이 먹어도 돼?")
check("TC-03 3개 조합 판정", "3개를 모든 조합" in out.reply, out.reply[:40])

# TC-05: 띄어쓰기 없는 입력
ms = matcher.extract("타이레놀이랑게보린같이먹어도돼?")
names = {m.name for m in ms if m.status == "confirmed"}
check("TC-05 붙여쓰기 분해", {"타이레놀", "게보린"} <= names, str(names))

# TC-07: 순수 성분(제품 없음) 병용금기 — 플루복사민+티자니딘
out = responder.handle("티자니딘이랑 플루복사민 같이 먹어도 돼?")
check("TC-07 순수성분 병용금기", out.level == "금기" and "병용금기" in out.reply, f"level={out.level}")

# TC-09: 성분중복에 약별 함량 표시
out = responder.handle("타이레놀이랑게보린같이먹어도돼?")
check("TC-09 약별 함량 표시", "밀리그램" in out.reply or "밀리그람" in out.reply, "")

# TC-01: 광범위 전문약 통칭은 건기식과 구분된 안내
out = responder.handle("혈압약 먹어도 돼?")
check("TC-01 광범위 통칭 안내", "종류가 매우 많아" in out.reply, out.reply[:30])

# TC-04: 술/음식 안내
out = responder.handle("타이레놀 먹고 술 마셔도 돼?")
check("TC-04 음주 안내", "음주" in out.reply or "음식" in out.reply, "")

# 어절 중간 오매칭 방지 유지 (마이프로틴 속 프로틴)
ms = matcher.extract("마이프로틴 먹으면서 타이레놀 먹어도 되나")
check("어절중간 오매칭 방지", not any(m.name == "프로틴" for m in ms), str([m.name for m in ms]))

# 확정 2개 + 접두일치 세 번째 약(판콜)은 무시되지 않고 되묻기해야 함
out = responder.handle("게보린 타이레놀 판콜 같이 먹어도 돼?")
picked_ok = out.clarify and any("판콜" in o for o in out.clarify["options"])
check("확정2+세번째약 되묻기", bool(picked_ok), f"clarify={'Y' if out.clarify else 'N'}")
# 되묻기 후 3개 전수 비교
out2 = responder.handle("게보린이랑 타이레놀이랑 판콜에스 같이 먹어도 돼?")
check("판콜 확정 후 3개 비교", "3개를 모든 조합" in out2.reply, out2.reply[:30])
# 저신뢰 잡음은 정상 2약 판정을 방해하지 않아야 함
out3 = responder.handle("타이레놀이랑 판콜에스 같이 먹어도 돼?")
check("잡음 되묻기 억제 유지", out3.clarify is None and out3.level == "주의", f"clarify={out3.clarify is not None}")

# 성분 질문 — 성분 목록으로 응답 (부작용으로 오라우팅 금지)
for q in ["게보린의 성분", "게보린정 성분알려줘", "판콜에스 성분이 뭐야?"]:
    out = responder.handle(q)
    ok = "주성분" in out.reply and "부작용 안내" not in out.reply
    check(f"성분 질문 [{q[:8]}]", ok, out.reply[:30])
# 성분 응답에 실제 성분·함량이 들어가야 함
out = responder.handle("게보린 성분 알려줘")
check("성분 응답 내용", "아세트아미노펜" in out.reply and "밀리그램" in out.reply, "")

print()
print(f"결과: {passed} PASS / {failed} FAIL")
sys.exit(1 if failed else 0)
