# 응답 생성 (v1: 슬롯 채움 템플릿 — KoGPT2 학습 후 생성 모델 + 검증 게이트로 교체 예정)
import random
import sqlite3
from dataclasses import dataclass, field

from dictionary import pick_josa

DISCLAIMER = "\n\n※ 이 답변은 식약처 공공데이터 기반 참고 정보예요. 정확한 판단은 약사·의사와 상담하세요."

LEVEL_HEAD = {
    "금기": ["🔴 함께 복용하면 안 되는 조합이에요.", "🔴 이 조합은 병용이 금지되어 있어요."],
    "주의": ["🟠 주의가 필요한 조합이에요.", "🟠 함께 드시기 전에 꼭 확인이 필요해요."],
    "정보없음": ["🟢 식약처 데이터에서 알려진 금기 정보는 찾지 못했어요.",
              "🟢 두 약 사이에 등록된 금기·주의 정보는 없었어요."],
}


@dataclass
class BotReply:
    reply: str
    level: str | None = None
    intent: str = "병용"
    drugs: list = field(default_factory=list)
    clarify: dict | None = None
    findings: list = field(default_factory=list)


def _fmt_findings(findings) -> str:
    lines = []
    for f in findings:
        detail = f.detail if len(f.detail) <= 150 else f.detail[:150] + "…"
        lines.append(f"• [{f.check}] {detail}")
    return "\n".join(lines)


LEVEL_EMOJI = {"금기": "🔴", "주의": "🟠", "정보없음": "🟢"}

# 종류가 매우 많은 전문약 통칭 — 특정 불가하므로 '건기식 미지원'과 구분해 안내
GENERIC_RX_WORDS = {"혈압약", "고혈압약", "혈압강하제", "당뇨약", "혈당약", "당뇨병약",
                    "고지혈증약", "콜레스테롤약", "갑상선약", "항생제", "항생재", "항응고제",
                    "수면제", "신경안정제", "항우울제", "우울증약", "스테로이드", "항암제",
                    "혈전약", "혈액응고제", "위장약", "제산제", "항히스타민제"}
# 약↔음식/술 상호작용은 v1 범위 밖 — 언급 시 안내
FOOD_ALCOHOL_WORDS = ["술", "음주", "맥주", "소주", "와인", "알코올", "알콜",
                      "커피", "카페인", "우유", "자몽", "홍삼", "녹차"]
# 성분 질문 신호어
INGREDIENT_WORDS = ["성분", "뭐가 들어", "뭐 들어", "무엇이 들어", "어떤 성분",
                    "무슨 성분", "성분이 뭐", "함유", "들어있", "들어 있"]


class Responder:
    def __init__(self, matcher, engine, intent_clf, generator=None):
        self.matcher = matcher
        self.engine = engine
        self.intent_clf = intent_clf
        self.generator = generator  # KoGPT2 생성기 (없거나 검증 실패 시 템플릿 폴백)

    def _generated(self, drug_a, drug_b, level, findings) -> str | None:
        if self.generator is None or not getattr(self.generator, "available", False):
            return None
        return self.generator.generate(drug_a, drug_b, level,
                                       [(f.check, f.detail) for f in findings])

    def handle(self, message: str, elderly=False, pregnant=False, age=None) -> BotReply:
        intent = self.intent_clf.predict(message)
        matches = self.matcher.extract(message)

        confirmed = [m for m in matches if m.status == "confirmed"]
        suggests = [m for m in matches if m.status == "suggest"]
        categories = [m for m in matches if m.status == "category"]
        unknowns = [m for m in matches if m.status == "unknown"]

        notice = ""
        for m in unknowns:
            token = self.matcher.dic.strip_josa(m.surface)
            if token in GENERIC_RX_WORDS or any(token.endswith(w) for w in GENERIC_RX_WORDS):
                notice += (f"'{token}'{pick_josa(token, '은')} 종류가 매우 많아 특정하기 어려워요. "
                           "정확한 제품명이나 성분명(예: 암로디핀)을 알려주시면 확인해 드릴게요.\n")
            else:
                notice += (f"'{token}'{pick_josa(token, '은')} 의약품 데이터베이스에서 찾지 못했어요. "
                           "건강기능식품이나 해외 제품은 아직 지원하지 않아요.\n")

        # 약↔음식/술 상호작용 안내 (v1 범위 밖)
        food = next((w for w in FOOD_ALCOHOL_WORDS if w in message), None)
        if food:
            notice += (f"참고로 약과 {food} 등 음식·음주와의 상호작용은 아직 지원하지 않아요. "
                       "약과 약 사이만 확인해 드릴 수 있어요.\n")

        # 되묻기: 애매한 이름이나 카테고리어가 있으면 후보 제시.
        # 카테고리어·접두일치(고신뢰, score≥95)는 사용자가 실제로 넣은 약이므로 확정 개수와
        # 무관하게 되묻는다. 저신뢰 유사도 제안만 확정 2개 이상일 때 잡음으로 무시한다.
        pend = suggests + categories
        strong = [m for m in pend if m.status == "category" or m.score >= 95]
        ask = strong if strong else (pend if len(confirmed) < 2 else [])
        if ask:
            m = ask[0]
            what = "말씀하신 약" if m.status == "suggest" else f"'{m.surface}'"
            q = f"{what}{pick_josa(what.rstrip(chr(39)) or '약', '이')} 아래 중 어떤 건가요?"
            # rest: 아직 해소 안 된 다른 되묻기 대상 (확정 약은 웹 UI가 drugs로 이미 반영)
            rest = [x.surface for x in pend if x is not m]
            return BotReply(reply=notice + q, intent=intent,
                            drugs=[self._drug_info(x) for x in confirmed],
                            clarify={"question": q, "options": m.candidates[:5], "rest": rest})

        # 확정된 약을 엔티티(제품 또는 순수 성분)로 해석
        entities = []
        for mt in confirmed:
            ent = self._resolve_entity(mt)
            if ent:
                entities.append(ent)

        # 성분 질문 처리 — 키워드 규칙 OR 의도 모델의 '성분' 분류(재학습 후) 이중 감지.
        # 약 1개 + 성분 질문이면 성분 목록으로 응답한다.
        is_ingredient_q = intent == "성분" or any(k in message for k in INGREDIENT_WORDS)
        if is_ingredient_q and len(entities) == 1:
            return self._ingredient_answer(entities[0], notice)

        # 약이 2개 이상이면 의도와 무관하게 병용 판정 우선 (의도 오분류가 판정을 막지 않게)
        if intent in ("부작용", "복용법") and len(entities) == 1 \
                and not (elderly or pregnant or age is not None) and not is_ingredient_q:
            return self._easy_info(confirmed[0], intent, notice)

        if len(entities) >= 3:
            return self._multi_answer(entities, elderly, pregnant, age, notice, intent)
        if len(entities) >= 2:
            return self._pair_answer(entities[0], entities[1], elderly, pregnant, age, notice, intent)
        if len(entities) == 1 and (elderly or pregnant or age is not None):
            return self._single_answer(entities[0], elderly, pregnant, age, notice, intent)
        if len(entities) == 1:
            name = entities[0].display
            return BotReply(reply=notice + f"'{name}'{pick_josa(name, '을')} 확인했어요. "
                            "어떤 약과 함께 드시는지 알려주시면 상호작용을 확인해 드릴게요.",
                            intent=intent, drugs=[{"name": name, "item_name": name}])
        if notice:
            return BotReply(reply=notice.rstrip(), intent=intent)
        return BotReply(reply="어떤 약인지 이름을 알려주시면 확인해 드릴게요. "
                        "예: \"타이레놀이랑 판콜 같이 먹어도 돼?\"", intent=intent)

    # ---------- 내부 ----------
    def _resolve_seq(self, match):
        rows = self.engine.resolve_products(match.name)
        if rows:
            return rows[0]
        # 사전이 이미 아는 이름(염 제거 성분명 등)은 문자열 재조회 대신 item_seq로 직접 해석
        seqs = self.matcher.dic.name_to_seqs.get(match.name)
        rows = self.engine.products_by_seqs(seqs or set())
        return rows[0] if rows else None

    def _resolve_entity(self, match):
        """확정 매칭 → Entity. 제품 우선, 없으면 순수 DUR 성분(D코드)."""
        row = self._resolve_seq(match)
        if row:
            return self.engine.make_entity(row["item_seq"], row["item_name"])
        dcode = self.matcher.dic.name_to_dcode.get(match.name)
        if dcode:
            return self.engine.entity_from_dcode(match.name, dcode)
        return None

    def _drug_info(self, match):
        row = self._resolve_seq(match)
        return {"surface": match.surface, "name": match.name,
                "item_name": row["item_name"] if row else match.name}

    def _ent_info(self, e):
        return {"name": e.display, "item_name": e.display}

    def _findings_json(self, findings):
        return [{"check": f.check, "level": f.level, "detail": f.detail} for f in findings]

    def _pair_answer(self, ea, eb, elderly, pregnant, age, notice, intent):
        level, findings = self.engine.check_entities(ea, eb, elderly=elderly, pregnant=pregnant, age=age)
        gen = self._generated(ea.display, eb.display, level, findings)
        if gen:
            head = f"{LEVEL_EMOJI[level]} {gen}"
            body = ("\n\n[식약처 근거]\n" + _fmt_findings(findings)) if findings else ""
        else:
            head = random.choice(LEVEL_HEAD[level])
            body = f"\n\n{ea.display} + {eb.display}\n"
            if findings:
                body += "\n" + _fmt_findings(findings)
            elif level == "정보없음":
                body += "\n다만 모든 상호작용이 데이터에 등록되어 있는 것은 아니에요."
        return BotReply(reply=notice + head + body + DISCLAIMER, level=level, intent=intent,
                        drugs=[self._ent_info(ea), self._ent_info(eb)],
                        findings=self._findings_json(findings))

    def _multi_answer(self, entities, elderly, pregnant, age, notice, intent):
        level, findings = self.engine.check_multi(entities, elderly=elderly, pregnant=pregnant, age=age)
        names = ", ".join(e.display for e in entities)
        head = random.choice(LEVEL_HEAD[level])
        body = f"\n\n{names}\n(약 {len(entities)}개를 모든 조합으로 확인했어요)\n"
        if findings:
            body += "\n" + _fmt_findings(findings)
        elif level == "정보없음":
            body += "\n다만 모든 상호작용이 데이터에 등록되어 있는 것은 아니에요."
        return BotReply(reply=notice + head + body + DISCLAIMER, level=level, intent=intent,
                        drugs=[self._ent_info(e) for e in entities],
                        findings=self._findings_json(findings))

    def _single_answer(self, e, elderly, pregnant, age, notice, intent):
        level, findings = self.engine.check_entity_single(e, elderly=elderly, pregnant=pregnant, age=age)
        who = "임산부" if pregnant else ("고령자" if elderly else f"{age}세")
        gen = self._generated(e.display, None, level, findings) if findings else None
        if gen:
            head = f"{LEVEL_EMOJI[level]} {gen}\n\n[식약처 근거]\n" + _fmt_findings(findings)
        elif level == "정보없음":
            head = f"🟢 {e.display} — {who} 관련해 등록된 금기·주의 정보는 없었어요."
        else:
            head = random.choice(LEVEL_HEAD[level]) + f"\n\n{e.display} ({who} 기준)\n\n" + _fmt_findings(findings)
        return BotReply(reply=notice + head + DISCLAIMER, level=level, intent=intent,
                        drugs=[self._ent_info(e)], findings=self._findings_json(findings))

    def _ingredient_answer(self, e, notice):
        if not e.item_seq:
            # 순수 성분명 입력 (예: 와파린) — 그 자체가 성분
            return BotReply(reply=notice + f"'{e.display}'{pick_josa(e.display, '은')} 그 자체가 "
                            "성분(주성분)명이에요. 이 성분이 든 제품명을 알려주시면 함량까지 확인해 드릴게요.",
                            intent="성분", drugs=[self._ent_info(e)])
        ings = self.engine.ingredient_detail(e.item_seq)
        if not ings:
            return BotReply(reply=notice + f"'{e.display}'의 성분 정보를 찾지 못했어요.",
                            intent="성분", drugs=[self._ent_info(e)])
        lines = [f"• {nm}" + (f" {amt}" if amt else "") for nm, amt in ings]
        head = f"💊 {e.display}의 주성분 ({len(ings)}종)\n\n" + "\n".join(lines)
        return BotReply(reply=notice + head + DISCLAIMER, intent="성분",
                        drugs=[self._ent_info(e)])

    def _easy_info(self, m, intent, notice):
        row = self._resolve_seq(m)
        if not row:
            return BotReply(reply=notice + f"'{m.name}' 제품 정보를 찾지 못했어요.", intent=intent)
        col = "side_effect" if intent == "부작용" else "use_method"
        info = self.engine.conn.execute(
            f"SELECT item_name, {col} AS v FROM easy_drug_info WHERE item_seq = ?",
            (row["item_seq"],)).fetchone()
        label = "부작용" if intent == "부작용" else "복용법"
        if not info or not info["v"]:
            reply = (f"'{row['item_name']}'의 {label} 정보가 일반의약품 안내(e약은요) 데이터에 없어요. "
                     "전문의약품이거나 안내 미등록 품목일 수 있어요.")
        else:
            text = info["v"].strip()
            if len(text) > 400:
                text = text[:400] + "…"
            reply = f"💊 {info['item_name']} — {label} 안내 (식약처 e약은요)\n\n{text}"
        return BotReply(reply=notice + reply + DISCLAIMER, intent=intent, drugs=[self._drug_info(m)])
