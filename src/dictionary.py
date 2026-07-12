# 제품명 사전: drug_light.db → Aho-Corasick 오토마타 + 자모 사전
import re
import sqlite3
from pathlib import Path

import ahocorasick
import hgtk

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "drug_light.db"

# 카테고리어 → 대표 제품(norm_name) 후보. 되묻기 트리거용 시작 세트.
CATEGORIES = {
    "감기약": ["판콜에스", "판피린큐", "타이레놀콜드에스", "테라플루나이트타임", "화이투벤"],
    "두통약": ["타이레놀", "게보린", "펜잘큐", "그날엔"],
    "진통제": ["타이레놀", "부루펜", "이지엔6", "탁센"],
    "해열제": ["타이레놀", "부루펜", "챔프시럽"],
    "소염제": ["부루펜", "탁센", "이지엔6"],
    "알레르기약": ["지르텍", "클라리틴", "알레그라"],
    "소화제": ["베아제", "훼스탈플러스"],
}

# 어절 끝에 붙는 흔한 조사 (긴 것부터 제거)
_JOSA = ["이랑", "하고", "에다", "까지", "부터", "이나", "랑", "과", "와", "을", "를", "은", "는", "이", "가", "도", "만", "요"]

_JOSA_TABLE = {"이랑": ("이랑", "랑"), "은": ("은", "는"), "이": ("이", "가"),
               "을": ("을", "를"), "과": ("과", "와")}


def pick_josa(word: str, spec: str) -> str:
    """받침 유무에 따라 조사 선택. 예: pick_josa('타이레놀', '은') → '은'"""
    with_b, without_b = _JOSA_TABLE[spec]
    last = word[-1]
    if not hgtk.checker.is_hangul(last):
        return with_b
    return with_b if hgtk.checker.has_batchim(last) else without_b


def jamo(text: str) -> str:
    return hgtk.text.decompose(text, compose_code="")


# 성분명 염(salt)·수화물 접미사 — 사용자는 "세티리진염산염"을 "세티리진"이라 부른다
_SALT_SUFFIXES = ("브롬화수소산염", "타르타르산염", "시트르산염", "아세트산염", "말레산염",
                  "베실산염", "메실산염", "토실산염", "푸마르산염", "숙신산염", "인산염",
                  "염산염", "황산염", "질산염", "이수화물", "반수화물", "수화물", "무수물",
                  "나트륨", "칼륨", "칼슘", "마그네슘")


def strip_salt(name: str) -> str:
    changed = True
    while changed:
        changed = False
        for s in _SALT_SUFFIXES:
            if name.endswith(s) and len(name) - len(s) >= 3:
                name = name[: -len(s)]
                changed = True
                break
    return name


class DrugDictionary:
    def __init__(self, db_path: Path = DB_PATH):
        conn = sqlite3.connect(db_path)
        self.name_to_seqs: dict[str, set[str]] = {}
        rows = conn.execute("""
            SELECT norm_name, item_name, item_seq FROM products
            WHERE cancel_name IS NULL OR cancel_name = '정상'
        """).fetchall()
        conn.close()
        for norm, full, seq in rows:
            for name in {norm, full}:
                if name and len(name) >= 2:
                    self.name_to_seqs.setdefault(name, set()).add(seq)

        # 성분명 사전: 사용자가 "와파린", "아스피린"처럼 성분명으로 부르는 경우 대응.
        # 해당 성분의 단일제(성분 1개짜리 제품)로 해석해 순수한 성분 검사가 되게 한다.
        conn2 = sqlite3.connect(db_path)
        ingr_rows = conn2.execute("""
            SELECT pi.mtral_nm, pi.item_seq FROM product_ingredients pi
            JOIN (SELECT item_seq FROM product_ingredients
                  GROUP BY item_seq HAVING COUNT(DISTINCT mtral_code) = 1) s
              ON s.item_seq = pi.item_seq
            JOIN products p ON p.item_seq = pi.item_seq
            WHERE (p.cancel_name IS NULL OR p.cancel_name = '정상')
        """).fetchall()
        conn2.close()
        self.ingredient_names: set[str] = set()
        for nm, seq in ingr_rows:
            if nm and 2 <= len(nm) <= 12 and re.fullmatch(r"[가-힣a-zA-Z0-9]+", nm):
                for name in {nm, strip_salt(nm)}:
                    if len(name) >= 2:
                        self.ingredient_names.add(name)
                        self.name_to_seqs.setdefault(name, set()).add(seq)

        # DUR 규칙 성분명 → D코드 사전. 제품에 없는 성분(예: 플루복사민)도 인식해
        # D코드로 병용금기 등을 직접 판정할 수 있게 한다.
        conn3 = sqlite3.connect(db_path)
        self.name_to_dcode: dict[str, str] = {}
        dur_srcs = [("dur_mix_taboo", "ingr_name", "ingr_code"),
                    ("dur_mix_taboo", "mix_ingr_name", "mix_ingr_code"),
                    ("dur_efcy_dup", "ingr_name", "ingr_code"),
                    ("dur_elderly", "ingr_name", "ingr_code"),
                    ("dur_pregnancy", "ingr_name", "ingr_code"),
                    ("dur_age_taboo", "ingr_name", "ingr_code")]
        for tbl, ncol, ccol in dur_srcs:
            for nm, code in conn3.execute(
                    f"SELECT DISTINCT {ncol}, {ccol} FROM {tbl} WHERE {ncol} IS NOT NULL AND {ccol} IS NOT NULL"):
                for name in {nm.strip(), strip_salt(nm.strip())}:
                    if 2 <= len(name) <= 15 and re.fullmatch(r"[가-힣a-zA-Z0-9]+", name):
                        self.name_to_dcode.setdefault(name, code)
        conn3.close()
        self.ingredient_names |= set(self.name_to_dcode)

        self.automaton = ahocorasick.Automaton()
        for name in self.name_to_seqs:
            self.automaton.add_word(name, name)
        for name in self.name_to_dcode:
            if name not in self.name_to_seqs:
                self.automaton.add_word(name, name)
        for cat in CATEGORIES:
            self.automaton.add_word(cat, cat)
        self.automaton.make_automaton()

        # 자모 사전은 norm_name(짧은 이름)만 대상 — 오타 보정용
        self.jamo_index = {name: jamo(name) for name in self.name_to_seqs if len(name) <= 12}

    def scan(self, text: str) -> list[tuple[int, int, str]]:
        """문장에서 사전 등재 이름을 찾아 (start, end, name) 목록으로. 겹치면 최장일치.
        띄어쓰기 없는 입력을 위해, 앞 매칭 끝~현재 매칭 사이가 조사/접속사면 어절
        중간 매칭도 허용한다 (예: '타이레놀이랑게보린'의 게보린). 반대로 앞이 일반
        어간이면 제외한다 (예: '마이프로틴'의 프로틴)."""
        raw = []
        for end, name in self.automaton.iter(text):
            start = end - len(name) + 1
            raw.append((start, end + 1, name))
        raw.sort(key=lambda h: (h[0], -(h[1] - h[0])))
        chosen, cursor = [], 0
        for start, end, name in raw:
            if start < cursor:
                continue
            if start > 0 and re.match(r"[가-힣A-Za-z0-9]", text[start - 1]):
                gap = text[cursor:start]  # 직전 채택 매칭 끝 ~ 이번 매칭 시작
                if not self._is_connector(gap):
                    continue
            chosen.append((start, end, name))
            cursor = end
        return chosen

    @staticmethod
    def _is_connector(gap: str) -> bool:
        """어절 사이 텍스트가 조사/접속 표현으로만 이뤄졌는지 (붙여쓰기 분해용)."""
        if gap == "":
            return True
        connectors = ["이랑", "랑", "하고", "이나", "나", "과", "와", "이며", "며",
                      "이고", "고", "에다", "에", "그리고", "또", "또는", "및", ",", " "]
        s = gap
        while s:
            for c in connectors:
                if s.startswith(c):
                    s = s[len(c):]
                    break
            else:
                return False
        return True

    @staticmethod
    def strip_josa(token: str) -> str:
        for j in _JOSA:
            if token.endswith(j) and len(token) - len(j) >= 2:
                return token[: -len(j)]
        return token
