# 3단계: 위험도 검사 엔진 — 제품 조합 → 등급(금기/주의/정보없음) + 근거 목록
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "drug_light.db"

M_CODE_RE = re.compile(r"\[(M\d+)\]")

LEVEL_ORDER = {"금기": 2, "주의": 1, "정보없음": 0}


def _dedupe(findings: list) -> list:
    """같은 규칙이 제형별로 여러 행일 때 근거가 중복 출력되지 않게 제거."""
    seen, out = set(), []
    for f in findings:
        key = (f.check, f.detail)
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out


@dataclass
class Finding:
    check: str      # 병용금기 | 성분중복 | 효능군중복 | 노인주의 | 임부금기
    level: str      # 금기 | 주의
    detail: str     # 근거 설명 (식약처 원문 인용)


@dataclass
class Entity:
    """검사 대상 하나 — 제품(item_seq 보유) 또는 순수 성분(D코드만)."""
    display: str                 # 화면 표시 이름
    item_seq: str | None         # 제품이면 품목기준코드
    mcodes: set                  # 성분 M코드 집합
    dcodes: set                  # DUR 성분 D코드 집합
    ing_by_m: dict               # mcode → 성분명
    ing_by_d: dict               # dcode → 성분명 (순수 성분 표시용)


class RiskEngine:
    def __init__(self, db_path: Path = DB_PATH):
        # 읽기 전용 사용 — FastAPI 워커 스레드에서 접근 가능하도록 스레드 검사 해제
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

    # ---------- 데이터 접근 ----------
    def resolve_products(self, norm_name: str) -> list[sqlite3.Row]:
        rows = self.conn.execute("""
            SELECT item_seq, item_name FROM products
            WHERE norm_name = ? AND (cancel_name IS NULL OR cancel_name = '정상')
        """, (norm_name,)).fetchall()
        if rows:
            return rows
        # 성분명으로 들어온 경우: 그 성분이 든 제품 중 성분 수가 적은 것(단일제) 우선
        return self.conn.execute("""
            SELECT p.item_seq, p.item_name FROM products p
            JOIN product_ingredients pi ON pi.item_seq = p.item_seq
            WHERE pi.mtral_nm = ? AND (p.cancel_name IS NULL OR p.cancel_name = '정상')
            ORDER BY (SELECT COUNT(DISTINCT mtral_code) FROM product_ingredients x
                      WHERE x.item_seq = p.item_seq), LENGTH(p.item_name)
            LIMIT 3
        """, (norm_name,)).fetchall()

    def products_by_seqs(self, seqs: set[str]) -> list[sqlite3.Row]:
        """사전이 이미 알고 있는 item_seq 집합 → 제품 행 (단일제 우선)."""
        if not seqs:
            return []
        ph = ",".join("?" * len(seqs))
        return self.conn.execute(f"""
            SELECT item_seq, item_name FROM products
            WHERE item_seq IN ({ph})
            ORDER BY (SELECT COUNT(DISTINCT mtral_code) FROM product_ingredients x
                      WHERE x.item_seq = products.item_seq), LENGTH(item_name)
            LIMIT 3
        """, list(seqs)).fetchall()

    def ingredients(self, item_seq: str) -> dict[str, str]:
        rows = self.conn.execute("""
            SELECT DISTINCT mtral_code, mtral_nm FROM product_ingredients WHERE item_seq = ?
        """, (item_seq,)).fetchall()
        return {r["mtral_code"]: r["mtral_nm"] for r in rows}

    def ingredient_detail(self, item_seq: str) -> list[tuple[str, str]]:
        """(성분명, '분량+단위') 목록 — 성분 질문 응답용. 중복 성분명 제거."""
        rows = self.conn.execute(
            "SELECT mtral_nm, qnt, unit FROM product_ingredients WHERE item_seq = ?",
            (item_seq,)).fetchall()
        out, seen = [], set()
        for r in rows:
            nm = r["mtral_nm"]
            if not nm or nm in seen:
                continue
            seen.add(nm)
            amt = f"{r['qnt']}{r['unit'] or ''}" if r["qnt"] else ""
            out.append((nm, amt))
        return out

    def ingredient_amounts(self, item_seq: str) -> dict[str, str]:
        """mcode → '분량+단위' (예: '500밀리그람'). 성분별 함량 표시용."""
        rows = self.conn.execute("""
            SELECT mtral_code, qnt, unit FROM product_ingredients WHERE item_seq = ?
        """, (item_seq,)).fetchall()
        out = {}
        for r in rows:
            if r["mtral_code"] and r["qnt"]:
                out[r["mtral_code"]] = f"{r['qnt']}{r['unit'] or ''}"
        return out

    def make_entity(self, item_seq: str, display: str) -> Entity:
        ing = self.ingredients(item_seq)
        m = set(ing)
        d = self.dur_codes(m)
        return Entity(display, item_seq, m, d, ing, {})

    def entity_from_dcode(self, name: str, dcode: str) -> Entity:
        """제품에 없는 순수 DUR 성분 → D코드만 가진 엔티티 (예: 플루복사민)."""
        return Entity(name, None, set(), {dcode}, {}, {dcode: name})

    def dur_codes(self, mcodes: set[str]) -> set[str]:
        if not mcodes:
            return set()
        ph = ",".join("?" * len(mcodes))
        rows = self.conn.execute(
            f"SELECT DISTINCT dur_ingr_code FROM mcode_dur_map WHERE mtral_code IN ({ph})",
            list(mcodes)).fetchall()
        return {r["dur_ingr_code"] for r in rows}

    # ---------- 규칙 측면 매칭 (단일/복합 공통) ----------
    @staticmethod
    def _side_hits(mix_type, ori, dcode, prod_mcodes, prod_dcodes) -> bool:
        if mix_type == "복합":
            rule_m = set(M_CODE_RE.findall(ori or ""))
            return bool(rule_m) and rule_m <= prod_mcodes
        return dcode in prod_dcodes

    # ---------- 검사들 ----------
    def check_pair(self, seq_a: str, seq_b: str, elderly: bool = False,
                   pregnant: bool = False, age: float | None = None) -> tuple[str, list[Finding]]:
        return self.check_entities(self.make_entity(seq_a, seq_a),
                                   self.make_entity(seq_b, seq_b),
                                   elderly=elderly, pregnant=pregnant, age=age)

    def check_entities(self, ea: Entity, eb: Entity, elderly: bool = False,
                       pregnant: bool = False, age: float | None = None) -> tuple[str, list[Finding]]:
        m_a, m_b, d_a, d_b = ea.mcodes, eb.mcodes, ea.dcodes, eb.dcodes
        findings: list[Finding] = []
        findings += self._mix_taboo(m_a, d_a, m_b, d_b)
        findings += self._dup_ingredient(ea, eb)
        findings += self._efcy_dup(m_a, d_a, m_b, d_b)
        if elderly:
            findings += self._single_rule("dur_elderly", "노인주의", "주의", m_a | m_b, d_a | d_b)
        if pregnant:
            findings += self._pregnancy(m_a | m_b, d_a | d_b)
        if age is not None:
            findings += self._age_taboo(m_a | m_b, d_a | d_b, age)
        return self._grade(_dedupe(findings))

    def check_multi(self, entities: list[Entity], elderly: bool = False,
                    pregnant: bool = False, age: float | None = None):
        """3개 이상: 모든 쌍을 전수 검사 + 각 약의 개별(노인/임부/연령) 검사."""
        findings: list[Finding] = []
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                _, fs = self.check_entities(entities[i], entities[j])
                findings += fs
        merged = set()  # 개별 검사는 성분 기준 1회만
        m_all = set().union(*[e.mcodes for e in entities]) if entities else set()
        d_all = set().union(*[e.dcodes for e in entities]) if entities else set()
        if elderly:
            findings += self._single_rule("dur_elderly", "노인주의", "주의", m_all, d_all)
        if pregnant:
            findings += self._pregnancy(m_all, d_all)
        if age is not None:
            findings += self._age_taboo(m_all, d_all, age)
        return self._grade(_dedupe(findings))

    def check_single(self, seq: str, elderly: bool = False,
                     pregnant: bool = False, age: float | None = None):
        return self.check_entity_single(self.make_entity(seq, seq),
                                        elderly=elderly, pregnant=pregnant, age=age)

    def check_entity_single(self, e: Entity, elderly: bool = False,
                            pregnant: bool = False, age: float | None = None):
        m, d = e.mcodes, e.dcodes
        findings: list[Finding] = []
        if elderly:
            findings += self._single_rule("dur_elderly", "노인주의", "주의", m, d)
        if pregnant:
            findings += self._pregnancy(m, d)
        if age is not None:
            findings += self._age_taboo(m, d, age)
        return self._grade(_dedupe(findings))

    @staticmethod
    def _grade(findings: list[Finding]) -> tuple[str, list[Finding]]:
        level = "정보없음"
        for f in findings:
            if LEVEL_ORDER[f.level] > LEVEL_ORDER[level]:
                level = f.level
        return level, findings

    def _mix_taboo(self, m_a, d_a, m_b, d_b) -> list[Finding]:
        out = []
        for r in self.conn.execute("SELECT * FROM dur_mix_taboo"):
            left = (r["mix_type"], r["ori"], r["ingr_code"])
            right = (r["mix_ingr_mix_type"], r["mix_ori"], r["mix_ingr_code"])
            ab = self._side_hits(*left, m_a, d_a) and self._side_hits(*right, m_b, d_b)
            ba = self._side_hits(*left, m_b, d_b) and self._side_hits(*right, m_a, d_a)
            if ab or ba:
                reason = (r["prohibit_content"] or "").strip() or "병용금기 고시 대상"
                out.append(Finding("병용금기", "금기",
                    f"{r['ingr_name']} + {r['mix_ingr_name']}: {reason}"))
        return out

    def _dup_ingredient(self, ea: "Entity", eb: "Entity") -> list[Finding]:
        out = []
        amt_a = self.ingredient_amounts(ea.item_seq) if ea.item_seq else {}
        amt_b = self.ingredient_amounts(eb.item_seq) if eb.item_seq else {}
        # 1) M코드 교집합 (제품-제품, 정밀)
        for m in ea.mcodes & eb.mcodes:
            name = ea.ing_by_m.get(m) or eb.ing_by_m.get(m) or "성분"
            detail = f"두 약 모두 '{name}' 성분을 포함 — 같은 성분을 이중 복용하면 과량 위험"
            parts = []
            if m in amt_a:
                parts.append(f"{ea.display} {amt_a[m]}")
            if m in amt_b:
                parts.append(f"{eb.display} {amt_b[m]}")
            if parts:
                detail += " (" + " + ".join(parts) + ")"
            max_qty = self._capacity_limit(m)
            if max_qty:
                detail += f" — 1일 최대용량 {max_qty}(성인 기준, 고령자는 더 낮음)"
            out.append(Finding("성분중복", "주의", detail))
        # 2) D코드 교집합 중 M코드로 안 잡힌 것 (순수 성분 입력 대응)
        m_dcodes = self.dur_codes(ea.mcodes & eb.mcodes)
        for dc in ea.dcodes & eb.dcodes:
            if dc in m_dcodes:
                continue
            name = ea.ing_by_d.get(dc) or eb.ing_by_d.get(dc) or "성분"
            out.append(Finding("성분중복", "주의",
                f"두 약 모두 '{name}' 성분을 포함 — 같은 성분을 이중 복용하면 과량 위험"))
        return out

    def _capacity_limit(self, mcode: str) -> str | None:
        """성분중복 근거 보강용: 해당 성분의 DUR 용량주의 한도를 찾는다."""
        row = self.conn.execute("""
            SELECT c.max_qty FROM dur_capacity c
            JOIN mcode_dur_map m ON m.dur_ingr_code = c.ingr_code
            WHERE m.mtral_code = ? AND c.mix_type = '단일'
            ORDER BY LENGTH(c.max_qty) DESC LIMIT 1
        """, (mcode,)).fetchone()
        return row["max_qty"] if row else None

    _AGE_RE = re.compile(r"(\d+)\s*(세|개월)\s*(이하|미만)")

    def _age_taboo(self, mcodes, dcodes, age_years: float) -> list[Finding]:
        out = []
        for r in self.conn.execute("SELECT * FROM dur_age_taboo"):
            if not self._side_hits(r["mix_type"], r["ori"], r["ingr_code"], mcodes, dcodes):
                continue
            m = self._AGE_RE.search(r["age_base"] or "")
            if not m:
                continue
            bound = int(m.group(1)) / (12 if m.group(2) == "개월" else 1)
            hit = age_years < bound if m.group(3) == "미만" else age_years <= bound
            if hit:
                reason = (r["prohibit_content"] or "").strip()
                out.append(Finding("연령금기", "금기",
                    f"{r['ingr_name']} ({r['age_base']} 금기): {reason}"))
        return out

    def _efcy_dup(self, m_a, d_a, m_b, d_b) -> list[Finding]:
        rows = self.conn.execute("SELECT * FROM dur_efcy_dup").fetchall()
        hits_a, hits_b = {}, {}
        for r in rows:
            side = (r["mix_type"], r["ori"], r["ingr_code"])
            if self._side_hits(*side, m_a, d_a):
                hits_a.setdefault(r["effect_code"], r)
            if self._side_hits(*side, m_b, d_b):
                hits_b.setdefault(r["effect_code"], r)
        out = []
        for code in set(hits_a) & set(hits_b):
            ra, rb = hits_a[code], hits_b[code]
            out.append(Finding("효능군중복", "주의",
                f"'{ra['ingr_name']}'({ra['sers_name']})와 '{rb['ingr_name']}'({rb['sers_name']}) — "
                f"같은 효능군({code}) 중복"))
        return out

    def _single_rule(self, table: str, check: str, level: str, mcodes, dcodes) -> list[Finding]:
        out = []
        for r in self.conn.execute(f"SELECT * FROM {table}"):
            if self._side_hits(r["mix_type"], r["ori"], r["ingr_code"], mcodes, dcodes):
                reason = (r["prohibit_content"] or "").strip()
                out.append(Finding(check, level, f"{r['ingr_name']}: {reason}"))
        return out

    def _pregnancy(self, mcodes, dcodes) -> list[Finding]:
        out = []
        for r in self.conn.execute("SELECT * FROM dur_pregnancy"):
            if self._side_hits(r["mix_type"], r["ori"], r["ingr_code"], mcodes, dcodes):
                level = "금기" if r["grade"] == "1등급" else "주의"
                reason = (r["prohibit_content"] or "").strip()
                out.append(Finding("임부금기", level,
                    f"{r['ingr_name']} (임부금기 {r['grade']}): {reason}"))
        return out
