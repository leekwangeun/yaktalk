# 원본 drug.db(2.3GB) → 챗봇용 경량 drug_light.db 생성
# 실행: python scripts/build_light_db.py  (재실행 시 기존 light DB를 새로 만든다)
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "drug.db"
DST = ROOT / "data" / "drug_light.db"

FORM_SUFFIX = (
    "필름코팅정|장용정|서방정|츄어블정|구강붕해정|발포정|연질캡슐|경질캡슐|캡슐|"
    "내복액|현탁액|건조시럽|시럽|과립|세립|산제|겔|크림|연고|주사액|주사|"
    "스프레이|패치|첩부제|점안액|점비액|좌제|환|정|액|산|고"
)
UNIT = "밀리그람|밀리그램|미리그람|그람|그램|밀리리터|리터|마이크로그램|mg|g|ml|mcg|㎍|iu|%"


def norm_name(name: str) -> str:
    """제품명에서 괄호·용량·제형 표기를 걷어낸 사전 매칭용 이름."""
    n = re.sub(r"\(.*?\)", "", name or "")
    n = re.sub(r"\[.*?\]", "", n)
    n = re.sub(rf"\d+(\.\d+)?\s*({UNIT})", "", n, flags=re.IGNORECASE)
    n = re.sub(rf"({FORM_SUFFIX})$", "", n)
    n = re.sub(r"\d+(\.\d+)?$", "", n)
    return n.strip()


M_CODE_RE = re.compile(r"\[(M\d+)\]([^/\[]*)")


def main():
    if DST.exists():
        DST.unlink()
    src = sqlite3.connect(SRC)
    dst = sqlite3.connect(DST)

    # 1. products (+ norm_name)
    dst.execute("""CREATE TABLE products (
        item_seq TEXT PRIMARY KEY, item_name TEXT, norm_name TEXT, entp_name TEXT,
        etc_otc TEXT, main_item_ingr TEXT, atc_code TEXT, cancel_name TEXT)""")
    rows = src.execute("""SELECT ITEM_SEQ, ITEM_NAME, ENTP_NAME, ETC_OTC_CODE,
        MAIN_ITEM_INGR, ATC_CODE, CANCEL_NAME FROM permit_detail""").fetchall()
    dst.executemany(
        "INSERT OR REPLACE INTO products VALUES (?,?,?,?,?,?,?,?)",
        [(r[0], r[1], norm_name(r[1]), r[2], r[3], r[4], r[5], r[6]) for r in rows],
    )
    print(f"products: {len(rows):,}행", flush=True)

    # 2. product_ingredients
    dst.execute("""CREATE TABLE product_ingredients (
        item_seq TEXT, mtral_code TEXT, mtral_nm TEXT, qnt TEXT, unit TEXT)""")
    rows = src.execute("""SELECT DISTINCT ITEM_SEQ, MTRAL_CODE, MTRAL_NM, QNT, INGD_UNIT_CD
        FROM permit_ingredient""").fetchall()
    dst.executemany("INSERT INTO product_ingredients VALUES (?,?,?,?,?)", rows)
    print(f"product_ingredients: {len(rows):,}행", flush=True)

    # 3. 병용금기 (DEL_YN 정상만)
    dst.execute("""CREATE TABLE dur_mix_taboo (
        ingr_code TEXT, ingr_name TEXT, ori TEXT, mix_type TEXT,
        mix_ingr_code TEXT, mix_ingr_name TEXT, mix_ori TEXT, mix_ingr_mix_type TEXT,
        prohibit_content TEXT, notification_date TEXT)""")
    rows = src.execute("""SELECT INGR_CODE, INGR_KOR_NAME, ORI, MIX_TYPE,
        MIXTURE_INGR_CODE, MIXTURE_INGR_KOR_NAME, MIXTURE_ORI, MIXTURE_MIX_TYPE,
        PROHBT_CONTENT, NOTIFICATION_DATE
        FROM dur_ingr_mix_taboo WHERE DEL_YN='정상'""").fetchall()
    dst.executemany("INSERT INTO dur_mix_taboo VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    print(f"dur_mix_taboo: {len(rows):,}행", flush=True)

    # 4~9. 단일 성분 DUR 규칙들 (공통 패턴 + 고유 컬럼)
    single_specs = [
        ("dur_efcy_dup", "dur_ingr_efcy_dup", "EFFECT_CODE, SERS_NAME", "effect_code TEXT, sers_name TEXT"),
        ("dur_elderly", "dur_ingr_elderly", "FORM_NAME", "form_name TEXT"),
        ("dur_pregnancy", "dur_ingr_pregnancy", "GRADE, FORM_NAME", "grade TEXT, form_name TEXT"),
        ("dur_age_taboo", "dur_ingr_age_taboo", "AGE_BASE, FORM_NAME", "age_base TEXT, form_name TEXT"),
        ("dur_capacity", "dur_ingr_capacity", "MAX_QTY, FORM_NAME", "max_qty TEXT, form_name TEXT"),
        ("dur_period", "dur_ingr_period", "MAX_DOSAGE_TERM, FORM_NAME", "max_term TEXT, form_name TEXT"),
    ]
    for dst_tbl, src_tbl, extra_src, extra_ddl in single_specs:
        dst.execute(f"""CREATE TABLE {dst_tbl} (
            ingr_code TEXT, ingr_name TEXT, ori TEXT, mix_type TEXT, {extra_ddl},
            prohibit_content TEXT, notification_date TEXT)""")
        rows = src.execute(f"""SELECT INGR_CODE, INGR_NAME, ORI_INGR, MIX_TYPE, {extra_src},
            PROHBT_CONTENT, NOTIFICATION_DATE FROM {src_tbl} WHERE DEL_YN='정상'""").fetchall()
        n_extra = extra_src.count(",") + 1
        ph = ",".join("?" * (6 + n_extra))
        dst.executemany(f"INSERT INTO {dst_tbl} VALUES ({ph})", rows)
        print(f"{dst_tbl}: {len(rows):,}행", flush=True)

    # 10. dur_item (원본 컬럼명 후행 공백 대응)
    src_cols = [r[1] for r in src.execute("PRAGMA table_info(dur_item_info)")]
    type_name_col = next(col for col in src_cols if col.strip() == "TYPE_NAME")
    dst.execute("""CREATE TABLE dur_item (
        item_seq TEXT, item_name TEXT, dur_type TEXT, material_name TEXT, etc_otc TEXT)""")
    rows = src.execute(f"""SELECT ITEM_SEQ, ITEM_NAME, TRIM("{type_name_col}"),
        MATERIAL_NAME, ETC_OTC_CODE FROM dur_item_info""").fetchall()
    dst.executemany("INSERT INTO dur_item VALUES (?,?,?,?,?)", rows)
    print(f"dur_item: {len(rows):,}행", flush=True)

    # 11. easy_drug_info
    dst.execute("""CREATE TABLE easy_drug_info (
        item_seq TEXT, item_name TEXT, efcy TEXT, use_method TEXT,
        atpn_warn TEXT, atpn TEXT, interaction TEXT, side_effect TEXT)""")
    rows = src.execute("""SELECT itemSeq, itemName, efcyQesitm, useMethodQesitm,
        atpnWarnQesitm, atpnQesitm, intrcQesitm, seQesitm FROM easy_drug""").fetchall()
    dst.executemany("INSERT INTO easy_drug_info VALUES (?,?,?,?,?,?,?,?)", rows)
    print(f"easy_drug_info: {len(rows):,}행", flush=True)

    # 12. mcode_dur_map — DUR 규칙의 ORI 필드에서 [M코드]표기명을 파싱해 M코드↔D코드 다리 생성
    pairs = set()

    def harvest(ori_text, dcode):
        if not ori_text or not dcode:
            return
        for m, nm in M_CODE_RE.findall(ori_text):
            pairs.add((m, nm.strip(), dcode))

    # 복합 규칙의 ORI는 조합된 성분 전체를 나열하므로 매핑 수확은 '단일' 규칙에서만.
    for ori, code in src.execute("SELECT ORI, INGR_CODE FROM dur_ingr_mix_taboo WHERE DEL_YN='정상' AND MIX_TYPE='단일'"):
        harvest(ori, code)
    for ori, code in src.execute("SELECT MIXTURE_ORI, MIXTURE_INGR_CODE FROM dur_ingr_mix_taboo WHERE DEL_YN='정상' AND MIXTURE_MIX_TYPE='단일'"):
        harvest(ori, code)
    for _, src_tbl, _, _ in single_specs:
        for ori, code in src.execute(f"SELECT ORI_INGR, INGR_CODE FROM {src_tbl} WHERE DEL_YN='정상' AND MIX_TYPE='단일'"):
            harvest(ori, code)

    dst.execute("CREATE TABLE mcode_dur_map (mtral_code TEXT, mtral_nm TEXT, dur_ingr_code TEXT)")
    dst.executemany("INSERT INTO mcode_dur_map VALUES (?,?,?)", sorted(pairs))
    print(f"mcode_dur_map: {len(pairs):,}행 (M코드↔DUR성분코드 매핑)", flush=True)

    # 인덱스
    for stmt in [
        "CREATE INDEX idx_pi_item ON product_ingredients(item_seq)",
        "CREATE INDEX idx_pi_mcode ON product_ingredients(mtral_code)",
        "CREATE INDEX idx_prod_norm ON products(norm_name)",
        "CREATE INDEX idx_map_mcode ON mcode_dur_map(mtral_code)",
        "CREATE INDEX idx_map_dcode ON mcode_dur_map(dur_ingr_code)",
        "CREATE INDEX idx_taboo_a ON dur_mix_taboo(ingr_code)",
        "CREATE INDEX idx_taboo_b ON dur_mix_taboo(mix_ingr_code)",
        "CREATE INDEX idx_duritem ON dur_item(item_seq)",
        "CREATE INDEX idx_easy ON easy_drug_info(item_seq)",
    ]:
        dst.execute(stmt)

    dst.commit()
    dst.execute("VACUUM")
    dst.close()
    src.close()
    print(f"완료: {DST} ({DST.stat().st_size / 1e6:.1f}MB)", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
