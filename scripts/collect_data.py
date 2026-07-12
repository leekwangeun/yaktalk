# 식약처 공공데이터 수집 → SQLite 적재
# 실행: python scripts/collect_data.py
# 중단 후 재실행하면 meta 테이블 기준으로 이어서 수집한다.
import json
import sqlite3
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "drug.db"
SECRETS_PATH = ROOT / "secrets.json"

BASE_URL = "https://apis.data.go.kr/1471000"
ROWS_PER_PAGE = 100
WORKERS = 6
MAX_RETRIES = 4

# (테이블명, 서비스/오퍼레이션, 설명)
OPERATIONS = [
    ("permit_detail", "DrugPrdtPrmsnInfoService07/getDrugPrdtPrmsnDtlInq06", "의약품 허가 상세"),
    ("permit_ingredient", "DrugPrdtPrmsnInfoService07/getDrugPrdtMcpnDtlInq07", "의약품 주성분 상세"),
    ("dur_ingr_mix_taboo", "DURIrdntInfoService03/getUsjntTabooInfoList02", "DUR성분 병용금기"),
    ("dur_ingr_efcy_dup", "DURIrdntInfoService03/getEfcyDplctInfoList02", "DUR성분 효능군중복"),
    ("dur_ingr_elderly", "DURIrdntInfoService03/getOdsnAtentInfoList02", "DUR성분 노인주의"),
    ("dur_ingr_pregnancy", "DURIrdntInfoService03/getPwnmTabooInfoList02", "DUR성분 임부금기"),
    ("dur_ingr_age_taboo", "DURIrdntInfoService03/getSpcifyAgrdeTabooInfoList02", "DUR성분 특정연령대금기"),
    ("dur_ingr_capacity", "DURIrdntInfoService03/getCpctyAtentInfoList02", "DUR성분 용량주의"),
    ("dur_ingr_period", "DURIrdntInfoService03/getMdctnPdAtentInfoList02", "DUR성분 투여기간주의"),
    ("dur_item_info", "DURPrdlstInfoService03/getDurPrdlstInfoList03", "DUR품목 통합조회"),
    ("easy_drug", "DrbEasyDrugInfoService/getDrbEasyDrugList", "e약은요 개요정보"),
]
# DUR품목 병용금기(80만 건)는 성분 규칙 + M코드 조인으로 대체 가능하므로 수집하지 않는다.


def load_key() -> str:
    with open(SECRETS_PATH, encoding="utf-8") as f:
        return json.load(f)["data_go_kr_key"]


def fetch_page(key: str, op: str, page: int) -> dict:
    params = {
        "serviceKey": key,
        "type": "json",
        "numOfRows": str(ROWS_PER_PAGE),
        "pageNo": str(page),
    }
    url = f"{BASE_URL}/{op}?" + urllib.parse.urlencode(params)
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = json.loads(r.read().decode("utf-8"))
            code = data.get("header", {}).get("resultCode")
            if code != "00":
                raise RuntimeError(f"resultCode={code} msg={data.get('header', {}).get('resultMsg')}")
            return data["body"]
        except Exception as e:  # XML 오류 응답, 네트워크 오류 포함
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"{op} page={page} 실패: {last_err}")


def normalize_items(body: dict) -> list[dict]:
    items = body.get("items") or []
    if isinstance(items, dict):
        items = items.get("item") or []
    if isinstance(items, dict):
        items = [items]
    out = []
    for it in items:
        if isinstance(it, dict) and set(it.keys()) == {"item"}:
            it = it["item"]
        out.append(it)
    return out


def ensure_table(conn: sqlite3.Connection, table: str, keys: list[str]):
    cols = ", ".join(f'"{k}" TEXT' for k in keys)
    conn.execute(f'CREATE TABLE IF NOT EXISTS "{table}" ({cols})')
    existing = {row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')}
    for k in keys:
        if k not in existing:
            conn.execute(f'ALTER TABLE "{table}" ADD COLUMN "{k}" TEXT')


def insert_items(conn: sqlite3.Connection, table: str, items: list[dict]):
    if not items:
        return
    keys = sorted({k for it in items for k in it.keys()})
    ensure_table(conn, table, keys)
    placeholders = ", ".join("?" for _ in keys)
    collist = ", ".join(f'"{k}"' for k in keys)
    rows = [[None if it.get(k) is None else str(it.get(k)) for k in keys] for it in items]
    conn.executemany(f'INSERT INTO "{table}" ({collist}) VALUES ({placeholders})', rows)


def collect(conn: sqlite3.Connection, key: str, table: str, op: str, label: str):
    meta = conn.execute("SELECT next_page, total_count, done FROM meta WHERE tbl=?", (table,)).fetchone()
    if meta and meta[2]:
        print(f"[skip] {label}: 이미 완료 ({meta[1]:,}건)", flush=True)
        return
    page = meta[0] if meta else 1
    if not meta:
        conn.execute("INSERT INTO meta (tbl, next_page, total_count, done) VALUES (?, 1, 0, 0)", (table,))
        conn.commit()

    body = fetch_page(key, op, page)
    total = int(body.get("totalCount") or 0)
    total_pages = max(1, -(-total // ROWS_PER_PAGE))
    print(f"[start] {label}: {total:,}건 / {total_pages}페이지 (page {page}부터)", flush=True)

    insert_items(conn, table, normalize_items(body))
    conn.execute("UPDATE meta SET next_page=?, total_count=? WHERE tbl=?", (page + 1, total, table))
    conn.commit()
    page += 1

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        while page <= total_pages:
            chunk = list(range(page, min(page + WORKERS, total_pages + 1)))
            futures = [ex.submit(fetch_page, key, op, p) for p in chunk]
            for body in (f.result() for f in futures):
                insert_items(conn, table, normalize_items(body))
            conn.execute("UPDATE meta SET next_page=? WHERE tbl=?", (chunk[-1] + 1, table))
            conn.commit()
            if page // 50 != (chunk[-1] + 1) // 50 or chunk[-1] == total_pages:
                print(f"  {label}: {chunk[-1]}/{total_pages} 페이지", flush=True)
            page = chunk[-1] + 1

    n = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
    conn.execute("UPDATE meta SET done=1 WHERE tbl=?", (table,))
    conn.commit()
    print(f"[done] {label}: {n:,}행 적재", flush=True)


def main():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    key = load_key()
    conn = sqlite3.connect(DB_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS meta (tbl TEXT PRIMARY KEY, next_page INTEGER, total_count INTEGER, done INTEGER)")
    t0 = time.time()
    for table, op, label in OPERATIONS:
        collect(conn, key, table, op, label)
    print(f"전체 완료: {time.time() - t0:.0f}초", flush=True)
    conn.close()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
