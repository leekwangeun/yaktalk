"""web/index.html → netlify/index.html (Netlify Drop용 단일 파일) 생성.

Netlify Drop은 파일 하나만 올리므로 config.js를 못 읽는다.
백엔드 주소를 인라인으로 박아 넣어 단독으로 동작하는 index.html을 만든다.

사용법: python scripts/build_netlify.py
"""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "web" / "index.html"
CFG = ROOT / "web" / "config.js"
DST = ROOT / "netlify" / "index.html"

CONFIG_TAG = '<script src="config.js"></script>'


def read_api_base() -> str:
    """config.js에서 API_BASE를 읽는다.

    주석 줄에 예시 URL(https://사용자ID-...)이 들어 있어, 주석을 걸러내지 않으면
    그 예시가 배포본에 박히는 사고가 난다. 반드시 대입문만 본다.
    """
    api_base = None
    for line in CFG.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("//"):
            continue
        m = re.search(r'window\.API_BASE\s*=\s*"([^"]*)"', line)
        if m:
            api_base = m.group(1)
    if not api_base:
        raise SystemExit(
            "config.js에서 API_BASE 대입문을 찾지 못했거나 값이 비어 있습니다.\n"
            "Netlify 배포본은 백엔드 절대주소가 필요합니다 (예: https://<ID>-yaktalk-api.hf.space)."
        )
    if "사용자ID" in api_base or not api_base.startswith("https://"):
        raise SystemExit(f"API_BASE가 실제 배포 주소로 보이지 않습니다: {api_base}")
    return api_base


def main() -> None:
    html = SRC.read_text(encoding="utf-8")
    if CONFIG_TAG not in html:
        raise SystemExit(f"web/index.html에서 {CONFIG_TAG} 를 찾지 못했습니다.")

    api_base = read_api_base()
    inline = (
        "<script>\n"
        "// Netlify 정적 배포용: web/config.js 내용을 인라인으로 고정\n"
        f'window.API_BASE = "{api_base}";\n'
        "</script>"
    )
    out = html.replace(CONFIG_TAG, inline)

    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text(out, encoding="utf-8")
    print(f"생성 완료: {DST}")
    print(f"  API_BASE = {api_base}")
    print(f"  크기     = {len(out):,} bytes")
    print("\nhttps://app.netlify.com/drop 에 이 파일을 끌어다 놓으면 배포됩니다.")


if __name__ == "__main__":
    main()
