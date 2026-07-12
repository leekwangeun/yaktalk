# 배포 가이드 — Vercel(프론트) + Hugging Face Spaces(백엔드)

## 구조
```
[사용자 브라우저] → Vercel (web/ 정적 채팅 UI)
                  → HF Spaces (FastAPI: 매칭·판정·모델 추론, Docker)
```
Vercel 서버리스는 용량 제한(250MB) 때문에 PyTorch 백엔드를 직접 못 돌린다 → 분리 배포.

## 1. 백엔드 — Hugging Face Spaces (무료 CPU, 16GB RAM)

1. huggingface.co 가입 → New Space → SDK: **Docker**, 이름 예: `yaktalk-api`
2. Space 저장소를 clone 후 아래 파일 복사:
   - `Dockerfile`, `requirements.txt`, `src/`, `web/`, `data/drug_light.db`, `models/ner-drug/`, `models/intent-drug/`
   - **주의**: `secrets.json`, `public data api/`, 원본 `drug.db`(2.3GB), 베이스 모델 2종은 올리지 말 것
3. Space의 README.md 상단에 메타데이터 추가:
   ```yaml
   ---
   title: yaktalk-api
   sdk: docker
   app_port: 7860
   ---
   ```
4. 대용량 파일은 git-lfs로: `git lfs track "*.db" "*.safetensors"` 후 add/commit/push
5. 빌드 완료 후 `https://<ID>-yaktalk-api.hf.space/api/health` 로 확인

## 2. 프론트엔드 — Vercel

1. `web/config.js`의 `window.API_BASE`를 Space 주소로 수정:
   `window.API_BASE = "https://<ID>-yaktalk-api.hf.space";`
2. vercel.com 가입 → New Project → `web/` 폴더 배포
   (GitHub 저장소 연결 시 Root Directory를 `web`으로 지정, Framework: Other)
3. 배포 URL 접속 → 채팅 테스트

## 3. 발표 데모 (오프라인 안전판)

인터넷 불안 시 로컬 단독 실행 — 백엔드가 web/까지 같이 서빙한다:
```
uvicorn api:app --app-dir src --port 8000
# 브라우저에서 http://localhost:8000
```
`web/config.js`의 API_BASE가 빈 문자열이면 자동으로 로컬 서버를 쓴다.

## 참고
- HF Spaces 무료 티어는 48시간 미사용 시 슬립 → 첫 요청이 느릴 수 있음. 발표 전 미리 한 번 호출해 깨워둘 것.
- CORS는 api.py에서 전체 허용(`*`)으로 설정돼 있어 Vercel 도메인에서 바로 호출 가능.
