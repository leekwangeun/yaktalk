# FastAPI 백엔드: 로컬 데모(웹 폴더 포함 서빙)와 클라우드 배포(HF Spaces 등) 공용
# 실행: uvicorn api:app --app-dir src --port 8000
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from dictionary import DrugDictionary
from generator import ResponseGenerator
from matcher import DrugMatcher
from nlu import DrugNER, IntentClassifier
from responder import Responder
from risk_engine import RiskEngine

ROOT = Path(__file__).resolve().parent.parent

app = FastAPI(title="약물 상호작용 챗봇 API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

print("엔진 로딩 중...")
# DISABLE_NLU=1 → 학습 모델 없이 사전+규칙만으로 동작 (저메모리 호스팅 안전판)
_lite = os.environ.get("DISABLE_NLU") == "1"
_none_dir = Path("__disabled__")
_matcher = DrugMatcher(DrugDictionary(), ner=None if _lite else DrugNER())
_engine = RiskEngine()
_generator = ResponseGenerator(_none_dir) if _lite else ResponseGenerator()
_responder = Responder(_matcher, _engine,
                       IntentClassifier(_none_dir) if _lite else IntentClassifier(),
                       generator=_generator)
print(f"엔진 준비 완료 (NLU 모델: {'꺼짐' if _lite else '켜짐'}, "
      f"KoGPT2 생성기: {'켜짐' if _generator.available else '꺼짐(템플릿)'})")


# 사람이 실제로 던지는 질문 길이의 상한. 이보다 길면 유사도 매칭이 어절 수에
# 비례해 느려져 응답이 지연된다(장문 입력으로 수 분간 멈추는 것을 확인).
MAX_MESSAGE_LEN = 300


class ChatRequest(BaseModel):
    message: str
    elderly: bool = False
    pregnant: bool = False
    # 나이는 0~120만 허용. 범위를 열어두면 음수가 '12세 미만 금기'로 판정되는 등
    # 잘못된 입력이 의학적 판정으로 이어진다.
    age: int | None = Field(default=None, ge=0, le=120)


@app.post("/api/chat")
def chat(req: ChatRequest):
    if len(req.message) > MAX_MESSAGE_LEN:
        return {"reply": f"질문이 너무 길어요. 약 이름 위주로 {MAX_MESSAGE_LEN}자 이내로 물어봐 주세요.",
                "level": None, "intent": "기타", "drugs": [], "clarify": None, "findings": []}
    r = _responder.handle(req.message, elderly=req.elderly, pregnant=req.pregnant, age=req.age)
    return {"reply": r.reply, "level": r.level, "intent": r.intent,
            "drugs": r.drugs, "clarify": r.clarify, "findings": r.findings}


@app.get("/api/health")
def health():
    return {"status": "ok", "ner": _matcher.ner.available if _matcher.ner else False}


# 로컬 데모: http://localhost:8000 에서 채팅 UI까지 바로 서빙
web_dir = ROOT / "web"
if web_dir.exists():
    app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")
