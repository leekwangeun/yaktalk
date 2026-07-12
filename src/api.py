# FastAPI 백엔드: 로컬 데모(웹 폴더 포함 서빙)와 클라우드 배포(HF Spaces 등) 공용
# 실행: uvicorn api:app --app-dir src --port 8000
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

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


class ChatRequest(BaseModel):
    message: str
    elderly: bool = False
    pregnant: bool = False
    age: int | None = None


@app.post("/api/chat")
def chat(req: ChatRequest):
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
