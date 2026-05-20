import os
import uuid
import time
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

if not os.environ.get("GROQ_API_KEY"):
    print("⚠️  WARNING: GROQ_API_KEY is not set!")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DevOps AI Copilot",
    description="Multi-turn chat with full history sent to Groq every time",
    version="1.0.0",
)

# ── Groq client ───────────────────────────────────────────────────────────────
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

MODEL         = "llama-3.1-8b-instant"
MAX_TOKENS    = 1024
TEMPERATURE   = 0.7

SYSTEM_PROMPT = """You are a senior DevOps engineer and infrastructure expert.
Help with Kubernetes, Terraform, CI/CD pipelines, Docker, ArgoCD, Helm, and cloud platforms.
Be concise and practical. Always include relevant commands or config snippets.
If you are unsure about something, say so clearly."""

# ── In-memory session store ───────────────────────────────────────────────────
sessions: dict[str, dict] = {}


# ── Request / Response Models ─────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    stream: bool = False


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    message_count: int
    total_tokens: int
    latency_ms: float


class HistoryResponse(BaseModel):
    session_id: str
    messages: list[dict]
    message_count: int
    created_at: str
    updated_at: str


class SessionSummary(BaseModel):
    session_id: str
    message_count: int
    total_tokens: int
    created_at: str
    updated_at: str


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_or_create_session(session_id: Optional[str]) -> str:
    if session_id and session_id in sessions:
        return session_id

    new_id = session_id or str(uuid.uuid4())
    sessions[new_id] = {
        "history":       [],
        "created_at":    datetime.utcnow().isoformat(),
        "updated_at":    datetime.utcnow().isoformat(),
        "message_count": 0,
        "total_tokens":  0,
    }
    return new_id


def build_messages(history: list[dict]) -> list[dict]:
    """ 
    IMPORTANT: Full history is sent on every call 
    System prompt + entire conversation history
    """
    return [{"role": "system", "content": SYSTEM_PROMPT}] + history


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL,
        "active_sessions": len(sessions),
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """ Full history is sent to Groq on every request """
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty")

    session_id = get_or_create_session(req.session_id)
    session = sessions[session_id]

    # Append new user message
    session["history"].append({"role": "user", "content": req.message})

    # Call Groq with FULL history every time
    start = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=build_messages(session["history"]),   # ← Full history sent
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
    except Exception as e:
        session["history"].pop()
        raise HTTPException(status_code=502, detail=f"Groq API error: {str(e)}")

    latency_ms = (time.perf_counter() - start) * 1000
    reply = response.choices[0].message.content
    tokens_used = response.usage.prompt_tokens + response.usage.completion_tokens

    # Append assistant reply
    session["history"].append({"role": "assistant", "content": reply})

    session["message_count"] += 2
    session["total_tokens"] += tokens_used
    session["updated_at"] = datetime.utcnow().isoformat()

    return ChatResponse(
        reply=reply,
        session_id=session_id,
        message_count=session["message_count"],
        total_tokens=session["total_tokens"],
        latency_ms=round(latency_ms, 1),
    )


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """ Streaming version - Full history sent to Groq every time """
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty")

    session_id = get_or_create_session(req.session_id)
    session = sessions[session_id]

    session["history"].append({"role": "user", "content": req.message})

    def generate():
        full_reply = ""
        try:
            stream = client.chat.completions.create(
                model=MODEL,
                messages=build_messages(session["history"]),   # ← Full history sent
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                stream=True,
            )
            for chunk in stream:
                token = chunk.choices[0].delta.content or ""
                full_reply += token
                yield token
        except Exception as e:
            session["history"].pop()
            yield f"\n[ERROR: {str(e)}]"
            return

        session["history"].append({"role": "assistant", "content": full_reply})
        session["message_count"] += 2
        session["updated_at"] = datetime.utcnow().isoformat()

    return StreamingResponse(generate(), media_type="text/plain")


@app.get("/history/{session_id}", response_model=HistoryResponse)
def get_history(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="session not found")
    session = sessions[session_id]
    return HistoryResponse(
        session_id=session_id,
        messages=session["history"],
        message_count=session["message_count"],
        created_at=session["created_at"],
        updated_at=session["updated_at"],
    )


@app.delete("/history/{session_id}")
def clear_history(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="session not found")
    sessions[session_id]["history"] = []
    sessions[session_id]["message_count"] = 0
    sessions[session_id]["total_tokens"] = 0
    sessions[session_id]["updated_at"] = datetime.utcnow().isoformat()
    return {"status": "cleared", "session_id": session_id}


@app.get("/sessions", response_model=list[SessionSummary])
def list_sessions():
    return [
        SessionSummary(
            session_id=sid,
            message_count=data["message_count"],
            total_tokens=data["total_tokens"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
        )
        for sid, data in sessions.items()
    ]


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="session not found")
    del sessions[session_id]
    return {"status": "deleted", "session_id": session_id}