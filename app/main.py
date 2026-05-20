"""
DevOps AI Copilot with RAG (ChromaDB + Sentence Transformers)
Full-featured FastAPI application
"""

import os
import uuid
import time
from datetime import datetime
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

# RAG
from sentence_transformers import SentenceTransformer
import chromadb

load_dotenv()

if not os.environ.get("GROQ_API_KEY"):
    print("⚠️  WARNING: GROQ_API_KEY is not set!")

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="devops_knowledge")
# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DevOps AI Copilot with RAG",
    description="Multi-turn chat with full history + Vector RAG (ChromaDB)",
    version="1.1.0",
)

# ── Clients ───────────────────────────────────────────────────────────────────
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Embedding Model & Vector DB
embedder = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="devops_knowledge")

MODEL         = "llama-3.1-8b-instant"
MAX_TOKENS    = 1024
TEMPERATURE   = 0.7

SYSTEM_PROMPT = """You are a senior DevOps engineer and infrastructure expert.
Use the provided context from knowledge base when relevant.
Be concise, practical, and always include commands or YAML snippets."""

# ── In-memory Session Store ───────────────────────────────────────────────────
sessions: dict[str, dict] = {}


# ── Pydantic Models ───────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    stream: bool = False


class IngestRequest(BaseModel):
    documents: List[str]
    metadatas: Optional[List[dict]] = None


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    message_count: int
    total_tokens: int
    latency_ms: float


class IngestResponse(BaseModel):
    status: str
    added_count: int
    message: str


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
        "history": [],
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "message_count": 0,
        "total_tokens": 0
    }
    return new_id


def build_messages(history: list, context: str = "") -> list:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if context:
        messages.append({"role": "system", "content": f"Relevant Context from Knowledge Base:\n{context}"})
    messages.extend(history)
    return messages


def retrieve_context(query: str, top_k: int = 3) -> str:
    """Retrieve relevant DevOps knowledge using embeddings"""
    query_embedding = embedder.encode([query])[0].tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    if results and results['documents'] and results['documents'][0]:
        return "\n\n".join(results['documents'][0])
    return ""


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL,
        "rag_enabled": True,
        "knowledge_base_size": collection.count(),
        "active_sessions": len(sessions)
    }


@app.post("/ingest", response_model=IngestResponse)
def ingest_knowledge(req: IngestRequest):
    """Add DevOps documents, Kubernetes YAMLs, troubleshooting guides, etc."""
    if not req.documents or len(req.documents) == 0:
        raise HTTPException(status_code=400, detail="documents list cannot be empty")

    metadatas = req.metadatas or [{} for _ in req.documents]
    embeddings = embedder.encode(req.documents).tolist()

    collection.add(
        documents=req.documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=[str(uuid.uuid4()) for _ in req.documents]
    )

    return IngestResponse(
        status="success",
        added_count=len(req.documents),
        message=f"Successfully added {len(req.documents)} document(s) to knowledge base."
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty")

    session_id = get_or_create_session(req.session_id)
    session = sessions[session_id]

    # Append user message
    session["history"].append({"role": "user", "content": req.message})

    # Retrieve context from RAG
    context = retrieve_context(req.message)

    # Call Groq
    start = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=build_messages(session["history"], context),
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
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty")

    session_id = get_or_create_session(req.session_id)
    session = sessions[session_id]

    session["history"].append({"role": "user", "content": req.message})
    context = retrieve_context(req.message)

    def generate():
        full_reply = ""
        try:
            stream = client.chat.completions.create(
                model=MODEL,
                messages=build_messages(session["history"], context),
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


@app.get("/sessions", response_model=List[SessionSummary])
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

@app.get("/chroma/status")
def chroma_status():
    return {
        "status": "running",
        "mode": "embedded_persistent",
        "storage_path": "./chroma_db",
        "collection": "devops_knowledge",
        "document_count": collection.count()
    }