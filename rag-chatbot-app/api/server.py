"""
api/server.py
FastAPI production API with JWT auth, RBAC, rate limiting, caching,
audit logging, and endpoints: /health, /ask, /metrics, /audit.
"""
from __future__ import annotations

import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Optional

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError
from pydantic import BaseModel, Field

from api.audit import AuditLogger
from api.auth import (
    UserStore,
    authenticate_user,
    create_access_token,
    decode_access_token,
)
from api.monitoring import MetricsTracker
from src.config import cfg

# ── App lifecycle ─────────────────────────────────────────────────────────────

_chatbot = None  # lazy-loaded RAG chatbot


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _chatbot
    try:
        from src.chatbot import RAGChatbot
        _chatbot = RAGChatbot()
        print("[server] RAGChatbot initialized.")
    except Exception as exc:
        print(f"[server] WARNING: Could not initialize RAGChatbot: {exc}")
    yield
    print("[server] Shutting down.")


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG Chatbot API",
    version="1.0.0",
    description="Production RAG chatbot with JWT auth, RBAC, and audit logging.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Singletons ────────────────────────────────────────────────────────────────

_user_store = UserStore(cfg.auth.users_file)
_audit = AuditLogger(cfg.logging.audit_log_file)
_metrics = MetricsTracker(cfg.logging.metrics_file)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

# ── Rate limiting (simple in-memory token bucket) ─────────────────────────────

import threading
from collections import defaultdict

_rate_buckets: dict[str, list[float]] = defaultdict(list)
_rate_lock = threading.Lock()
_RATE_WINDOW = 60  # seconds
_RATE_LIMIT = cfg.api.rate_limit_per_minute


def _check_rate_limit(identifier: str) -> bool:
    """Return True if request is allowed, False if rate-limited."""
    now = time.time()
    with _rate_lock:
        bucket = _rate_buckets[identifier]
        # Remove timestamps outside the window
        _rate_buckets[identifier] = [t for t in bucket if now - t < _RATE_WINDOW]
        if len(_rate_buckets[identifier]) >= _RATE_LIMIT:
            return False
        _rate_buckets[identifier].append(now)
        return True


# ── Auth dependency ───────────────────────────────────────────────────────────

async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = decode_access_token(token)
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = _user_store.get_by_id(user_id)
    if user is None or user.disabled:
        raise credentials_exception
    return user


def require_role(minimum_role: str):
    """Dependency factory for role-based access control."""
    async def _check(current_user=Depends(get_current_user)):
        if not current_user.has_role(minimum_role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires role '{minimum_role}' or higher.",
            )
        return current_user
    return _check


# ── Request / Response models ─────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = Field(default=None)
    persona: Optional[str] = Field(default="default")
    stream: bool = Field(default=False)


class AskResponse(BaseModel):
    answer: str
    session_id: str
    sources: list[dict] = []
    quality: Optional[dict] = None
    cached: bool = False
    latency_ms: float = 0.0


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    expires_in: int


class HealthResponse(BaseModel):
    status: str
    version: str
    chatbot_ready: bool


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Health check endpoint – no auth required."""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        chatbot_ready=_chatbot is not None,
    )


@app.post("/auth/token", response_model=TokenResponse, tags=["Auth"])
async def login(
    request: Request,
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
):
    """Authenticate and receive a JWT access token."""
    ip = request.client.host if request.client else "unknown"

    if not _check_rate_limit(f"login:{ip}"):
        _audit.log("login_rate_limited", ip_address=ip, status="denied")
        raise HTTPException(status_code=429, detail="Too many login attempts.")

    user = authenticate_user(_user_store, form_data.username, form_data.password)
    if not user:
        _audit.log("login_failed", username=form_data.username, ip_address=ip, status="denied")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token(user.user_id, user.username, user.role)
    _audit.log("login", user_id=user.user_id, role=user.role, ip_address=ip, status="ok")

    return TokenResponse(
        access_token=token,
        role=user.role,
        expires_in=cfg.auth.token_expire_minutes * 60,
    )


@app.post("/ask", response_model=AskResponse, tags=["Chat"])
async def ask(
    request: Request,
    body: AskRequest,
    current_user=Depends(require_role("employee")),
):
    """Ask a question to the RAG chatbot."""
    ip = request.client.host if request.client else "unknown"
    start = time.time()

    # Rate limiting per user
    if not _check_rate_limit(f"ask:{current_user.user_id}"):
        _audit.log("ask_rate_limited", user_id=current_user.user_id, status="denied")
        raise HTTPException(status_code=429, detail="Rate limit exceeded.")

    if _chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized.")

    session_id = body.session_id or f"sess-{current_user.user_id}-{int(time.time())}"

    try:
        if body.stream:
            # Streaming response
            async def generate():
                async for chunk in _chatbot.astream(
                    body.question, session_id=session_id, persona=body.persona
                ):
                    yield chunk

            _audit.log(
                "ask_stream",
                user_id=current_user.user_id,
                role=current_user.role,
                ip_address=ip,
                session_id=session_id,
                question=body.question[:100],
                status="ok",
            )
            return StreamingResponse(generate(), media_type="text/plain")

        # Non-streaming
        result = _chatbot.ask(
            body.question,
            session_id=session_id,
            persona=body.persona,
        )

        latency_ms = (time.time() - start) * 1000
        _metrics.record(
            endpoint="/ask",
            latency_ms=latency_ms,
            status_code=200,
            user_id=current_user.user_id,
            cache_hit=result.get("cached", False),
            quality_score=result.get("quality", {}).get("overall") if result.get("quality") else None,
        )
        _audit.log(
            "ask",
            user_id=current_user.user_id,
            role=current_user.role,
            ip_address=ip,
            session_id=session_id,
            question=body.question[:100],
            cached=result.get("cached", False),
            status="ok",
        )

        return AskResponse(
            answer=result["answer"],
            session_id=session_id,
            sources=result.get("sources", []),
            quality=result.get("quality"),
            cached=result.get("cached", False),
            latency_ms=round(latency_ms, 2),
        )

    except Exception as exc:
        latency_ms = (time.time() - start) * 1000
        _metrics.record("/ask", latency_ms, 500, current_user.user_id)
        _audit.log(
            "ask_error",
            user_id=current_user.user_id,
            ip_address=ip,
            error=str(exc),
            status="error",
        )
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")


@app.get("/metrics", tags=["Admin"])
async def get_metrics(current_user=Depends(require_role("hr_admin"))):
    """Return API metrics. Requires hr_admin or admin role."""
    return _metrics.get_summary()


@app.get("/audit", tags=["Admin"])
async def get_audit(
    n: int = 100,
    current_user=Depends(require_role("admin")),
):
    """Return recent audit log entries. Requires admin role."""
    return _audit.read_recent(n)


@app.post("/ingest", tags=["Admin"])
async def ingest_documents(
    directory: str = "./data/documents",
    current_user=Depends(require_role("admin")),
):
    """Ingest documents from a directory. Requires admin role."""
    if _chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized.")
    try:
        count = _chatbot.ingest(directory)
        _audit.log(
            "ingest",
            user_id=current_user.user_id,
            directory=directory,
            documents_ingested=count,
            status="ok",
        )
        return {"status": "ok", "documents_ingested": count}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/feedback", tags=["Chat"])
async def submit_feedback(
    session_id: str,
    question: str,
    answer: str,
    feedback: str,  # "positive" | "negative"
    comment: Optional[str] = None,
    current_user=Depends(require_role("employee")),
):
    """Submit thumbs up/down feedback for an answer."""
    if _chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized.")
    if feedback not in ("positive", "negative"):
        raise HTTPException(status_code=400, detail="feedback must be 'positive' or 'negative'")

    entry = _chatbot.feedback_store.record(
        session_id=session_id,
        question=question,
        answer=answer,
        context="",
        feedback=feedback,  # type: ignore
        comment=comment,
        user_id=current_user.user_id,
    )
    return {"feedback_id": entry.feedback_id, "status": "recorded"}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.server:app",
        host=cfg.api.host,
        port=cfg.api.port,
        workers=1,  # use 1 for dev; cfg.api.workers for prod
        reload=False,
    )
