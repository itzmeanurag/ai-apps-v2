"""
api/server.py
FastAPI production API.

Endpoints: /login, /health, /ask, /metrics, /audit, /audit/stats,
           /ingest, /dashboard

Features: ResponseCache, rate limiter, lifespan startup,
          Pydantic models for all request/response.
"""
from __future__ import annotations

import sys
import time
import threading
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError
from pydantic import BaseModel, Field

from api.audit import AuditLogger
from api.auth import (
    UserStore, User,
    authenticate_user, create_token, verify_token,
    setup_default_users,
)
from api.monitoring import MetricsTracker
from src.config import cfg


# ── ResponseCache ─────────────────────────────────────────────────────────────

class ResponseCache:
    """
    Simple in-memory response cache with TTL.
    Keyed by (question, session_id).
    """

    def __init__(self, ttl_seconds: int = 300) -> None:
        self._ttl = ttl_seconds
        self._cache: dict[str, tuple[Any, float]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._cache.get(key)
            if entry and time.time() - entry[1] < self._ttl:
                return entry[0]
            if entry:
                del self._cache[key]
        return None

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._cache[key] = (value, time.time())

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def stats(self) -> dict:
        with self._lock:
            now = time.time()
            live = sum(1 for _, (_, ts) in self._cache.items() if now - ts < self._ttl)
            return {"total_entries": len(self._cache), "live_entries": live}


# ── Rate limiter ──────────────────────────────────────────────────────────────

_rate_buckets: dict[str, list[float]] = defaultdict(list)
_rate_lock = threading.Lock()
_RATE_WINDOW = 60
_RATE_LIMIT = cfg.api.rate_limit_per_minute


def _check_rate_limit(identifier: str) -> bool:
    now = time.time()
    with _rate_lock:
        _rate_buckets[identifier] = [t for t in _rate_buckets[identifier] if now - t < _RATE_WINDOW]
        if len(_rate_buckets[identifier]) >= _RATE_LIMIT:
            return False
        _rate_buckets[identifier].append(now)
        return True


# ── App lifecycle ─────────────────────────────────────────────────────────────

_chatbot = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _chatbot
    setup_default_users()
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
    description="Production RAG chatbot — JWT auth, RBAC, audit logging.",
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
_cache = ResponseCache(ttl_seconds=cfg.api.cache_ttl_seconds)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")


# ── Auth dependencies ─────────────────────────────────────────────────────────

async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]) -> User:
    exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = verify_token(token)
        user_id: str = payload.get("sub", "")
        if not user_id:
            raise exc
    except JWTError:
        raise exc
    user = _user_store.get_by_id(user_id)
    if user is None or user.disabled:
        raise exc
    return user


def require_role(minimum_role: str):
    async def _check(current_user: User = Depends(get_current_user)) -> User:
        if not current_user.has_role(minimum_role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires role '{minimum_role}' or higher.",
            )
        return current_user
    return _check


# ── Pydantic models ───────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    username: str
    expires_in: int


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None
    persona: str = "default"
    stream: bool = False


class AskResponse(BaseModel):
    answer: str
    session_id: str
    sources: list[dict] = []
    quality: Optional[dict] = None
    cached: bool = False
    latency_ms: float = 0.0
    confidence: float = 0.0


class HealthResponse(BaseModel):
    status: str
    version: str
    chatbot_ready: bool
    cache_stats: dict = {}


class IngestResponse(BaseModel):
    status: str
    documents_ingested: int
    directory: str


class DashboardResponse(BaseModel):
    metrics: dict
    audit_stats: dict
    cache_stats: dict
    chatbot_ready: bool


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Health check — no auth required."""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        chatbot_ready=_chatbot is not None,
        cache_stats=_cache.stats(),
    )


@app.post("/login", response_model=TokenResponse, tags=["Auth"])
async def login(request: Request, form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    """Authenticate and receive a JWT access token."""
    ip = request.client.host if request.client else "unknown"

    if not _check_rate_limit(f"login:{ip}"):
        _audit.log_auth_event("login_rate_limited", form_data.username, ip, success=False)
        raise HTTPException(status_code=429, detail="Too many login attempts.")

    user = authenticate_user(form_data.username, form_data.password, cfg.auth.users_file)
    if not user:
        _audit.log_auth_event("login_failed", form_data.username, ip, success=False)
        _metrics.record_auth("login_failed", success=False)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_token(user.user_id, user.username, user.role)
    _audit.log_auth_event("login", user.username, ip, success=True, role=user.role)
    _metrics.record_auth("login", success=True)

    return TokenResponse(
        access_token=token,
        role=user.role,
        username=user.username,
        expires_in=cfg.auth.token_expire_minutes * 60,
    )


# Keep /auth/token as an alias for OAuth2 compatibility
@app.post("/auth/token", response_model=TokenResponse, tags=["Auth"], include_in_schema=False)
async def login_compat(request: Request, form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    return await login(request, form_data)


@app.post("/ask", response_model=AskResponse, tags=["Chat"])
async def ask(
    request: Request,
    body: AskRequest,
    current_user: User = Depends(require_role("employee")),
):
    """Ask a question to the RAG chatbot."""
    ip = request.client.host if request.client else "unknown"
    start = time.time()

    if not _check_rate_limit(f"ask:{current_user.user_id}"):
        _audit.log("ask_rate_limited", user_id=current_user.user_id, status="denied")
        raise HTTPException(status_code=429, detail="Rate limit exceeded.")

    if _chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized.")

    session_id = body.session_id or f"sess-{current_user.user_id}-{int(time.time())}"

    # Check response cache
    cache_key = f"{body.question}::{session_id}"
    cached_resp = _cache.get(cache_key)
    if cached_resp and not body.stream:
        latency_ms = (time.time() - start) * 1000
        _metrics.record_request("/ask", latency_ms, 200, current_user.user_id, cache_hit=True)
        return AskResponse(**{**cached_resp, "cached": True, "latency_ms": round(latency_ms, 2)})

    try:
        if body.stream:
            async def generate():
                async for chunk in _chatbot.ask_stream_async(
                    body.question, session_id=session_id, persona=body.persona
                ):
                    yield chunk
            _audit.log_question(current_user.user_id, body.question, "[streaming]", session_id)
            return StreamingResponse(generate(), media_type="text/plain")

        result = _chatbot.ask(body.question, session_id=session_id, persona=body.persona)
        latency_ms = (time.time() - start) * 1000

        _metrics.record_request(
            "/ask", latency_ms, 200, current_user.user_id,
            cache_hit=result.get("cached", False),
            quality_score=result.get("quality", {}).get("overall") if result.get("quality") else None,
        )
        _audit.log_question(
            current_user.user_id, body.question, result["answer"],
            session_id=session_id,
            quality=result.get("quality"),
            latency_ms=latency_ms,
            cached=result.get("cached", False),
            blocked=result.get("blocked", False),
        )

        resp_data = {
            "answer": result["answer"],
            "session_id": session_id,
            "sources": result.get("sources", []),
            "quality": result.get("quality"),
            "cached": result.get("cached", False),
            "latency_ms": round(latency_ms, 2),
            "confidence": result.get("confidence", 0.0),
        }
        _cache.set(cache_key, resp_data)
        return AskResponse(**resp_data)

    except Exception as exc:
        latency_ms = (time.time() - start) * 1000
        _metrics.record_request("/ask", latency_ms, 500, current_user.user_id)
        _audit.log("ask_error", user_id=current_user.user_id, error=str(exc), status="error")
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")


@app.get("/metrics", tags=["Admin"])
async def get_metrics(current_user: User = Depends(require_role("hr_admin"))):
    """Return API metrics. Requires hr_admin or admin role."""
    return _metrics.get_summary()


@app.get("/audit", tags=["Admin"])
async def get_audit(
    n: int = 100,
    event_type: Optional[str] = None,
    user_id: Optional[str] = None,
    current_user: User = Depends(require_role("admin")),
):
    """Return recent audit log entries. Requires admin role."""
    return _audit.query_logs(event_type=event_type, user_id=user_id, limit=n)


@app.get("/audit/stats", tags=["Admin"])
async def get_audit_stats(current_user: User = Depends(require_role("admin"))):
    """Return aggregate audit statistics. Requires admin role."""
    return _audit.get_stats()


@app.post("/ingest", response_model=IngestResponse, tags=["Admin"])
async def ingest_documents(
    directory: str = "./data/documents",
    current_user: User = Depends(require_role("admin")),
):
    """Ingest documents from a directory. Requires admin role."""
    if _chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized.")
    try:
        count = _chatbot.ingest_documents(directory)
        _audit.log_admin_action(
            current_user.user_id, "ingest",
            {"directory": directory, "documents_ingested": count},
        )
        return IngestResponse(status="ok", documents_ingested=count, directory=directory)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/dashboard", response_model=DashboardResponse, tags=["Admin"])
async def get_dashboard(current_user: User = Depends(require_role("hr_admin"))):
    """Combined dashboard: metrics + audit stats + cache stats."""
    return DashboardResponse(
        metrics=_metrics.get_summary(),
        audit_stats=_audit.get_stats(),
        cache_stats=_cache.stats(),
        chatbot_ready=_chatbot is not None,
    )


@app.post("/feedback", tags=["Chat"])
async def submit_feedback(
    session_id: str,
    question: str,
    answer: str,
    feedback: str,
    comment: Optional[str] = None,
    current_user: User = Depends(require_role("employee")),
):
    """Submit thumbs up/down feedback."""
    if _chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized.")
    if feedback not in ("positive", "negative"):
        raise HTTPException(status_code=400, detail="feedback must be 'positive' or 'negative'")
    entry = _chatbot.feedback.record(
        question=question, answer=answer,
        rating=5 if feedback == "positive" else 1,
        comment=comment or "", session_id=session_id,
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
        workers=1,
        reload=False,
    )
