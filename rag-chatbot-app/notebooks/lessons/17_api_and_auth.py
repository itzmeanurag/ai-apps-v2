"""
LESSON 17: Production API — FastAPI + JWT + RBAC
=================================================
CONCEPT: Securing the chatbot for multi-user production use

WHAT THIS DOES:
  Demonstrates the production API from api/server.py:
    - JWT authentication (login → token → authorized requests)
    - Role-based access control (employee / hr_admin / admin)
    - Rate limiting (per-user request throttling)
    - Audit logging (every request recorded)
    - Metrics tracking (latency, quality, cache hits)

WHY THIS MATTERS:
  A chatbot without auth is open to anyone.
  Without rate limiting, one user can overwhelm the server.
  Without audit logging, you can't investigate incidents.
  This lesson shows how to run the API and test all its features.

RUN (from project root):
  python notebooks/lessons/17_api_and_auth.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from api.auth import (
    UserStore, authenticate_user, create_access_token,
    decode_access_token, hash_password, verify_password,
)
from api.audit import AuditLogger
from api.monitoring import MetricsTracker
from src.config import cfg


# ── Demo 1: JWT authentication ────────────────────────────────────────────────

def demo_jwt_auth() -> None:
    """Show the JWT authentication flow."""
    print("\n--- DEMO 1: JWT Authentication ---")

    store = UserStore(cfg.auth.users_file)

    # Authenticate
    user = authenticate_user(store, "admin", "admin123")
    if user:
        print(f"\n  ✅ Login successful: {user.username} (role: {user.role})")

        # Create token
        token = create_access_token(user.user_id, user.username, user.role)
        print(f"  Token (first 50 chars): {token[:50]}...")

        # Decode and verify
        payload = decode_access_token(token)
        print(f"\n  Decoded payload:")
        print(f"    sub (user_id): {payload['sub']}")
        print(f"    username:      {payload['username']}")
        print(f"    role:          {payload['role']}")
        print(f"    exp:           {payload['exp']} (Unix timestamp)")
    else:
        print("  ❌ Login failed (check users.json exists)")

    # Failed login
    bad_user = authenticate_user(store, "admin", "wrongpassword")
    print(f"\n  Wrong password: {'❌ Rejected' if bad_user is None else '⚠️ Accepted (bug!)'}")

    print("""
  JWT FLOW:
    1. POST /auth/token  username=admin&password=admin123
    2. Server returns:   {"access_token": "eyJ...", "role": "admin"}
    3. Client stores token
    4. Every request:    Authorization: Bearer eyJ...
    5. Server validates token on each request (no database lookup needed)

  WHY JWT?
    Stateless — server doesn't need to store sessions.
    Self-contained — token contains user_id, role, expiry.
    Signed — can't be forged without the SECRET_KEY.
""")


# ── Demo 2: Role-based access control ────────────────────────────────────────

def demo_rbac() -> None:
    """Show role hierarchy and permission checking."""
    print("\n--- DEMO 2: Role-Based Access Control (RBAC) ---")

    store = UserStore(cfg.auth.users_file)

    users_to_test = [
        ("admin",     "admin123"),
        ("hr_user",   "hr_password"),
        ("employee1", "emp_password"),
    ]

    print(f"\n  {'Username':<12} {'Role':<10} {'Ask':<6} {'Metrics':<10} {'Audit':<8} {'Ingest'}")
    print("  " + "-" * 60)

    for username, password in users_to_test:
        user = authenticate_user(store, username, password)
        if user:
            can_ask     = "✅" if user.has_role("employee") else "❌"
            can_metrics = "✅" if user.has_role("hr_admin") else "❌"
            can_audit   = "✅" if user.has_role("admin") else "❌"
            can_ingest  = "✅" if user.has_role("admin") else "❌"
            print(f"  {user.username:<12} {user.role:<10} {can_ask:<6} {can_metrics:<10} {can_audit:<8} {can_ingest}")
        else:
            print(f"  {username:<12} (login failed)")

    print(f"""
  ROLE HIERARCHY (lowest → highest):
    employee  → can ask questions, submit feedback
    hr_admin  → employee + view metrics
    admin     → hr_admin + view audit log, ingest documents

  Higher roles inherit all lower role permissions.
  user.has_role("hr_admin") returns True for both hr_admin AND admin.
""")


# ── Demo 3: Audit logging ─────────────────────────────────────────────────────

def demo_audit_logging() -> None:
    """Show the audit logger."""
    print("\n--- DEMO 3: Audit Logging ---")

    logger = AuditLogger("./data/audit_demo.jsonl")

    # Log some events
    logger.log("login",    user_id="usr-001", role="admin",    ip_address="192.168.1.1", status="ok")
    logger.log("ask",      user_id="usr-003", role="employee", ip_address="10.0.0.5",
               question="How many leave days?", cached=False, status="ok")
    logger.log("ask",      user_id="usr-003", role="employee", ip_address="10.0.0.5",
               question="Remote work policy?", cached=True, status="ok")
    logger.log("ask_error",user_id="usr-002", role="hr_admin", ip_address="10.0.0.8",
               error="Connection refused", status="error")

    # Read back
    entries = logger.read_recent(10)
    print(f"\n  Logged {len(entries)} audit entries:")
    for entry in entries:
        print(f"  [{entry['timestamp'][:19]}] {entry['event']:<15} "
              f"user={entry.get('user_id','?'):<8} status={entry['status']}")

    # Clean up
    Path("./data/audit_demo.jsonl").unlink(missing_ok=True)

    print(f"""
  AUDIT LOG FORMAT (JSONL — one JSON per line):
    {{"timestamp": "2025-01-01T10:00:00Z", "event": "ask",
      "user_id": "usr-003", "role": "employee",
      "ip": "10.0.0.5", "question": "How many leave days?",
      "cached": false, "status": "ok"}}

  STORED AT: {cfg.logging.audit_log_file}
  ACCESSIBLE VIA: GET /audit (admin role required)
""")


# ── Demo 4: How to run the API ────────────────────────────────────────────────

def demo_run_api() -> None:
    """Show how to start and test the API."""
    print("\n--- DEMO 4: Running the Production API ---")
    print(f"""
  START THE API:
    pip install -r requirements-prod.txt
    uvicorn api.server:app --host 0.0.0.0 --port {cfg.api.port} --reload

  SWAGGER UI (interactive docs):
    http://localhost:{cfg.api.port}/docs

  TEST WITH CURL:

    # 1. Health check (no auth needed)
    curl http://localhost:{cfg.api.port}/health

    # 2. Login
    curl -X POST http://localhost:{cfg.api.port}/auth/token \\
      -d "username=admin&password=admin123"
    # Returns: {{"access_token": "eyJ...", "role": "admin"}}

    # 3. Ask a question
    curl -X POST http://localhost:{cfg.api.port}/ask \\
      -H "Authorization: Bearer YOUR_TOKEN" \\
      -H "Content-Type: application/json" \\
      -d '{{"question": "How many leave days?"}}'

    # 4. View metrics (hr_admin+)
    curl http://localhost:{cfg.api.port}/metrics \\
      -H "Authorization: Bearer YOUR_TOKEN"

    # 5. View audit log (admin only)
    curl http://localhost:{cfg.api.port}/audit?n=20 \\
      -H "Authorization: Bearer YOUR_TOKEN"

  DEFAULT USERS (change passwords in production!):
    admin     / admin123    → admin role
    hr_user   / hr_password → hr_admin role
    employee1 / emp_password → employee role

  RATE LIMITING: {cfg.api.rate_limit_per_minute} requests/minute per user
  CORS ORIGINS:  {cfg.api.cors_origins}
""")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("LESSON 17: PRODUCTION API")
    print("=" * 60)
    print(f"""
FastAPI server (api/server.py):
  JWT auth + RBAC + rate limiting + audit logging + metrics

Endpoints:
  POST /auth/token  → get JWT token
  GET  /health      → health check (no auth)
  POST /ask         → ask a question (employee+)
  GET  /metrics     → API metrics (hr_admin+)
  GET  /audit       → audit log (admin only)
  POST /ingest      → ingest documents (admin only)
""")

    demo_jwt_auth()
    demo_rbac()
    demo_audit_logging()
    demo_run_api()


if __name__ == "__main__":
    main()
