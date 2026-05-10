"""
api/auth.py
Local JWT auth.

Spec-required functions:
  create_user(username, password, role) -> User
  authenticate_user(username, password) -> User | None
  create_token(user_id, username, role) -> str
  verify_token(token) -> dict
  setup_default_users()

Default users: admin/admin123, hr_manager/hr123, employee1/emp123
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext

# ── Config ────────────────────────────────────────────────────────────────────

SECRET_KEY: str = os.getenv("SECRET_KEY", "change-me-in-production-use-env-var-32chars!")
ALGORITHM: str = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("TOKEN_EXPIRE_MINUTES", "60"))

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

VALID_ROLES = {"employee", "hr_admin", "admin"}
ROLE_HIERARCHY = ["employee", "hr_admin", "admin"]

_DEFAULT_USERS_FILE = "./data/users.json"


# ── User model ────────────────────────────────────────────────────────────────

class User:
    def __init__(
        self,
        user_id: str,
        username: str,
        hashed_password: str,
        role: str,
        disabled: bool = False,
    ) -> None:
        self.user_id = user_id
        self.username = username
        self.hashed_password = hashed_password
        self.role = role
        self.disabled = disabled

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "username": self.username,
            "hashed_password": self.hashed_password,
            "role": self.role,
            "disabled": self.disabled,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "User":
        return cls(**data)

    def has_role(self, required_role: str) -> bool:
        try:
            return ROLE_HIERARCHY.index(self.role) >= ROLE_HIERARCHY.index(required_role)
        except ValueError:
            return False


# ── User store ────────────────────────────────────────────────────────────────

class UserStore:
    """Thread-safe local JSON user store."""

    def __init__(self, users_file: str = _DEFAULT_USERS_FILE) -> None:
        self._path = Path(users_file)
        self._lock = threading.Lock()
        setup_default_users(self._path)

    def _load(self) -> list[dict]:
        with open(self._path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _save(self, users: list[dict]) -> None:
        with open(self._path, "w", encoding="utf-8") as fh:
            json.dump(users, fh, indent=2)

    def get_by_username(self, username: str) -> Optional[User]:
        with self._lock:
            for u in self._load():
                if u["username"] == username:
                    return User.from_dict(u)
        return None

    def get_by_id(self, user_id: str) -> Optional[User]:
        with self._lock:
            for u in self._load():
                if u["user_id"] == user_id:
                    return User.from_dict(u)
        return None

    def all_users(self) -> list[User]:
        with self._lock:
            return [User.from_dict(u) for u in self._load()]

    def save_user(self, user: User) -> None:
        with self._lock:
            users = self._load()
            for i, u in enumerate(users):
                if u["user_id"] == user.user_id:
                    users[i] = user.to_dict()
                    self._save(users)
                    return
            users.append(user.to_dict())
            self._save(users)

    def disable_user(self, username: str) -> bool:
        with self._lock:
            users = self._load()
            for u in users:
                if u["username"] == username:
                    u["disabled"] = True
                    self._save(users)
                    return True
        return False


# ── Spec-required standalone functions ───────────────────────────────────────

def setup_default_users(path: Optional[Path] = None) -> None:
    """
    Create the users file with default users if it doesn't exist.
    Default users: admin/admin123, hr_manager/hr123, employee1/emp123
    """
    p = Path(path) if path else Path(_DEFAULT_USERS_FILE)
    if p.exists():
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    defaults = [
        {
            "user_id": "usr-001",
            "username": "admin",
            "hashed_password": _pwd_context.hash("admin123"),
            "role": "admin",
            "disabled": False,
        },
        {
            "user_id": "usr-002",
            "username": "hr_manager",
            "hashed_password": _pwd_context.hash("hr123"),
            "role": "hr_admin",
            "disabled": False,
        },
        {
            "user_id": "usr-003",
            "username": "employee1",
            "hashed_password": _pwd_context.hash("emp123"),
            "role": "employee",
            "disabled": False,
        },
    ]
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(defaults, fh, indent=2)


def create_user(
    username: str,
    password: str,
    role: str = "employee",
    users_file: str = _DEFAULT_USERS_FILE,
) -> User:
    """Create a new user and persist to the users file."""
    if role not in VALID_ROLES:
        raise ValueError(f"Invalid role '{role}'. Must be one of {VALID_ROLES}")
    path = Path(users_file)
    setup_default_users(path)
    with open(path, "r", encoding="utf-8") as fh:
        users = json.load(fh)
    if any(u["username"] == username for u in users):
        raise ValueError(f"Username '{username}' already exists.")
    user_id = f"usr-{len(users) + 1:03d}"
    new_user = {
        "user_id": user_id,
        "username": username,
        "hashed_password": _pwd_context.hash(password),
        "role": role,
        "disabled": False,
    }
    users.append(new_user)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(users, fh, indent=2)
    return User.from_dict(new_user)


def authenticate_user(
    username: str,
    password: str,
    users_file: str = _DEFAULT_USERS_FILE,
) -> Optional[User]:
    """
    Authenticate a user by username and password.
    Returns User if valid, None otherwise.

    Also accepts (UserStore, username, password) for backward compat.
    """
    # Backward compat: first arg may be a UserStore
    if isinstance(username, UserStore):
        store: UserStore = username
        username = password
        password = users_file  # type: ignore[assignment]
        user = store.get_by_username(username)
        if user is None or user.disabled:
            return None
        return user if _pwd_context.verify(password, user.hashed_password) else None

    path = Path(users_file)
    setup_default_users(path)
    with open(path, "r", encoding="utf-8") as fh:
        users = json.load(fh)
    for u in users:
        if u["username"] == username and not u.get("disabled", False):
            if _pwd_context.verify(password, u["hashed_password"]):
                return User.from_dict(u)
    return None


def create_token(
    user_id: str,
    username: str,
    role: str,
    expires_minutes: int = ACCESS_TOKEN_EXPIRE_MINUTES,
) -> str:
    """Create a signed JWT access token."""
    expire = datetime.now(timezone.utc) + timedelta(minutes=expires_minutes)
    payload = {
        "sub": user_id,
        "username": username,
        "role": role,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(token: str) -> dict:
    """
    Decode and verify a JWT token.
    Returns the payload dict.
    Raises JWTError on invalid/expired tokens.
    """
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])


# ── Backward-compat aliases ───────────────────────────────────────────────────

def create_access_token(
    user_id: str,
    username: str,
    role: str,
    expires_delta=None,
) -> str:
    minutes = int(expires_delta.total_seconds() / 60) if expires_delta else ACCESS_TOKEN_EXPIRE_MINUTES
    return create_token(user_id, username, role, expires_minutes=minutes)


def decode_access_token(token: str) -> dict:
    return verify_token(token)


def verify_password(plain: str, hashed: str) -> bool:
    return _pwd_context.verify(plain, hashed)


def hash_password(plain: str) -> str:
    return _pwd_context.hash(plain)
