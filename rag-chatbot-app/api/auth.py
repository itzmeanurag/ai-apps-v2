"""
api/auth.py
JWT create/verify, BCrypt passwords, local JSON user store.
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

# ── User model ────────────────────────────────────────────────────────────────

VALID_ROLES = {"employee", "hr_admin", "admin"}

# Role hierarchy: higher index = more permissions
ROLE_HIERARCHY = ["employee", "hr_admin", "admin"]


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
        """Return True if user's role is >= required_role in hierarchy."""
        try:
            user_level = ROLE_HIERARCHY.index(self.role)
            required_level = ROLE_HIERARCHY.index(required_role)
            return user_level >= required_level
        except ValueError:
            return False


# ── User store ────────────────────────────────────────────────────────────────

class UserStore:
    """
    Local JSON-backed user store.
    Thread-safe reads and writes.
    """

    def __init__(self, users_file: str = "./data/users.json") -> None:
        self._path = Path(users_file)
        self._lock = threading.Lock()
        self._ensure_defaults()

    def _ensure_defaults(self) -> None:
        """Create the users file with a default admin if it doesn't exist."""
        if self._path.exists():
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        default_users = [
            {
                "user_id": "usr-001",
                "username": "admin",
                "hashed_password": _pwd_context.hash(
                    os.getenv("ADMIN_PASSWORD", "admin123")
                ),
                "role": "admin",
                "disabled": False,
            },
            {
                "user_id": "usr-002",
                "username": "hr_user",
                "hashed_password": _pwd_context.hash("hr_password"),
                "role": "hr_admin",
                "disabled": False,
            },
            {
                "user_id": "usr-003",
                "username": "employee1",
                "hashed_password": _pwd_context.hash("emp_password"),
                "role": "employee",
                "disabled": False,
            },
        ]
        with open(self._path, "w", encoding="utf-8") as fh:
            json.dump(default_users, fh, indent=2)

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

    def create_user(
        self,
        username: str,
        password: str,
        role: str = "employee",
    ) -> User:
        if role not in VALID_ROLES:
            raise ValueError(f"Invalid role '{role}'. Must be one of {VALID_ROLES}")
        with self._lock:
            users = self._load()
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
            self._save(users)
            return User.from_dict(new_user)

    def update_password(self, username: str, new_password: str) -> bool:
        with self._lock:
            users = self._load()
            for u in users:
                if u["username"] == username:
                    u["hashed_password"] = _pwd_context.hash(new_password)
                    self._save(users)
                    return True
        return False

    def disable_user(self, username: str) -> bool:
        with self._lock:
            users = self._load()
            for u in users:
                if u["username"] == username:
                    u["disabled"] = True
                    self._save(users)
                    return True
        return False


# ── Password helpers ──────────────────────────────────────────────────────────

def verify_password(plain: str, hashed: str) -> bool:
    return _pwd_context.verify(plain, hashed)


def hash_password(plain: str) -> str:
    return _pwd_context.hash(plain)


# ── JWT helpers ───────────────────────────────────────────────────────────────

def create_access_token(
    user_id: str,
    username: str,
    role: str,
    expires_delta: Optional[timedelta] = None,
) -> str:
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    payload = {
        "sub": user_id,
        "username": username,
        "role": role,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict:
    """
    Decode and validate a JWT token.
    Raises JWTError on invalid/expired tokens.
    """
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])


def authenticate_user(store: UserStore, username: str, password: str) -> Optional[User]:
    """Return User if credentials are valid, else None."""
    user = store.get_by_username(username)
    if user is None or user.disabled:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user
