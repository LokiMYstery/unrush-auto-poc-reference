from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from app import config

_TOKEN_STATE_FILE = "resource-auth-token-state.json"
_MEMORY_TOKENS: dict[str, "ScopedToken"] = {}
_REFRESH_TASKS: dict[tuple[str, int], asyncio.Task[bool]] = {}
_LAST_PERSIST_ERROR: str | None = None


@dataclass(frozen=True)
class ScopedToken:
    access_token: str
    refresh_token: str
    scope: dict[str, str]


def auth_headers(base_url: str) -> tuple[dict[str, str], str]:
    headers = {
        "Language": os.getenv("AUTO_RESOURCE_LANGUAGE", "zh-CN"),
        "Accept-Language": os.getenv("AUTO_RESOURCE_LANGUAGE", "zh-CN"),
        "X-Region": os.getenv("AUTO_RESOURCE_REGION", "CN"),
    }
    token = _access_token(base_url)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers, token


async def refresh_resource_token(base_url: str, *, rejected_access_token: str) -> bool:
    scope = _scope(base_url)
    scope_key = _scope_key(scope)
    current = _access_token(base_url)
    if rejected_access_token and current and current != rejected_access_token:
        return True
    refresh_token = _refresh_token(base_url)
    if not refresh_token:
        return False
    loop = asyncio.get_running_loop()
    task_key = (scope_key, id(loop))
    task = _REFRESH_TASKS.get(task_key)
    if task is None or task.done():
        task = loop.create_task(_refresh_resource_token_once(base_url.rstrip("/"), scope, rejected_access_token, refresh_token))
        _REFRESH_TASKS[task_key] = task
        task.add_done_callback(lambda finished, key=task_key: _forget_refresh_task(key, finished))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        raise
    except Exception:
        return False


def _forget_refresh_task(key: tuple[str, int], finished: asyncio.Task[bool]) -> None:
    if _REFRESH_TASKS.get(key) is finished:
        _REFRESH_TASKS.pop(key, None)


async def _refresh_resource_token_once(base_url: str, scope: dict[str, str], rejected_access_token: str, refresh_token: str) -> bool:
    current = _access_token_for_scope(scope)
    if rejected_access_token and current and current != rejected_access_token:
        return True
    try:
        async with httpx.AsyncClient(timeout=float(os.getenv("AUTO_RESOURCE_AUTH_TIMEOUT_SECONDS", "6"))) as client:
            response = await client.post(
                f"{base_url}/api/auth/refresh-token",
                json={"refreshToken": refresh_token},
                headers=_refresh_headers(),
            )
    except Exception:
        return False
    if response.status_code >= 400:
        return False
    try:
        payload = response.json()
    except Exception:
        return False
    if not _business_ok(payload):
        return False
    new_access = _find_value(payload, "newAccessToken") or _find_value(payload, "accessToken")
    new_refresh = _find_value(payload, "newRefreshToken") or _find_value(payload, "refreshToken")
    if not isinstance(new_access, str) or not new_access.strip():
        return False
    token = ScopedToken(
        access_token=new_access.strip(),
        refresh_token=new_refresh.strip() if isinstance(new_refresh, str) and new_refresh.strip() else refresh_token,
        scope=scope,
    )
    _MEMORY_TOKENS[_scope_key(scope)] = token
    _save_token_state(token)
    return True


def _access_token(base_url: str) -> str:
    scope = _scope(base_url)
    return _access_token_for_scope(scope)


def _access_token_for_scope(scope: dict[str, str]) -> str:
    token = _token_for_scope(scope)
    if token:
        return token.access_token
    return os.getenv("AUTO_RESOURCE_AUTH_TOKEN", "").strip()


def _refresh_token(base_url: str) -> str:
    scope = _scope(base_url)
    token = _token_for_scope(scope)
    if token:
        return token.refresh_token
    return os.getenv("AUTO_RESOURCE_REFRESH_TOKEN", "").strip()


def _token_for_scope(scope: dict[str, str]) -> ScopedToken | None:
    scope_key = _scope_key(scope)
    token = _MEMORY_TOKENS.get(scope_key)
    if token:
        return token
    token = _load_token_state(scope)
    if token:
        _MEMORY_TOKENS[scope_key] = token
    return token


def _scope(base_url: str) -> dict[str, str]:
    parsed = urlsplit(base_url.rstrip("/"))
    origin = f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme and parsed.netloc else base_url.rstrip("/")
    return {
        "origin": origin,
        "credentialFingerprint": _credential_fingerprint(),
        "region": os.getenv("AUTO_RESOURCE_REGION", "CN"),
    }


def _scope_key(scope: dict[str, str]) -> str:
    return json.dumps(scope, sort_keys=True, separators=(",", ":"))


def _credential_fingerprint() -> str:
    material = "\0".join([
        os.getenv("AUTO_RESOURCE_AUTH_TOKEN", "").strip(),
        os.getenv("AUTO_RESOURCE_REFRESH_TOKEN", "").strip(),
    ])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _refresh_headers() -> dict[str, str]:
    return {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Language": os.getenv("AUTO_RESOURCE_LANGUAGE", "zh-CN"),
        "Accept-Language": os.getenv("AUTO_RESOURCE_LANGUAGE", "zh-CN"),
        "X-Region": os.getenv("AUTO_RESOURCE_REGION", "CN"),
    }


def _token_state_path() -> Path:
    return Path(config.DATA_DIR) / _TOKEN_STATE_FILE


def _load_token_state(scope: dict[str, str]) -> ScopedToken | None:
    path = _token_state_path()
    try:
        raw = json.loads(path.read_text()) if path.exists() else {}
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None
    raw_scope = raw.get("scope")
    if not isinstance(raw_scope, dict):
        return None
    if {key: raw_scope.get(key) for key in scope} != scope:
        return None
    access = raw.get("accessToken")
    refresh = raw.get("refreshToken")
    if not isinstance(access, str) or not access.strip() or not isinstance(refresh, str) or not refresh.strip():
        return None
    return ScopedToken(access.strip(), refresh.strip(), scope)


def _save_token_state(token: ScopedToken) -> None:
    global _LAST_PERSIST_ERROR
    path = _token_state_path()
    tmp_name: str | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({
            "scope": token.scope,
            "accessToken": token.access_token,
            "refreshToken": token.refresh_token,
        }, ensure_ascii=False, separators=(",", ":"))
        fd, tmp_name = tempfile.mkstemp(prefix=f".{_TOKEN_STATE_FILE}.", suffix=".tmp", dir=str(path.parent))
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, path)
            os.chmod(path, 0o600)
            _LAST_PERSIST_ERROR = None
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            raise
    except Exception as exc:
        _LAST_PERSIST_ERROR = type(exc).__name__
        if tmp_name:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


def _business_ok(value: object) -> bool:
    if not isinstance(value, dict):
        return True
    for key in ("code", "statusCode"):
        if key in value and value.get(key) not in {0, "0", 200, "200"}:
            return False
    if value.get("success") is False:
        return False
    return True


def _find_value(value: object, key: str) -> object | None:
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for child in value.values():
            found = _find_value(child, key)
            if found is not None:
                return found
    if isinstance(value, list):
        for child in value:
            found = _find_value(child, key)
            if found is not None:
                return found
    return None
