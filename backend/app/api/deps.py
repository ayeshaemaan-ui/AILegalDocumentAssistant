"""Shared FastAPI dependencies."""
from __future__ import annotations

from fastapi import Header, Request


async def client_key(
    request: Request,
    x_session_id: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> str:
    """The identity rate limits and budgets are keyed on.

    Preference order matters: an API key identifies a caller across devices, a
    session id identifies one browser tab, and the IP is the last resort. Using
    the IP alone would rate-limit an entire office as if it were one user.
    """
    if x_api_key:
        return f"key:{x_api_key[:24]}"
    if x_session_id:
        return f"session:{x_session_id}"
    client = request.client
    return f"ip:{client.host if client else 'unknown'}"
