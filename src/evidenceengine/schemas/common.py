"""Common error response schemas."""

from typing import Any

from pydantic import BaseModel


class ErrorDetail(BaseModel):
    """Per-file error detail (used by packets.py upload validation)."""

    file: str  # filename
    role: str  # "report" or "source"
    error: str  # human-readable error message


class ErrorResponse(BaseModel):
    """Error response body for validation/parse failures (used by packets.py)."""

    message: str
    errors: list[ErrorDetail]


# ──────────────────────────────────────────────────────────────────────────────
# New unified error types for Phase 5+ API routes
# ──────────────────────────────────────────────────────────────────────────────


class ErrorBody(BaseModel):
    """Inner error object: structured code + message + optional detail."""

    code: str
    message: str
    detail: Any | None = None


class ErrorEnvelope(BaseModel):
    """Standard error response envelope: {"error": {"code": ..., "message": ..., "detail": ...}}."""

    error: ErrorBody


class APIError(Exception):
    """Raise this in route handlers instead of HTTPException for structured error responses.

    Attributes:
        code: Machine-readable error code (e.g. "PACKET_NOT_FOUND").
        message: Human-readable error message.
        detail: Optional structured detail (dict, list, or string).
        status: HTTP status code (default 400).
    """

    def __init__(
        self,
        code: str,
        message: str,
        status: int = 400,
        detail: Any | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.detail = detail
