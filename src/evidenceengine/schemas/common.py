"""Common error response schemas."""

from pydantic import BaseModel


class ErrorDetail(BaseModel):
    """Per-file error detail."""

    file: str  # filename
    role: str  # "report" or "source"
    error: str  # human-readable error message


class ErrorResponse(BaseModel):
    """Error response body for validation/parse failures."""

    message: str
    errors: list[ErrorDetail]
