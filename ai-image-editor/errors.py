"""Application-level exceptions used across API and core services."""
from typing import Any, Dict, Optional


class AppError(Exception):
    """Base application error with optional machine-readable details."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 500,
        error_code: str = "internal_error",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}


class ValidationError(AppError):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            message,
            status_code=400,
            error_code="validation_error",
            details=details,
        )


class NotFoundError(AppError):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            message,
            status_code=404,
            error_code="not_found",
            details=details,
        )


class EmbeddingError(AppError):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            message,
            status_code=500,
            error_code="embedding_error",
            details=details,
        )


class StorageError(AppError):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            message,
            status_code=500,
            error_code="storage_error",
            details=details,
        )
