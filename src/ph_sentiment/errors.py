"""Expected errors safe to display without credentials or HTTP response bodies."""

from pathlib import Path


class PipelineError(Exception):
    """An actionable failure in the research pipeline."""

    diagnostics_path: Path | None = None


class ConfigurationError(PipelineError):
    """Invalid settings or missing credentials."""


class CollectionError(PipelineError):
    """An upstream API prevented collection."""


class AccessCollectionError(CollectionError):
    """A verified access problem with a credential-safe diagnostic category."""

    def __init__(self, category: str, message: str) -> None:
        super().__init__(message)
        self.category = category


class HTTPCollectionError(AccessCollectionError):
    """Credential-safe HTTP status and a fixed diagnostic category."""

    def __init__(self, status_code: int, category: str, message: str) -> None:
        super().__init__(category, message)
        self.status_code = status_code


class DataValidationError(PipelineError):
    """The input cannot support a valid analysis."""
