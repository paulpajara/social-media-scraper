"""Expected errors safe to display without credentials or HTTP response bodies."""


class PipelineError(Exception):
    """An actionable failure in the research pipeline."""


class ConfigurationError(PipelineError):
    """Invalid settings or missing credentials."""


class CollectionError(PipelineError):
    """An upstream API prevented collection."""


class HTTPCollectionError(CollectionError):
    """Credential-safe HTTP status and a fixed diagnostic category."""

    def __init__(self, status_code: int, category: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.category = category


class DataValidationError(PipelineError):
    """The input cannot support a valid analysis."""
