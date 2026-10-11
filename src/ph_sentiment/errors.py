"""Expected errors safe to display without credentials or HTTP response bodies."""


class PipelineError(Exception):
    """An actionable failure in the research pipeline."""


class ConfigurationError(PipelineError):
    """Invalid settings or missing credentials."""


class CollectionError(PipelineError):
    """An upstream API prevented collection."""


class DataValidationError(PipelineError):
    """The input cannot support a valid analysis."""
