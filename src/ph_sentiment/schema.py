"""Common public-post fields without separate profile metadata.

Public post identifiers and provenance URLs can identify authors. These records
are attributable research data and must not be described as anonymous.
"""

from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

from .errors import DataValidationError

PLATFORMS = frozenset(
    {"reddit", "x", "threads", "facebook", "bluesky", "mastodon", "news", "blog", "other"}
)
LABELS = ("negative", "neutral", "positive")


def parse_timestamp(value: str) -> datetime:
    """Parse an ISO timestamp, requiring an explicit timezone for reproducibility."""
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, TypeError, AttributeError) as exc:
        raise DataValidationError(
            "created_at must be an ISO-8601 timestamp with timezone."
        ) from exc
    if result.tzinfo is None or result.utcoffset() is None:
        raise DataValidationError("created_at must include a timezone, e.g. 2026-06-01T12:00:00Z.")
    return result.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class Record:
    record_id: str
    platform: str
    created_at: str
    text: str
    source_url: str = ""
    query: str = ""
    record_type: str = "post"
    collected_at: str = ""
    gold_label: str = ""

    def validate(self) -> None:
        if any(not isinstance(value, str) for value in asdict(self).values()):
            raise DataValidationError("Record fields must be strings.")
        if not self.record_id.strip() or len(self.record_id) > 1024:
            raise DataValidationError("record_id must be nonempty and at most 1024 characters.")
        if self.platform not in PLATFORMS:
            raise DataValidationError(
                f"Unsupported platform. Choose: {', '.join(sorted(PLATFORMS))}."
            )
        if not self.text.strip() or len(self.text) > 100_000:
            raise DataValidationError("text must be nonempty and at most 100,000 characters.")
        parse_timestamp(self.created_at)
        if self.collected_at:
            parse_timestamp(self.collected_at)
        if self.record_type not in {"post", "comment", "article"}:
            raise DataValidationError("record_type must be post, comment, or article.")
        if self.gold_label and self.gold_label not in LABELS:
            raise DataValidationError("gold_label must be positive, negative, neutral, or blank.")
        if self.source_url:
            try:
                parts = urlsplit(self.source_url)
            except ValueError as exc:
                raise DataValidationError("source_url is malformed.") from exc
            if parts.scheme != "https" or not parts.hostname or parts.username or parts.password:
                raise DataValidationError("source_url must be an HTTPS URL without credentials.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


RECORD_FIELDS = tuple(field.name for field in fields(Record))
