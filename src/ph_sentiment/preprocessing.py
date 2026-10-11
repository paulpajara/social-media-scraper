"""Transparent preprocessing and deduplication, including a per-record CSV audit."""

import html
import re
import unicodedata
from dataclasses import replace
from datetime import UTC, date, datetime, time
from zoneinfo import ZoneInfo

from .errors import ConfigurationError, DataValidationError
from .schema import Record, parse_timestamp

_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_MENTION = re.compile(r"(?<![\w@])@[\w.]+")
_TOKEN = re.compile(
    r"[^\W_]+(?:['\u2019][^\W_]+)?|[\U0001F300-\U0001FAFF\u2600-\u27BF]", re.UNICODE
)
AUDIT_FIELDS = ("row_number", "record_id", "platform", "disposition", "reason")


def clean_text(text: str) -> str:
    """Keep punctuation, negation, hashtag words, and emoji; remove markup/URLs/mentions."""
    text = unicodedata.normalize("NFKC", html.unescape(text)).casefold()
    text = re.sub(r"<[^>]*>", " ", text)
    text = _URL.sub(" ", text)
    text = _MENTION.sub(" ", text)
    text = text.replace("#", "").replace("\u2019", "'")
    text = "".join(" " if unicodedata.category(char).startswith("C") else char for char in text)
    return " ".join(text.split())


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(clean_text(text))


def date_window(start_date: str, end_date: str) -> tuple[datetime, datetime]:
    try:
        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("Dates must use YYYY-MM-DD.") from exc
    if start >= end:
        raise ConfigurationError("start_date must precede the exclusive end_date.")
    zone = ZoneInfo("Asia/Manila")
    return (
        datetime.combine(start, time.min, zone).astimezone(UTC),
        datetime.combine(end, time.min, zone).astimezone(UTC),
    )


def prepare_records(
    records: list[Record], start_date: str = "2026-01-01", end_date: str = "2027-01-01"
) -> tuple[list[Record], list[dict]]:
    """Drop invalid/empty/out-of-window/duplicate records with explicit reasons."""
    start, end = date_window(start_date, end_date)
    identities: dict[tuple[str, str], str] = {}
    texts: dict[str, str] = {}
    retained: list[Record] = []
    audit: list[dict] = []
    for index, record in enumerate(records, start=1):
        reason = ""
        cleaned = clean_text(record.text) if isinstance(record.text, str) else ""
        try:
            record.validate()
        except DataValidationError:
            reason = "invalid_record"
        if not reason and not start <= parse_timestamp(record.created_at) < end:
            reason = "outside_date_window"
        if not reason and (cleaned in {"[deleted]", "[removed]"} or not tokenize(cleaned)):
            reason = "empty_or_removed_text"
        identity = (record.platform, record.record_id)
        # Token-normalized identity catches exact reposts differing only in punctuation or URLs.
        text_identity = " ".join(tokenize(cleaned))
        previous_labels = [identities.get(identity, ""), texts.get(text_identity, "")]
        if (
            not reason
            and record.gold_label
            and any(label and label != record.gold_label for label in previous_labels)
        ):
            raise DataValidationError(
                "Duplicate records have conflicting human labels; adjudicate them."
            )
        if not reason and identity in identities:
            reason = "duplicate_record_id"
        if not reason and text_identity in texts:
            previous_label = texts[text_identity]
            if previous_label and record.gold_label and previous_label != record.gold_label:
                raise DataValidationError(
                    "Duplicate texts have conflicting human labels; adjudicate them."
                )
            reason = "duplicate_text"
        if not reason:
            identities[identity] = record.gold_label
            texts[text_identity] = record.gold_label
            retained.append(
                replace(
                    record, text=cleaned, created_at=parse_timestamp(record.created_at).isoformat()
                )
            )
        audit.append(
            {
                "row_number": index,
                "record_id": record.record_id,
                "platform": record.platform,
                "disposition": "dropped" if reason else "retained",
                "reason": reason or "valid_unique",
            }
        )
    return retained, audit
