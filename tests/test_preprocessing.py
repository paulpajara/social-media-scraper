import pytest

from ph_sentiment.errors import DataValidationError
from ph_sentiment.preprocessing import clean_text, prepare_records, tokenize
from ph_sentiment.schema import Record, parse_timestamp


def test_preserves_negation_hashtags_and_taglish():
    text = "@someone Hindi maganda! #Serbisyo &amp; trabaho https://example.com\n😊"
    cleaned = clean_text(text)
    assert cleaned == "hindi maganda! serbisyo & trabaho 😊"
    assert "hindi" in tokenize(cleaned)


def test_manila_calendar_boundaries_and_deduplication_audit():
    records = [
        Record("before", "reddit", "2025-12-31T15:59:59Z", "before the window"),
        Record("first", "reddit", "2025-12-31T16:00:00Z", "Maganda ang serbisyo!"),
        Record("repost", "x", "2026-01-01T00:00:00Z", "MAGANDA ANG SERBISYO https://x.com"),
        Record("end", "x", "2026-12-31T16:00:00Z", "end exclusive"),
        Record("bad", "reddit", "not a timestamp", "bad timestamp"),
        Record("removed", "reddit", "2026-02-01T00:00:00Z", "[removed]"),
    ]
    prepared, audit = prepare_records(records)
    assert [row.record_id for row in prepared] == ["first"]
    assert [row["reason"] for row in audit] == [
        "outside_date_window",
        "valid_unique",
        "duplicate_text",
        "outside_date_window",
        "invalid_record",
        "empty_or_removed_text",
    ]


def test_conflicting_duplicate_labels_need_adjudication():
    rows = [
        Record(str(index), "reddit", "2026-06-01T00:00:00Z", "same text", gold_label=label)
        for index, label in enumerate(["positive", "negative"])
    ]
    with pytest.raises(DataValidationError, match="conflicting reference labels"):
        prepare_records(rows)


def test_requires_timezone():
    with pytest.raises(DataValidationError, match="timezone"):
        parse_timestamp("2026-06-01T10:00:00")


def test_conflicting_same_id_labels_rejected():
    records = [
        Record("same-id", "reddit", "2026-06-01T00:00:00Z", "same text", gold_label=label)
        for label in ["positive", "negative"]
    ]
    with pytest.raises(DataValidationError, match="conflicting reference labels"):
        prepare_records(records)
