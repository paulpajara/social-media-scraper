"""Reference audits cannot silently add, misjoin, or invent sentiment labels."""

from dataclasses import replace

import pytest

from ph_sentiment.curation import curate_records, text_digest
from ph_sentiment.errors import DataValidationError
from ph_sentiment.schema import Record


@pytest.fixture
def candidate():
    return Record("bluesky:test", "bluesky", "2026-09-01T00:00:00Z", "Philippine policy report")


def decision(record, **overrides):
    return {
        "index": "0",
        "record_id": record.record_id,
        "platform": record.platform,
        "relevance": "include",
        "label": "neutral",
        "reason": "Factual policy announcement.",
        "content_kind": "news_report",
        "context_needed": "no",
        "label_source": "ai_assisted",
        "annotator": "independent_ai_annotator",
        "annotation_date": "2026-10-12",
        "rubric_version": "ph_target_polarity_v1",
        "text_sha256": text_digest(record.text),
        **overrides,
    }


def test_curated_label_preserves_frozen_content_and_provenance(candidate):
    assert curate_records([candidate], [decision(candidate, label="negative")]) == [
        replace(candidate, gold_label="negative")
    ]


@pytest.mark.parametrize("status", ["exclude", "unresolved"])
def test_off_target_and_unclear_posts_are_not_neutral(candidate, status):
    assert curate_records([candidate], [decision(candidate, relevance=status, label="")]) == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"text_sha256": "bad"},
        {"label": ""},
        {"label": "unknown"},
        {"relevance": "exclude"},
        {"relevance": "unresolved"},
        {"relevance": "other"},
        {"context_needed": "yes"},
        {"label_source": "model_prediction"},
        {"reason": ""},
        {"annotator": ""},
        {"record_id": "another"},
        {"platform": "mastodon"},
    ],
)
def test_invalid_or_misjoined_audits_fail(candidate, overrides):
    with pytest.raises(DataValidationError):
        curate_records([candidate], [decision(candidate, **overrides)])


def test_audit_must_be_complete_and_unique(candidate):
    with pytest.raises(DataValidationError):
        curate_records([candidate], [])
    with pytest.raises(DataValidationError):
        curate_records([candidate], [decision(candidate), decision(candidate)])
    with pytest.raises(DataValidationError):
        curate_records([candidate, candidate], [decision(candidate)])
    with pytest.raises(DataValidationError):
        curate_records([replace(candidate, gold_label="neutral")], [decision(candidate)])
